"""Screen every SMART attribute in the raw Backblaze CSVs for predictive signal,
and test whether failures cluster by vault/pod.

The pipeline ingests 9 of the 186 SMART columns Backblaze publishes
(`src/ingest/backblaze.py`). Onboarding more is not free - each priority
attribute adds ~31 columns to the gold table - and a previous expansion of four
context attributes moved nothing. So pick them from measured separation on this
fleet, not from what the literature likes.

Two independent questions, two sections:

1. ATTRIBUTES. For every `smart_*_raw` and `smart_*_normalized` column: coverage
   (how often it is even populated) and single-attribute drive-day AUC for
   "fails within `--horizon` days". AUC near 0.5 means no univariate signal;
   far from 0.5 in either direction means signal (a low AUC is still useful,
   it just points the other way). This is a screen, not a model: an attribute
   can be useless alone and useful in combination, so treat a weak result as
   weak evidence rather than proof. Columns already ingested are marked, so the
   new candidates can be compared against the ones carrying the model today.

2. SPATIAL. Failures per vault and per pod, against the spread a no-clustering
   (binomial) null predicts. A variance-to-mean ratio well above 1 means
   failures concentrate in particular vaults/pods, which is the precondition
   for neighbour-failure features being worth building. At/below 1 means the
   pod/vault columns carry nothing and Tier 5 should be dropped.

Reads the raw CSVs directly - no pipeline stage required - and never holds more
than one day's selected columns at a time. Every failing drive's pre-failure
rows are kept; healthy drive-days are sampled (`--healthy-rate`) on a
deterministic hash of the serial number, so a rerun screens the same rows.

    uv run python scripts/screen_raw_attributes.py                  # all days
    uv run python scripts/screen_raw_attributes.py --days 30        # quick look
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

DEFAULT_RAW_GLOB = "data/raw/backblaze/*.csv"
DEFAULT_OUT = Path("data/audit/data_quality_reports/raw_attribute_screen.json")
#: What src/ingest/backblaze.py lands today, for side-by-side comparison.
INGESTED_IDS = (5, 7, 9, 187, 188, 194, 197, 198, 199)
SPATIAL_COLUMNS = ("datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num")


def _daily_files(limit: int | None, raw_glob: str) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(raw_glob))
    if not files:
        raise SystemExit(f"no raw CSVs matched {raw_glob}")
    # Spread a limited sample across the whole period rather than taking the
    # first N days, so a quick run still covers the fleet's later months.
    if limit is not None and limit < len(files):
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    return files


def _smart_columns(path: Path) -> list[str]:
    names = pl.scan_csv(path, n_rows=0).collect_schema().names()
    return [c for c in names if c.startswith("smart_")]


def failure_events(files: list[Path]) -> pl.DataFrame:
    """One row per drive that fails in the window: drive_id, failure_date.

    Scans three columns per file, so this stays cheap over a full quarter."""
    parts = []
    for path in files:
        part = (
            pl.scan_csv(path, try_parse_dates=True)
            .select(["date", "serial_number", "failure"])
            .filter(pl.col("failure").cast(pl.Int64, strict=False) == 1)
            .collect()
        )
        if part.height:
            parts.append(part)
    if not parts:
        return pl.DataFrame(
            schema={"serial_number": pl.String, "failure_date": pl.Date},
        )
    return (
        pl.concat(parts)
        .group_by("serial_number")
        .agg(pl.col("date").min().alias("failure_date"))
        .with_columns(pl.col("failure_date").cast(pl.Date))
    )


def _sample_rows(
    path: Path,
    smart_cols: list[str],
    failures: pl.DataFrame,
    *,
    horizon_days: int,
    healthy_rate: float,
    seed: int,
) -> pl.DataFrame | None:
    """One day's screen rows: every pre-failure row, plus a hash-sampled
    fraction of the healthy ones, labelled `y` (1 = fails within horizon)."""
    keep = int(round(healthy_rate * 10_000))
    lf = pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10_000)
    schema = lf.collect_schema().names()
    cols = [c for c in smart_cols if c in schema]
    lf = lf.select(["date", "serial_number", *cols]).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in cols]
    )
    day = lf.join(failures.lazy(), on="serial_number", how="left").with_columns(
        (
            pl.col("failure_date").is_not_null()
            & (pl.col("failure_date") >= pl.col("date"))
            & (pl.col("failure_date") <= pl.col("date") + pl.duration(days=horizon_days))
        ).alias("is_pre_failure")
    )
    # Drives that fail LATER than the horizon are neither positive nor a clean
    # negative, so they are dropped rather than diluting either class.
    sampled = day.filter(
        pl.col("is_pre_failure")
        | (
            pl.col("failure_date").is_null()
            & ((pl.col("serial_number").hash(seed=seed) % 10_000) < keep)
        )
    ).select(
        [
            pl.col("is_pre_failure").cast(pl.Int8).alias("y"),
            *[pl.col(c) for c in cols],
        ]
    )
    out = sampled.collect()
    return out if out.height else None


def screen_attributes(
    files: list[Path],
    failures: pl.DataFrame,
    *,
    horizon_days: int,
    healthy_rate: float,
    seed: int,
) -> tuple[list[dict], dict]:
    smart_cols = _smart_columns(files[0])
    frames = []
    for path in files:
        part = _sample_rows(
            path,
            smart_cols,
            failures,
            horizon_days=horizon_days,
            healthy_rate=healthy_rate,
            seed=seed,
        )
        if part is not None:
            frames.append(part)
    if not frames:
        raise SystemExit("no screen rows collected; check --healthy-rate and the CSVs")
    data = pl.concat(frames, how="diagonal")
    y = data["y"].to_numpy().astype(int)

    rows: list[dict] = []
    for col in smart_cols:
        if col not in data.columns:
            continue
        values = data[col].to_numpy().astype(float)
        present = ~np.isnan(values)
        n_present = int(present.sum())
        record: dict = {
            "column": col,
            "smart_id": int(col.split("_")[1]),
            "kind": "normalized" if col.endswith("_normalized") else "raw",
            "coverage": round(n_present / len(values), 4),
            "already_ingested": int(col.split("_")[1]) in INGESTED_IDS and col.endswith("_raw"),
        }
        # An attribute needs both classes and some variation among the rows
        # where it is populated before an AUC means anything.
        if n_present == 0 or len(np.unique(y[present])) < 2:
            record.update(auc=None, note="no usable rows")
        elif np.unique(values[present]).size < 2:
            record.update(auc=None, note="constant")
        else:
            auc = float(roc_auc_score(y[present], values[present]))
            record.update(
                auc=round(auc, 4),
                separation=round(abs(auc - 0.5), 4),
                nonzero_frac_positive=round(
                    float((values[present & (y == 1)] != 0).mean())
                    if (present & (y == 1)).any()
                    else float("nan"),
                    4,
                ),
                nonzero_frac_negative=round(
                    float((values[present & (y == 0)] != 0).mean())
                    if (present & (y == 0)).any()
                    else float("nan"),
                    4,
                ),
            )
        rows.append(record)

    rows.sort(key=lambda r: r.get("separation") or -1.0, reverse=True)
    meta = {
        "files_screened": len(files),
        "screen_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "horizon_days": horizon_days,
        "healthy_rate": healthy_rate,
        "seed": seed,
    }
    return rows, meta


def spatial_clustering(files: list[Path], failures: pl.DataFrame) -> dict:
    """Do failures concentrate in particular vaults/pods, beyond chance?

    Counts each drive once (its last seen placement), then compares the
    variance of per-group failure counts with the binomial variance a
    no-clustering null gives. Ratios well above 1 mean clustering."""
    first = pl.scan_csv(files[0], n_rows=0).collect_schema().names()
    available = [c for c in SPATIAL_COLUMNS if c in first]
    if not available:
        return {"available": [], "note": "no spatial columns in these CSVs"}

    parts = []
    for path in files:
        parts.append(
            pl.scan_csv(path, try_parse_dates=True)
            .select(["date", "serial_number", *available])
            .collect()
        )
    placement = (
        pl.concat(parts)
        .sort("date")
        .group_by("serial_number")
        .agg([pl.col(c).last().alias(c) for c in available])
        .join(
            failures.with_columns(pl.lit(1).alias("failed")).select(["serial_number", "failed"]),
            on="serial_number",
            how="left",
        )
        .with_columns(pl.col("failed").fill_null(0))
    )

    # `pod_id` turned out to be the pod INDEX within a vault (~20 values), not a
    # unique chassis, while `vault_id` matches a Backblaze vault (20 pods x ~60
    # drives). The pair therefore identifies a physical chassis - the grain at
    # which "the drives beside this one" actually means that.
    if "vault_id" in available and "pod_id" in available:
        placement = placement.with_columns(
            pl.concat_str([pl.col("vault_id"), pl.col("pod_id")], separator="/").alias("vault_pod")
        )
        available = [*available, "vault_pod"]

    overall = placement["failed"].mean()
    result: dict = {
        "available": available,
        "drives": int(placement.height),
        "failures": int(placement["failed"].sum()),
        "overall_failure_rate": round(float(overall or 0.0), 6),
        "groups": {},
    }
    for col in available:
        if col == "pod_slot_num":
            continue  # a slot index is not a grouping of drives; see below
        grouped = placement.group_by(col).agg(
            pl.len().alias("drives"), pl.col("failed").sum().alias("failures")
        )
        grouped = grouped.filter(pl.col("drives") >= 20)
        if grouped.height < 2:
            result["groups"][col] = {"note": "too few populated groups"}
            continue
        counts = grouped["failures"].to_numpy().astype(float)
        sizes = grouped["drives"].to_numpy().astype(float)
        observed_var = float(counts.var())
        expected_var = float((sizes * overall * (1 - overall)).mean())
        result["groups"][col] = {
            "groups": int(grouped.height),
            "median_drives_per_group": int(np.median(sizes)),
            "mean_failures_per_group": round(float(counts.mean()), 3),
            "observed_variance": round(observed_var, 3),
            "binomial_expected_variance": round(expected_var, 3),
            "variance_ratio": (round(observed_var / expected_var, 2) if expected_var > 0 else None),
            "max_group_failure_rate": round(float((counts / sizes).max()), 5),
            # With only a handful of very large groups, a high ratio mostly
            # reflects different drive models and age cohorts per group - which
            # the model already has features for - rather than local clustering.
            "interpretable": bool(grouped.height >= 30),
        }
    # Slot position is a within-chassis gradient, not a cohort: report the
    # failure rate by slot so a thermal/vibration trend would show up.
    if "pod_slot_num" in available:
        by_slot = (
            placement.group_by("pod_slot_num")
            .agg(pl.len().alias("drives"), pl.col("failed").sum().alias("failures"))
            .filter(pl.col("drives") >= 20)
            .with_columns((pl.col("failures") / pl.col("drives")).alias("rate"))
            .sort("pod_slot_num")
        )
        result["by_slot"] = by_slot.to_dicts()
    return result


def _print_attributes(rows: list[dict], meta: dict, top: int) -> None:
    print(
        f"\n== ATTRIBUTE SCREEN ({meta['files_screened']} days, "
        f"{meta['screen_rows']:,} rows, {meta['positive_rows']:,} pre-failure, "
        f"horizon={meta['horizon_days']}d)"
    )
    print(f"{'column':<28}{'cov':>7}{'AUC':>8}{'sep':>7}  {'nz+':>7}{'nz-':>7}  in?")
    scored = [r for r in rows if r.get("auc") is not None]
    for r in scored[:top]:
        print(
            f"{r['column']:<28}{r['coverage']:>7.2f}{r['auc']:>8.3f}"
            f"{r['separation']:>7.3f}  {r['nonzero_frac_positive']:>7.3f}"
            f"{r['nonzero_frac_negative']:>7.3f}  {'yes' if r['already_ingested'] else ''}"
        )
    unusable = [r for r in rows if r.get("auc") is None]
    print(f"\n  {len(scored)} scoreable, {len(unusable)} unusable (constant or empty)")
    print("  cov=fraction of rows populated, sep=|AUC-0.5|, nz+/nz-=non-zero share by class")


def _print_spatial(spatial: dict) -> None:
    print("\n== SPATIAL CLUSTERING")
    if not spatial.get("available"):
        print(f"  {spatial.get('note')}")
        return
    print(
        f"  {spatial['drives']:,} drives, {spatial['failures']:,} failures, "
        f"overall rate {spatial['overall_failure_rate']:.5f}"
    )
    for col, info in spatial["groups"].items():
        if "note" in info:
            print(f"  {col:<12} {info['note']}")
            continue
        print(
            f"  {col:<12} groups={info['groups']:<5} "
            f"median drives={info['median_drives_per_group']:<5} "
            f"var ratio={info['variance_ratio']}  "
            f"(observed {info['observed_variance']} vs binomial "
            f"{info['binomial_expected_variance']})"
            f"{'' if info.get('interpretable') else '  [too few groups to read]'}"
        )
    print("  variance ratio >> 1 means failures cluster -> neighbour features have signal")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=None, help="screen only N days")
    parser.add_argument("--horizon", type=int, default=30, help="failure horizon in days")
    parser.add_argument("--healthy-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=40, help="rows to print")
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--raw-glob", default=DEFAULT_RAW_GLOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    files = _daily_files(args.days, args.raw_glob)
    print(f"screening {len(files)} daily files: {files[0].name} .. {files[-1].name}")
    failures = failure_events(files)
    print(f"failing drives in window: {failures.height:,}")

    rows, meta = screen_attributes(
        files,
        failures,
        horizon_days=args.horizon,
        healthy_rate=args.healthy_rate,
        seed=args.seed,
    )
    _print_attributes(rows, meta, args.top)

    spatial = {} if args.skip_spatial else spatial_clustering(files, failures)
    if spatial:
        _print_spatial(spatial)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now(dt.UTC).isoformat(),
                "meta": meta,
                "attributes": rows,
                "spatial": spatial,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
