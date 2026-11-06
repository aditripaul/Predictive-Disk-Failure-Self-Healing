# ruff: noqa: E501  (markdown table rows are written as single lines)
"""Cluster-based action policy: fit on validation, judge once on Q3.

    uv run python scripts/cluster_action_policy.py \\
        --run-dir frozen_runs/c459046e4f1645169151e7be25ff064d \\
        --scores data/tmp/lightgbm_scores_h30_h30_q1q2_q3.parquet \\
        [--validation-only] [--figures-only] [--overwrite]

Validation (no test row is read): the Stage 1 suspects of the frozen run, their
SMART signature on the day they first passed Stage 1, a clustering ablation
(k-means, Gaussian mixture, agglomerative, HDBSCAN, spectral) with the selection
rule of docs/cluster_action_policy.md, the tier x cluster matrix and the action
map. Writes `validation.json`, `cluster_model.json`, `action_map.json`.

Q3 (one recorded look): assign the Q3 suspects, rebuild the matrix, compare the
cluster policy with the tier-only policy per tier with paired bootstraps, check
cluster stability and apply the rule fixed in advance for building the agent
integration. Writes `q3_evaluation.json`.

`tables.md` and `figures/` are rewritten from the JSON each time;
`--figures-only` redraws them without reading any data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import adjusted_rand_score

from src.agent.cluster_policy import (
    BANDS,
    TIERS,
    add_outcomes,
    apply_policy,
    band_rates,
    cluster_shares,
    compare_policies,
    derive_action_map,
    drive_table,
    tier_matrix,
)
from src.models.access_log import _git_sha, record_test_access
from src.models.evaluation import drive_level_table
from src.models.failure_clusters import (
    SOURCE_COLUMNS,
    ClusterModel,
    Scaler,
    ablation,
    build_model,
    feature_columns,
    fit_labels,
    model_centroids,
    name_clusters,
    select,
    signature_frame,
)
from src.models.threshold import precision_targets_from_lift
from src.outputs import refuse_overwrite
from src.reporting import cluster_figures as cf
from src.reporting.evaluation_figures import Curve, operating_point, pr_curve
from src.reporting.plots import save_figure

MODEL_CONFIG_PATH = Path("configs/model.yaml")


_T0 = time.perf_counter()


def log(message: str) -> None:
    """Progress line with elapsed minutes, flushed so a long run can be watched."""
    print(f"[{(time.perf_counter() - _T0) / 60:6.1f} min] {message}", flush=True)


def load_thresholds(run_dir: Path) -> dict[str, float]:
    spec = json.loads((run_dir / "frozen_spec.json").read_text())
    two_stage = json.loads((run_dir / "two_stage.json").read_text())
    tiers = spec["action_tier_thresholds"]
    unreachable = 1.01
    return {
        "stage1": float(two_stage["candidate_threshold"]),
        **{t: float(tiers[t]) if tiers.get(t) is not None else unreachable for t in TIERS},
        "primary": float(spec["drive_threshold"]),
    }


def gold_rows(features: Path, pairs: pl.DataFrame) -> pl.DataFrame:
    """The gold row of each (drive_id, date) in `pairs`: signature columns and
    the drive family."""
    lo, hi = pairs["date"].min(), pairs["date"].max()
    return (
        pl.scan_parquet(features)
        .filter(pl.col("date").is_between(lo, hi))
        .select("drive_id", "date", "model_family", *SOURCE_COLUMNS)
        .join(pairs.lazy().unique(), on=["drive_id", "date"], how="inner")
        .collect(engine="streaming")
    )


def signatures_at(features: Path, drives: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Signature frame aligned with `drives`, read on each drive's `date_col`."""
    pairs = drives.select("drive_id", pl.col(date_col).alias("date"))
    rows = gold_rows(features, pairs)
    aligned = pairs.join(rows, on=["drive_id", "date"], how="left")
    return aligned.select("model_family").hstack(signature_frame(aligned))


def curve_dict(curve: Curve) -> dict[str, Any]:
    return {
        "label": curve.label,
        "precision": curve.precision.tolist(),
        "recall": curve.recall.tolist(),
    }


def curve_from(d: dict[str, Any]) -> Curve:
    return Curve(d["label"], np.asarray(d["precision"]), np.asarray(d["recall"]))


def leads_by_cluster(drives: pl.DataFrame) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for band in ("warn", "cordon"):
        part = drives.filter(
            (pl.col("band") == band) & pl.col("failing") & pl.col("lead_days").is_not_null()
        )
        out[band] = {
            str(c): part.filter(pl.col("cluster") == c)["lead_days"].cast(pl.Float64).to_list()
            for c in part["cluster"].unique().to_list()
        }
    return out


def family_shares(drives: pl.DataFrame, n_clusters: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    groups = [(f"C{i}", drives.filter(pl.col("monitor_cluster") == i)) for i in range(n_clusters)]
    for label, part in [*groups, ("all suspects", drives)]:
        counts = part.group_by("model_family").agg(pl.len().alias("n"))
        total = max(part.height, 1)
        out[label] = {str(r["model_family"]): r["n"] / total for r in counts.to_dicts()}
    return out


# --- validation ---------------------------------------------------------------------


def run_validation(
    args: argparse.Namespace,
    thresholds: dict[str, float],
    scores: pl.DataFrame,
    metadata: pl.DataFrame,
) -> dict[str, Any]:
    rows = scores.filter(pl.col("split") == "validation")
    n_drives = rows["drive_id"].n_unique()
    n_failing = rows.filter(pl.col("label") == 1)["drive_id"].n_unique()
    drives = add_outcomes(drive_table(rows, thresholds), metadata, args.data_end)
    log(f"validation: {drives.height:,} suspects of {n_drives:,} drives; reading gold rows")

    monitor_sig = signatures_at(args.features, drives, "monitor_date")
    decision_sig = signatures_at(args.features, drives, "decision_date")
    log("validation: signatures read")
    columns = feature_columns()
    x = monitor_sig.select(columns).to_numpy().astype(np.float64)
    scaler = Scaler.fit(x)
    z = scaler.transform(x)

    bands = drives["band"].to_numpy()
    failing = drives["failing"].to_numpy()
    halves = (drives.select(pl.col("drive_id").hash(seed=7) % 2).to_series().to_numpy()).astype(int)

    partial_path = args.out / "ablation_partial.json"
    fingerprint = {
        "suspects": drives.height,
        "ks": list(args.ks),
        "hdbscan_sizes": list(args.hdbscan_sizes),
        "seed": args.seed,
        "thresholds": thresholds,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    finished: list[dict[str, Any]] = []
    if partial_path.exists():
        saved = json.loads(partial_path.read_text())
        if saved["fingerprint"] == fingerprint:
            finished = saved["rows"]
            log(f"resuming the ablation: {len(finished)} candidates already done")

    def progress(done: int, total: int, row: dict[str, Any]) -> None:
        finished.append(row) if row not in finished else None
        partial_path.write_text(json.dumps({"fingerprint": fingerprint, "rows": finished}))
        log(
            f"ablation {done}/{total}: {row['algorithm']} {row['param']} -> {row['clusters']} "
            f"clusters, separation {row['separation']:.2e}, stability {row['stability_ari']:.2f}, "
            f"{row['seconds']} s"
        )

    rows_ablation = ablation(
        z,
        bands,
        failing,
        halves,
        scaler,
        columns,
        ks=args.ks,
        hdbscan_sizes=args.hdbscan_sizes,
        seed=args.seed,
        progress=progress,
        done=finished,
    )
    chosen = select(rows_ablation)
    log(f"chosen: {chosen['algorithm']} {chosen['param']} (rule met: {chosen['rule_met']})")
    labels, est = fit_labels(chosen["algorithm"], z, chosen["param"], seed=args.seed)
    model = build_model(chosen["algorithm"], chosen["param"], z, labels, est, scaler, columns)
    centroids = model_centroids(model, z)
    model.names, model.action_kinds = name_clusters(centroids, columns)

    drives = drives.with_columns(
        cluster=pl.Series(model.assign(decision_sig)).cast(pl.Int64),
        monitor_cluster=pl.Series(model.assign(monitor_sig)).cast(pl.Int64),
        model_family=decision_sig["model_family"],
    )
    rates = band_rates(drives)
    matrix = tier_matrix(drives, model.names)
    action_map = derive_action_map(matrix, rates, min_drives=args.min_drives)
    drives_p = apply_policy(drives, action_map)
    in_sample = compare_policies(
        drives_p, total_failing=n_failing, n_boot=args.n_boot, seed=args.seed
    )

    # Drive-level validation curve and the tier thresholds, for "how the tiers are set".
    lab, sc = drive_level_table(
        rows["drive_id"].to_numpy(), rows["label"].to_numpy(), rows["score"].to_numpy()
    )
    lifts = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["threshold"]["action_tier_lift"]
    base_rate = n_failing / n_drives
    targets = precision_targets_from_lift(lifts, base_rate)
    points = [
        operating_point(n, lab, sc, thresholds[n])
        for n in ("warn", "cordon", "migrate", "drain", "primary")
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cluster_model.json").write_text(json.dumps(model.to_dict(), indent=2))
    (args.out / "action_map.json").write_text(json.dumps(action_map, indent=2))
    return {
        "settings": {
            "run_dir": str(args.run_dir),
            "scores": str(args.scores),
            "thresholds": thresholds,
            "data_end": str(args.data_end),
            "min_drives": args.min_drives,
            "seed": args.seed,
            "ks": args.ks,
            "hdbscan_sizes": args.hdbscan_sizes,
            "git_commit": _git_sha(),
        },
        "drives": n_drives,
        "failing_drives": n_failing,
        "suspects": drives.height,
        "suspect_failing": int(failing.sum()),
        "ablation": rows_ablation,
        "chosen": chosen,
        "clusters": [
            {
                "cluster": i,
                "name": model.names[i],
                "action_kind": model.action_kinds[i],
                "share": float((drives["monitor_cluster"] == i).mean()),
                "failure_rate": float(
                    drives.filter(pl.col("monitor_cluster") == i)["failing"].mean() or 0
                ),
            }
            for i in range(model.n_clusters)
        ],
        "centroids": centroids.tolist(),
        "columns": columns,
        "family_shares": family_shares(drives, model.n_clusters),
        "band_rates": rates,
        "matrix": matrix.drop("band_order").to_dicts(),
        "action_map": action_map,
        "in_sample_comparison": in_sample,
        "leads": leads_by_cluster(drives),
        "tiers": {
            "lifts": lifts,
            "base_rate": base_rate,
            "targets": targets,
            "points": [(p.name, p.precision, p.recall) for p in points],
            "curve": curve_dict(pr_curve("drives (validation)", lab, sc)),
            "test_base_rate": None,
        },
    }


# --- Q3 -------------------------------------------------------------------------------


def run_test(
    args: argparse.Namespace,
    thresholds: dict[str, float],
    scores: pl.DataFrame,
    metadata: pl.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    model = ClusterModel.from_dict(json.loads((args.out / "cluster_model.json").read_text()))
    action_map = json.loads((args.out / "action_map.json").read_text())
    record_test_access(
        variant="cluster_action_policy",
        purpose="scripts/cluster_action_policy.py: tier x cluster policy against tier-only, Q3",
        extra={
            "run_dir": str(args.run_dir),
            "algorithm": model.algorithm,
            "clusters": model.n_clusters,
        },
    )
    rows = scores.filter(pl.col("split") == "test")
    n_drives = rows["drive_id"].n_unique()
    n_failing = rows.filter(pl.col("label") == 1)["drive_id"].n_unique()
    drives = add_outcomes(drive_table(rows, thresholds), metadata, args.data_end)
    monitor_sig = signatures_at(args.features, drives, "monitor_date")
    decision_sig = signatures_at(args.features, drives, "decision_date")
    drives = drives.with_columns(
        cluster=pl.Series(model.assign(decision_sig)).cast(pl.Int64),
        monitor_cluster=pl.Series(model.assign(monitor_sig)).cast(pl.Int64),
        model_family=decision_sig["model_family"],
    )
    matrix = tier_matrix(drives, model.names)
    policy = apply_policy(drives, action_map)
    comparison = compare_policies(
        policy, total_failing=n_failing, n_boot=args.n_boot, seed=args.seed
    )

    # Stability: shares and failure rates per cluster, and a refit on Q3 against the assignment.
    z_test = model.matrix(monitor_sig)
    labels, _ = fit_labels(
        model.algorithm, z_test, int(next(iter(model.params.values()))), seed=args.seed
    )
    refit_ari = float(adjusted_rand_score(labels, drives["monitor_cluster"].to_numpy()))
    test_shares = cluster_shares(drives["monitor_cluster"].to_numpy(), model.n_clusters)
    test_rates = [
        float(drives.filter(pl.col("monitor_cluster") == i)["failing"].mean())
        if (drives["monitor_cluster"] == i).any()
        else None
        for i in range(model.n_clusters)
    ]
    val_shares = [c["share"] for c in validation["clusters"]]
    share_ratio_ok = all(
        (vs == 0 and ts == 0) or (vs > 0 and ts > 0 and 0.5 <= ts / vs <= 2.0)
        for vs, ts in zip(val_shares, test_shares, strict=True)
    )
    # The rule fixed in advance (docs/cluster_action_policy.md section 6).
    checks = {}
    for tier in ("warn", "cordon"):
        p = comparison[tier].get("paired")
        checks[f"{tier}_precision_up"] = bool(p and p["difference_ci"][0] > 0)
        # "Not lower": the paired recall interval includes 0 or lies above it.
        checks[f"{tier}_recall_not_lower"] = bool(p and p["recall_difference_ci"][1] >= 0)
    checks["shares_within_2x"] = share_ratio_ok
    checks["refit_ari_ge_0_5"] = refit_ari >= 0.5
    checks["no_small_cell_moved"] = all(
        m["move"] == 0 or m["drives"] >= args.min_drives for m in action_map
    )
    lab, sc = drive_level_table(
        rows["drive_id"].to_numpy(), rows["label"].to_numpy(), rows["score"].to_numpy()
    )
    return {
        "drives": n_drives,
        "failing_drives": n_failing,
        "suspects": drives.height,
        "suspect_failing": int(drives["failing"].sum()),
        "band_rates": band_rates(drives),
        "matrix": matrix.drop("band_order").to_dicts(),
        "comparison": comparison,
        "stability": {
            "validation_shares": val_shares,
            "test_shares": test_shares,
            "validation_rates": [c["failure_rate"] for c in validation["clusters"]],
            "test_rates": test_rates,
            "refit_ari": refit_ari,
        },
        "rule": {"checks": checks, "passed": all(checks.values())},
        "leads": leads_by_cluster(drives),
        "curve": curve_dict(pr_curve("drives (Q3 test)", lab, sc)),
        "base_rate": n_failing / n_drives,
        "family_shares": family_shares(drives, model.n_clusters),
    }


# --- outputs -----------------------------------------------------------------------------


def figures(val: dict[str, Any], test: dict[str, Any] | None, out: Path) -> list[Path]:
    f = out / "figures"
    names = [c["name"] for c in val["clusters"]]
    moves = {(m["cluster"], m["band"]): m["move"] for m in val["action_map"]}
    counts = [round(c["share"] * val["suspects"]) for c in val["clusters"]]
    paths = [
        save_figure(
            cf.plot_ablation(val["ablation"], val["chosen"]), f / "clustering_ablation.png"
        ),
        save_figure(
            cf.plot_cluster_signatures(np.asarray(val["centroids"]), val["columns"], names, counts),
            f / "cluster_signatures.png",
        ),
        save_figure(
            cf.plot_cluster_families(val["family_shares"], names), f / "cluster_families.png"
        ),
        save_figure(
            cf.plot_tier_matrix(
                val["matrix"],
                names,
                val["band_rates"],
                moves,
                "Validation: failure rate per (cluster, band); red = cell the action map moves",
            ),
            f / "tier_cluster_matrix_validation.png",
        ),
        save_figure(
            cf.plot_tier_thresholds(
                curve_from(val["tiers"]["curve"]),
                val["tiers"]["points"],
                val["tiers"]["targets"],
                val["tiers"]["base_rate"],
            ),
            f / "tier_thresholds_validation.png",
        ),
        save_figure(
            cf.plot_lift_targets(
                val["tiers"]["lifts"],
                {
                    "validation": val["tiers"]["base_rate"],
                    **({"Q3 test": test["base_rate"]} if test else {}),
                },
            ),
            f / "tier_lift_targets.png",
        ),
    ]
    if test:
        paths += [
            save_figure(
                cf.plot_tier_matrix(
                    test["matrix"],
                    names,
                    test["band_rates"],
                    moves,
                    "Q3 test: failure rate per (cluster, band); red = cell the action map moves",
                ),
                f / "tier_cluster_matrix_test.png",
            ),
            save_figure(
                cf.plot_action_outcomes(test["comparison"]), f / "action_outcomes_by_cell.png"
            ),
            save_figure(cf.plot_policy_comparison(test["comparison"]), f / "policy_comparison.png"),
            save_figure(
                cf.plot_policy_pr(curve_from(test["curve"]), test["comparison"], test["base_rate"]),
                f / "policy_precision_recall.png",
            ),
            save_figure(cf.plot_lead_by_cluster(test["leads"], names), f / "lead_time_by_cell.png"),
            save_figure(
                cf.plot_cluster_stability(
                    names,
                    test["stability"]["validation_shares"],
                    test["stability"]["test_shares"],
                    test["stability"]["validation_rates"],
                    test["stability"]["test_rates"],
                ),
                f / "cluster_stability.png",
            ),
        ]
    return paths


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def tables(val: dict[str, Any], test: dict[str, Any] | None) -> str:
    th = val["settings"]["thresholds"]
    ch = val["chosen"]
    lines = [
        "# Cluster-based action policy (30-day run)",
        "",
        f"Run `{val['settings']['run_dir']}`; thresholds "
        + ", ".join(f"{k} {v:.3f}" for k, v in th.items())
        + f". Validation: {val['suspects']:,} suspects ({val['suspect_failing']} failing) of {val['drives']:,} drives.",
        "",
        "## Clustering ablation (validation suspects)",
        "",
        "| Algorithm | Param | Clusters | Separation | Stability (ARI) | Silhouette | Smallest cluster | Noise | Seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(val["ablation"], key=lambda r: (r["algorithm"], r["param"])):
        lines.append(
            f"| {r['algorithm']} | {r['param']} | {r['clusters']} | {r['separation']:.2e} | {r['stability_ari']:.2f} "
            f"| {r['silhouette']:.2f} | {r['smallest_cluster_share']:.1%} | {r['noise_share']:.1%} | {r['seconds']} |"
        )
    lines += [
        "",
        f"Chosen: **{ch['algorithm']}, {ch['clusters']} clusters** (rule {'met' if ch['rule_met'] else 'NOT met: most stable candidate taken'}).",
        "",
        "## Clusters",
        "",
        "| Cluster | Name (largest standardised features) | Action kind | Share of suspects | Failure rate |",
        "|---|---|---|---|---|",
    ]
    for c in val["clusters"]:
        lines.append(
            f"| C{c['cluster']} | {c['name']} | {c['action_kind']} | {c['share']:.1%} | {c['failure_rate']:.1%} |"
        )
    lines += [
        "",
        "Band failure rates (validation): "
        + ", ".join(f"{b} {_pct(val['band_rates'].get(b))}" for b in BANDS),
        "",
    ]
    for title, block in (("Validation", val), ("Q3 test", test)):
        if not block:
            continue
        lines += [
            f"## Tier x cluster matrix: {title}",
            "",
            "| Cluster | Band | Drives | Failing | Failure rate | Median lead (days) | Removed | Active | Top family |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for m in block["matrix"]:
            lead = m["median_lead_days"]
            lines.append(
                f"| C{m['cluster']} | {m['band']} | {m['drives']:,} | {m['failing']:,} | {m['failure_rate']:.1%} "
                f"| {'-' if lead is None else f'{lead:.0f}'} | {m['share_removed']:.1%} | {m['share_active']:.1%} | {m['top_family']} |"
            )
        lines.append("")
    lines += [
        "## Action map (fixed on validation)",
        "",
        "| Cell | Drives | Failure rate | Effective band | Reason |",
        "|---|---|---|---|---|",
    ]
    for m in val["action_map"]:
        arrow = {1: " (up)", -1: " (down)"}.get(int(np.sign(m["move"])), "")
        lines.append(
            f"| C{m['cluster']} / {m['band']} | {m['drives']:,} | {m['failure_rate']:.1%} | {m['effective_band']}{arrow} | {m['reason']} |"
        )
    for title, comp in (
        ("Validation, in sample", val["in_sample_comparison"]),
        ("Q3 test", test["comparison"] if test else None),
    ):
        if not comp:
            continue
        lines += [
            "",
            f"## Policy comparison: {title} (tiers counted cumulatively)",
            "",
            "| Tier | Tier-only P / R / flagged | Cluster P / R / flagged | Precision gap [95%] | Recall gap [95%] | Moved down (failing) | Moved up (failing) | Cost tier-only / cluster |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for tier in TIERS:
            c = comp[tier]
            a, b = c["tier_only"], c["cluster"]
            p = c.get("paired")
            gap = (
                f"{p['difference']:+.1%} [{p['difference_ci'][0]:+.1%}, {p['difference_ci'][1]:+.1%}]"
                if p
                else "-"
            )
            rgap = (
                f"{p['recall_difference']:+.1%} [{p['recall_difference_ci'][0]:+.1%}, {p['recall_difference_ci'][1]:+.1%}]"
                if p
                else "-"
            )
            lines.append(
                f"| {tier} | {_pct(a['precision'])} / {_pct(a['recall'])} / {a['flagged']:,} | {_pct(b['precision'])} / {_pct(b['recall'])} / {b['flagged']:,} "
                f"| {gap} | {rgap} | {c['moved_down']:,} ({c['moved_down_failing']}) | {c['moved_up']:,} ({c['moved_up_failing']}) | {a['cost']:,.0f} / {b['cost']:,.0f} |"
            )
    if test:
        s = test["stability"]
        lines += [
            "",
            "## Stability and the rule for the agent integration",
            "",
            "| Cluster | Validation share | Q3 share | Validation failure rate | Q3 failure rate |",
            "|---|---|---|---|---|",
        ]
        for i in range(len(val["clusters"])):
            lines.append(
                f"| C{i} | {s['validation_shares'][i]:.1%} | {s['test_shares'][i]:.1%} | {_pct(s['validation_rates'][i])} | {_pct(s['test_rates'][i])} |"
            )
        lines += ["", f"Refit on Q3 against the assignment: ARI {s['refit_ari']:.2f}.", ""]
        lines += ["| Check | Result |", "|---|---|"] + [
            f"| {k} | {'pass' if v else 'fail'} |" for k, v in test["rule"]["checks"].items()
        ]
        lines += [
            "",
            f"**Rule {'passed: build the agent integration' if test['rule']['passed'] else 'not passed: the agent integration is not built'}.**",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/gold/features/part.parquet"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/silver/drive_metadata/part.parquet")
    )
    parser.add_argument("--data-end", type=dt.date.fromisoformat, default=dt.date(2026, 9, 30))
    parser.add_argument("--min-drives", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ks", type=int, nargs="+", default=list(range(3, 11)))
    parser.add_argument("--hdbscan-sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--out", type=Path, default=Path("data/audit/cluster_policy_h30"))
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    val_path, test_path = args.out / "validation.json", args.out / "q3_evaluation.json"
    if args.figures_only:
        val = json.loads(val_path.read_text())
        test = json.loads(test_path.read_text()) if test_path.exists() else None
    else:
        thresholds = load_thresholds(args.run_dir)
        scores = pl.read_parquet(
            args.scores, columns=["split", "drive_id", "date", "label", "score"]
        )
        metadata = pl.read_parquet(args.metadata)
        if val_path.exists() and not args.overwrite:
            val = json.loads(val_path.read_text())  # the map is fixed; never refit it silently
        else:
            val = run_validation(args, thresholds, scores, metadata)
            val_path.write_text(json.dumps(val, indent=2, default=str))
        test = None
        if not args.validation_only:
            refuse_overwrite([test_path], overwrite=args.overwrite)
            test = run_test(args, thresholds, scores, metadata, val)
            test_path.write_text(json.dumps(test, indent=2, default=str))
    (args.out / "tables.md").write_text(tables(val, test))
    paths = figures(val, test, args.out)
    print((args.out / "tables.md").read_text())
    print(f"written to {args.out} ({len(paths)} figures)")


if __name__ == "__main__":
    main()
