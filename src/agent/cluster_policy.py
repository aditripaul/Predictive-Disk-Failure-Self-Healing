"""Cluster-based action policy: the action is a function of (score tier, cluster)
instead of the tier alone (docs/cluster_action_policy.md).

Every drive is judged per DRIVE with `src/models/evaluation.py::drive_level_table`'s
convention, so the numbers line up with runbook 1.1.4: a failing drive (a positive
row in the window) is scored over its positive rows, a healthy drive over all its
rows. A drive's *band* is the highest tier its score reaches; its *decision day*
is the first day in that window its score reaches the band's threshold - the row
the agent would act on, and the row its cluster is read from.

The action map moves a (cluster, band) cell up or down at most one band, to the
highest band whose own validation failure rate the cell's rate matches or beats
(`derive_action_map`). Tier thresholds were tuned so the *cumulative* set at or
above a tier meets a precision target; a band on its own fails less often than
that set, so a cell is compared with band rates, not with the cumulative targets.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from src.models.uncertainty import paired_flag_difference_ci

#: Bands, lowest first. "monitor" = passed Stage 1, below warn.
BANDS = ("monitor", "warn", "cordon", "migrate", "drain")
TIERS = BANDS[1:]
OUTCOMES = ("correct", "early", "later", "removed", "active")


def band_of(score: pl.Expr, thresholds: dict[str, float]) -> pl.Expr:
    """Band name for a score, null below the Stage 1 threshold. Highest tier
    first, as `src/models/action_tiers.py::determine_action_tier` checks them."""
    return (
        pl.when(score >= thresholds["drain"])
        .then(pl.lit("drain"))
        .when(score >= thresholds["migrate"])
        .then(pl.lit("migrate"))
        .when(score >= thresholds["cordon"])
        .then(pl.lit("cordon"))
        .when(score >= thresholds["warn"])
        .then(pl.lit("warn"))
        .when(score >= thresholds["stage1"])
        .then(pl.lit("monitor"))
        .otherwise(pl.lit(None))
    )


def drive_table(rows: pl.DataFrame, thresholds: dict[str, float]) -> pl.DataFrame:
    """One row per suspect drive (drive score >= the Stage 1 threshold):
    `drive_id, failing, drive_score, band, decision_date, monitor_date`, where
    `monitor_date` is the first day the drive passed Stage 1 (the row the
    clusters are fitted on). `rows` needs `drive_id, date, label, score`."""
    positive = pl.col("label") == 1
    rows = rows.sort("drive_id", "date").with_columns(failing=positive.any().over("drive_id"))
    window = pl.when(pl.col("failing")).then(positive).otherwise(pl.lit(True))
    rows = rows.filter(window)
    aggs = [
        pl.col("failing").first(),
        pl.col("score").max().alias("drive_score"),
        pl.col("date").filter(pl.col("score") >= thresholds["stage1"]).min().alias("monitor_date"),
    ]
    for band in TIERS:
        aggs.append(
            pl.col("date").filter(pl.col("score") >= thresholds[band]).min().alias(f"first_{band}")
        )
    drives = (
        rows.group_by("drive_id").agg(aggs).filter(pl.col("drive_score") >= thresholds["stage1"])
    )
    drives = drives.with_columns(band=band_of(pl.col("drive_score"), thresholds))
    decision = pl.col("monitor_date")
    for band in TIERS:
        decision = pl.when(pl.col("band") == band).then(pl.col(f"first_{band}")).otherwise(decision)
    return drives.with_columns(decision_date=decision).sort("drive_id")


def add_outcomes(
    drives: pl.DataFrame, metadata: pl.DataFrame, data_end: dt.date, horizon: int = 30
) -> pl.DataFrame:
    """What happened to each drive after its decision day:
    `correct` (a failing drive: it failed within the horizon of a scored row),
    `early` (a healthy drive that failed within 2 x horizon of the decision day),
    `later` (failed after that), `removed` (left service with no recorded
    failure before `data_end`), `active` (still in service). `lead_days` is
    days from the decision day to failure for drives that failed."""
    meta = metadata.select("drive_id", "failure_date", "last_seen_date")
    out = drives.join(meta, on="drive_id", how="left")
    days = (pl.col("failure_date") - pl.col("decision_date")).dt.total_days()
    outcome = (
        pl.when(pl.col("failing"))
        .then(pl.lit("correct"))
        .when(pl.col("failure_date").is_not_null() & (days <= 2 * horizon))
        .then(pl.lit("early"))
        .when(pl.col("failure_date").is_not_null())
        .then(pl.lit("later"))
        .when(pl.col("last_seen_date") < data_end)
        .then(pl.lit("removed"))
        .otherwise(pl.lit("active"))
    )
    return out.with_columns(outcome=outcome, lead_days=days)


def band_index(col: str) -> pl.Expr:
    return pl.col(col).replace_strict({b: i for i, b in enumerate(BANDS)}, return_dtype=pl.Int8)


def band_rates(drives: pl.DataFrame) -> dict[str, float | None]:
    rates = drives.group_by("band").agg(pl.col("failing").mean().alias("r"), pl.len().alias("n"))
    lookup = {r["band"]: r["r"] for r in rates.to_dicts()}
    return {b: lookup.get(b) for b in BANDS}


def tier_matrix(drives: pl.DataFrame, cluster_names: list[str]) -> pl.DataFrame:
    """Per (cluster, band) cell: drives, failing, failure rate, median lead (days
    from the decision day to failure, failing drives), outcome shares and the
    most common drive family. Bands are exclusive here (a drive sits in its
    highest band); the policy comparison counts tiers cumulatively."""
    cells = (
        drives.group_by("cluster", "band")
        .agg(
            pl.len().alias("drives"),
            pl.col("failing").sum().alias("failing"),
            pl.col("failing").mean().alias("failure_rate"),
            pl.col("lead_days").filter(pl.col("failing")).median().alias("median_lead_days"),
            *[(pl.col("outcome") == o).mean().alias(f"share_{o}") for o in OUTCOMES],
            pl.col("model_family").drop_nulls().mode().first().alias("top_family"),
        )
        .with_columns(
            cluster_name=pl.col("cluster").map_elements(
                lambda c: cluster_names[c], return_dtype=pl.String
            ),
            band_order=band_index("band"),
        )
        .sort("cluster", "band_order")
    )
    return cells


def derive_action_map(
    matrix: pl.DataFrame, rates: dict[str, float | None], *, min_drives: int = 30, max_step: int = 1
) -> list[dict[str, Any]]:
    """The rule fixed before any test row is read. For each cell with at least
    `min_drives` validation drives: the effective band is the highest band
    whose own validation failure rate the cell's failure rate matches or beats
    (bands with no drives are skipped), capped at `max_step` bands from the
    score band; a cell beating no band rate goes to "monitor". Cells below
    `min_drives` keep their band."""
    present = [(i, b, rates[b]) for i, b in enumerate(BANDS) if rates.get(b) is not None]
    out = []
    for cell in matrix.to_dicts():
        score_idx = BANDS.index(cell["band"])
        if cell["drives"] < min_drives:
            eff_idx, reason = score_idx, f"fewer than {min_drives} drives"
        else:
            candidates = [i for i, _, r in present if cell["failure_rate"] >= r]
            target = max(candidates) if candidates else 0
            eff_idx = int(np.clip(target, score_idx - max_step, score_idx + max_step))
            reason = (
                "matches band rate"
                if eff_idx == score_idx
                else (
                    "fails more often than its band"
                    if eff_idx > score_idx
                    else "fails less often than its band"
                )
            )
        out.append(
            {
                "cluster": int(cell["cluster"]),
                "cluster_name": cell["cluster_name"],
                "band": cell["band"],
                "effective_band": BANDS[eff_idx],
                "move": eff_idx - score_idx,
                "drives": int(cell["drives"]),
                "failure_rate": float(cell["failure_rate"]),
                "reason": reason,
            }
        )
    return out


def apply_policy(drives: pl.DataFrame, action_map: list[dict[str, Any]]) -> pl.DataFrame:
    """Adds `effective_band`: the map's band for the drive's (cluster, band)
    cell, or its own band for a cell the map does not list."""
    lookup = pl.DataFrame(
        [
            {"cluster": m["cluster"], "band": m["band"], "effective_band": m["effective_band"]}
            for m in action_map
        ],
        schema={"cluster": pl.Int64, "band": pl.String, "effective_band": pl.String},
    )
    out = drives.with_columns(pl.col("cluster").cast(pl.Int64)).join(
        lookup, on=["cluster", "band"], how="left"
    )
    return out.with_columns(pl.col("effective_band").fill_null(pl.col("band")))


def _flag_metrics(
    failing: np.ndarray,
    flagged: np.ndarray,
    lead: np.ndarray,
    total_failing: int,
    cost_ratio: float,
) -> dict[str, Any]:
    caught = int((flagged & failing).sum())
    fa = int((flagged & ~failing).sum())
    return {
        "flagged": int(flagged.sum()),
        "caught": caught,
        "false_alarms": fa,
        "precision": caught / (caught + fa) if caught + fa else None,
        "recall": caught / total_failing if total_failing else None,
        "recall_lead_7": float(((flagged & failing) & (lead >= 7)).sum() / total_failing)
        if total_failing
        else None,
        "median_lead_days": float(np.median(lead[flagged & failing])) if caught else None,
        "cost": float((total_failing - caught) * cost_ratio + fa),
    }


def compare_policies(
    drives: pl.DataFrame,
    *,
    total_failing: int,
    n_boot: int = 1000,
    seed: int = 0,
    cost_ratio: float = 20.0,
) -> dict[str, Any]:
    """Per tier, counted cumulatively (a drive at cordon counts at warn): the
    tier-only policy (`band`) against the cluster policy (`effective_band`) on
    the same suspect drives, with a paired drive bootstrap of the precision and
    recall gaps. `total_failing` is every failing drive in the split, suspects
    or not, so recall matches runbook 1.1.4."""
    idx_a = drives.select(band_index("band")).to_series().to_numpy()
    idx_b = drives.select(band_index("effective_band")).to_series().to_numpy()
    failing = drives["failing"].to_numpy()
    lead = drives["lead_days"].fill_null(-1).to_numpy()
    outcome = drives["outcome"].to_numpy()
    # Failing drives Stage 1 never passed: in the denominator, flagged by neither.
    missing = total_failing - int(failing.sum())
    labels_full = np.concatenate([failing.astype(int), np.ones(missing, dtype=int)])
    out: dict[str, Any] = {}
    for t, tier in enumerate(TIERS, start=1):
        fa_, fb_ = idx_a >= t, idx_b >= t
        entry = {
            "tier_only": _flag_metrics(failing, fa_, lead, total_failing, cost_ratio),
            "cluster": _flag_metrics(failing, fb_, lead, total_failing, cost_ratio),
            "moved_down": int((fa_ & ~fb_).sum()),
            "moved_down_failing": int((fa_ & ~fb_ & failing).sum()),
            "moved_up": int((~fa_ & fb_).sum()),
            "moved_up_failing": int((~fa_ & fb_ & failing).sum()),
            "outcomes": {
                policy: {o: int(((flags) & (outcome == o)).sum()) for o in OUTCOMES}
                for policy, flags in (("tier_only", fa_), ("cluster", fb_))
            },
        }
        if fa_.any() and fb_.any():
            pad = np.zeros(missing, dtype=bool)
            entry["paired"] = paired_flag_difference_ci(
                labels_full,
                np.concatenate([fa_, pad]),
                np.concatenate([fb_, pad]),
                n_boot=n_boot,
                seed=seed,
            )
        out[tier] = entry
    return out


def cluster_shares(clusters: np.ndarray, n_clusters: int) -> list[float]:
    return (np.bincount(clusters, minlength=n_clusters) / max(len(clusters), 1)).tolist()
