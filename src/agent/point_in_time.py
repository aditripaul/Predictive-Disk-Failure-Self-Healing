"""Measuring the flagging policy the way it would actually run: at a date.

A deployed policy scores the drives that are reporting TODAY and is judged on
what happens to them over the next horizon. The first agent backtest did
neither (runbook 4.7): it scored each drive's latest in-window row, which for
a drive that left mid-window is its death day, and it gated on a maturity
label derived from the dataset's last date, which is the outcome. 61.8%
precision became 17.9% once measured at a fixed date.

This module is that measurement. A snapshot at date T is every drive with a
gold-feature row ON T, scored once, with the label at that same row - did it
fail within the horizon of T. Drives that had already gone are not in it,
because a fleet at T does not contain them.

Taken at several dates it also supports the one loop mechanism that could add
information rather than just strictness: a second, independent look. Flagging
a drive only when it scores above threshold at both T-lag and T is a policy
over two different days' telemetry, which is not the same experiment as a
persistence FEATURE inside one row's window (refuted three times). The
two-look score is `min(p_T-lag, p_T)`, so it ranks like any other score and is
compared with the one-look policy at matched false-alarm rate.

Two cautions that shape the statistics:

  * Snapshots a week apart share most of their drives, and a failing drive is
    positive in every snapshot within one horizon of its death (about four of
    them at a 30-day horizon, weekly). Observations are therefore NOT
    independent, and every interval here resamples whole drives.
  * Precision is a property of the base rate. Quote it with the base rate, and
    compare across runs in ROC space (recall at a matched false-alarm rate).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.agent.fleet import score_snapshot

TIERS = ("warn", "cordon", "migrate", "drain")
DEFAULT_FAR_BUDGETS = (1e-4, 2e-4, 5e-4, 1e-3, 3e-3)


def last_observable_date(data_end: dt.date, horizon_days: int) -> dt.date:
    """The last date whose label is observable: the data must outlive the
    horizon. Later dates have censored labels, so they would silently drop out
    of every count rather than fail loudly."""
    return data_end - dt.timedelta(days=horizon_days)


def select_snapshot_dates(start: dt.date, end: dt.date, step_days: int) -> list[dt.date]:
    """Snapshot dates `step_days` apart, anchored on `end` and walking back.

    Anchored on the latest observable date because it is the one a reader
    expects to see; the anchor is otherwise arbitrary. Ascending order."""
    if step_days < 1:
        raise ValueError("step_days must be at least 1")
    dates, day = [], end
    while day >= start:
        dates.append(day)
        day -= dt.timedelta(days=step_days)
    return sorted(dates)


def snapshot_observations(
    features_path: Path,
    labels_path: Path,
    model: Any,
    day: dt.date,
    *,
    horizon_days: int,
) -> pl.DataFrame:
    """`drive_id, date, p_fail, label` for every drive reporting on `day`
    that has an observable label there.

    The feature scan is FILTERED and then collected, never sliced. Slicing a
    filtered scan forces Polars to evaluate the filter across the whole scan
    to resolve row offsets, which materializes the window at full width
    (runbook 2.6); a plain filter-then-collect keeps only the matching rows.

    A drive without an observable label at `day` - removed rather than failed
    inside the horizon, or too close to the end of the data - is dropped
    rather than counted as a negative: an outcome that cannot be observed is
    not evidence that nothing happened."""
    fleet = pl.scan_parquet(features_path).filter(pl.col("date") == day).collect()
    if fleet.is_empty():
        return pl.DataFrame(
            schema={
                "drive_id": pl.String,
                "date": pl.Date,
                "p_fail": pl.Float64,
                "label": pl.Int8,
            }
        )
    scored = score_snapshot(fleet, model).select(["drive_id", "date", "p_fail"])
    del fleet
    labels = (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("horizon_days") == horizon_days)
            & (pl.col("date") == day)
            & pl.col("label").is_not_null()
        )
        .select(["drive_id", "label"])
        .collect()
    )
    return scored.join(labels, on="drive_id", how="inner")


# --- tier metrics --------------------------------------------------------


@dataclass(frozen=True)
class TierCounts:
    """Per-drive counts, so a cluster bootstrap is a weighted sum rather than a
    resample of millions of observations."""

    tiers: tuple[str, ...]
    flagged: np.ndarray  # (n_tiers, n_drives) observations at or above the tier
    caught: np.ndarray  # (n_tiers, n_drives) of those, the ones that failed
    positives: np.ndarray  # (n_drives,)
    negatives: np.ndarray  # (n_drives,)


def dense_drive_codes(obs: pl.DataFrame) -> np.ndarray:
    """Integer code per observation, 0..n_drives-1, shared by a drive's rows."""
    physical = obs["drive_id"].cast(pl.Categorical).to_physical().to_numpy()
    return np.unique(physical, return_inverse=True)[1]


def tier_counts(
    obs: pl.DataFrame, thresholds: dict[str, float], tiers: tuple[str, ...] = TIERS
) -> TierCounts:
    codes = dense_drive_codes(obs)
    n = int(codes.max()) + 1 if len(codes) else 0
    p = obs["p_fail"].to_numpy()
    failed = obs["label"].to_numpy() == 1
    flagged = np.zeros((len(tiers), n))
    caught = np.zeros((len(tiers), n))
    for i, tier in enumerate(tiers):
        above = p >= thresholds[tier]
        flagged[i] = np.bincount(codes, weights=above, minlength=n)
        caught[i] = np.bincount(codes, weights=above & failed, minlength=n)
    return TierCounts(
        tiers=tiers,
        flagged=flagged,
        caught=caught,
        positives=np.bincount(codes, weights=failed, minlength=n),
        negatives=np.bincount(codes, weights=~failed, minlength=n),
    )


def tier_metrics(counts: TierCounts, weights: np.ndarray | None = None) -> dict[str, dict]:
    """Precision, recall, false-alarm rate, likelihood ratio and lift per tier.

    `weights` are per-drive multiplicities; omit them for the point estimate."""
    w = np.ones(counts.positives.shape[0]) if weights is None else weights
    positives, negatives = float(counts.positives @ w), float(counts.negatives @ w)
    base_rate = positives / (positives + negatives) if positives + negatives else 0.0
    out: dict[str, dict] = {}
    for i, tier in enumerate(counts.tiers):
        flagged, caught = float(counts.flagged[i] @ w), float(counts.caught[i] @ w)
        precision = caught / flagged if flagged else 0.0
        recall = caught / positives if positives else 0.0
        far = (flagged - caught) / negatives if negatives else 0.0
        out[tier] = {
            "flagged": flagged,
            "caught": caught,
            "precision": precision,
            "recall": recall,
            "false_alarm_rate": far,
            "positive_likelihood_ratio": recall / far if far else None,
            "lift": precision / base_rate if base_rate else 0.0,
            "base_rate": base_rate,
        }
    return out


def bootstrap_tier_ci(
    counts: TierCounts, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, dict[str, tuple[float, float]]]:
    """Percentile intervals for precision and recall, resampling whole drives.

    A drive appears in every snapshot it was alive for and is positive in
    about four of them if it failed, so resampling observations would
    understate the uncertainty badly."""
    n_drives = counts.positives.shape[0]
    if n_drives == 0 or counts.positives.sum() == 0:
        raise ValueError("bootstrap needs at least one failing observation")
    rng = np.random.default_rng(seed)
    n_tiers = len(counts.tiers)
    precision = np.empty((n_boot, n_tiers))
    recall = np.empty((n_boot, n_tiers))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_drives, n_drives), minlength=n_drives).astype(float)
        metrics = tier_metrics(counts, w)
        for i, tier in enumerate(counts.tiers):
            precision[b, i], recall[b, i] = metrics[tier]["precision"], metrics[tier]["recall"]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        tier: {
            "precision": (
                float(np.percentile(precision[:, i], lo)),
                float(np.percentile(precision[:, i], hi)),
            ),
            "recall": (
                float(np.percentile(recall[:, i], lo)),
                float(np.percentile(recall[:, i], hi)),
            ),
        }
        for i, tier in enumerate(counts.tiers)
    }


# --- matched false-alarm rate, one look vs two ------------------------------


@dataclass(frozen=True)
class _FarCurve:
    """Everything about one scoring policy that does not depend on resampling:
    which observations are negatives, in descending score order. Per-resample
    work is then a cumulative sum of weights, not a sort."""

    neg_idx: np.ndarray
    neg_scores: np.ndarray
    pos_idx: np.ndarray
    pos_scores: np.ndarray


def _prepare_far(scores: np.ndarray, failed: np.ndarray) -> _FarCurve:
    neg_idx = np.flatnonzero(~failed)
    pos_idx = np.flatnonzero(failed)
    order = np.argsort(-scores[neg_idx], kind="stable")
    neg_idx = neg_idx[order]
    return _FarCurve(neg_idx, scores[neg_idx], pos_idx, scores[pos_idx])


def recall_at_far(
    curve: _FarCurve, obs_weights: np.ndarray, budgets: tuple[float, ...]
) -> list[float]:
    """Weighted recall at the strictest threshold whose false-alarm rate stays
    within each budget.

    Flags scores strictly above the first negative the budget cannot afford,
    so ties at the boundary are excluded and the budget is never exceeded."""
    cumulative = np.cumsum(obs_weights[curve.neg_idx])
    total_neg = cumulative[-1] if len(cumulative) else 0.0
    total_pos = float(obs_weights[curve.pos_idx].sum())
    if total_neg <= 0 or total_pos <= 0:
        return [0.0] * len(budgets)
    pos_w = obs_weights[curve.pos_idx]
    recalls = []
    for budget in budgets:
        k = int(np.searchsorted(cumulative, budget * total_neg, side="right"))
        if k >= len(curve.neg_scores):
            recalls.append(1.0)
            continue
        threshold = curve.neg_scores[k]
        recalls.append(float(pos_w[curve.pos_scores > threshold].sum() / total_pos))
    return recalls


def pair_observations(previous: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    """Drives alive and labelled on BOTH dates: `drive_id, p_prev, p_cur, label`
    with the label at the later date. A drive new at `current` has no earlier
    look, so it cannot be held to a two-look rule and is excluded from both
    arms - otherwise the comparison would differ in population as well as in
    policy."""
    return current.select(["drive_id", pl.col("p_fail").alias("p_cur"), "label"]).join(
        previous.select(["drive_id", pl.col("p_fail").alias("p_prev")]),
        on="drive_id",
        how="inner",
    )


def compare_looks(
    pairs: pl.DataFrame,
    *,
    budgets: tuple[float, ...] = DEFAULT_FAR_BUDGETS,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Recall at matched false-alarm rate: one look (`p_cur`) against two
    (`min(p_prev, p_cur)`), on the same drives, with a paired cluster-bootstrap
    interval on the difference.

    Both policies see identical drives and identical resamples, so shared
    noise cancels and the interval is on the quantity of interest. Recall at a
    matched FAR is base-rate free, which precision at a threshold is not."""
    if pairs.is_empty():
        return []
    failed = pairs["label"].to_numpy() == 1
    one = pairs["p_cur"].to_numpy()
    two = np.minimum(one, pairs["p_prev"].to_numpy())
    curve_one, curve_two = _prepare_far(one, failed), _prepare_far(two, failed)

    codes = dense_drive_codes(pairs)
    n_drives = int(codes.max()) + 1
    ones = np.ones(len(codes))
    base_one = recall_at_far(curve_one, ones, budgets)
    base_two = recall_at_far(curve_two, ones, budgets)

    rng = np.random.default_rng(seed)
    diffs = np.empty((n_boot, len(budgets)))
    for b in range(n_boot):
        drive_w = np.bincount(rng.integers(0, n_drives, n_drives), minlength=n_drives)
        w = drive_w[codes].astype(float)
        diffs[b] = np.array(recall_at_far(curve_two, w, budgets)) - np.array(
            recall_at_far(curve_one, w, budgets)
        )
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return [
        {
            "false_alarm_budget": budget,
            "one_look_recall": base_one[i],
            "two_look_recall": base_two[i],
            "difference": base_two[i] - base_one[i],
            "difference_ci": (
                float(np.percentile(diffs[:, i], lo)),
                float(np.percentile(diffs[:, i], hi)),
            ),
        }
        for i, budget in enumerate(budgets)
    ]
