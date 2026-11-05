"""Search the weights of a rank blend on validation scores, without retraining anything.

`train-merged` blends its members with equal weights (`src/models/ensemble.py::rank_blend`).
This module asks whether other weights rank drives better, using only the scores that
already exist. Everything here is validation-only; nothing reads the test split.

The drive-level metric the repo reports (`evaluation.drive_level_table`) collapses rows to
one score per drive: the maximum over the drive's rows, where a failing drive keeps only
its positive-window rows. Which rows count does not depend on the score, so `DriveRows`
works that out once and a candidate weight vector then costs one matrix product and one
segmented maximum, instead of a group-by over 15 million rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

#: False-alarm rates the search is judged at: low enough to be the regime the goal is about.
FAR_BUDGETS = (1e-4, 2e-4, 5e-4)


def simplex_grid(n_members: int, step: float) -> np.ndarray:
    """Every weight vector with entries that are multiples of `step` and sum to 1."""
    parts = round(1 / step)
    if abs(parts * step - 1) > 1e-9:
        raise ValueError(f"step {step} does not divide 1")
    rows: list[list[int]] = []

    def split(prefix: list[int], left: int, slots: int) -> None:
        if slots == 1:
            rows.append([*prefix, left])
            return
        for taken in range(left + 1):
            split([*prefix, taken], left - taken, slots - 1)

    split([], parts, n_members)
    return np.asarray(rows, dtype=np.float64) / parts


@dataclass(frozen=True)
class DriveRows:
    """The rows that count toward each drive's score, arranged for a fast segmented maximum."""

    keep: np.ndarray  # bool over the input rows
    order: np.ndarray  # sorts the kept rows by drive
    starts: np.ndarray  # first sorted kept row of each drive
    labels: np.ndarray  # 1 = failing drive, aligned with `starts`

    @classmethod
    def build(cls, drive_ids: np.ndarray, y: np.ndarray) -> DriveRows:
        codes = pl.Series(drive_ids).cast(pl.Categorical).to_physical().to_numpy()
        positive = np.asarray(y) == 1
        failing = np.zeros(int(codes.max()) + 1, dtype=bool)
        failing[codes[positive]] = True
        keep = ~failing[codes] | positive
        kept_codes = codes[keep]
        order = np.argsort(kept_codes, kind="stable")
        sorted_codes = kept_codes[order]
        starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
        return cls(keep, order, starts, failing[sorted_codes[starts]].astype(int))

    def arrange(self, rows: np.ndarray) -> np.ndarray:
        """`rows` (n_rows,) or (n_rows, k), cut to the kept rows in drive order."""
        return rows[self.keep][self.order]

    def drive_scores(self, arranged: np.ndarray) -> np.ndarray:
        """One score per drive: the maximum over its kept rows."""
        return np.maximum.reduceat(arranged, self.starts)


def recall_at_far(labels: np.ndarray, scores: np.ndarray, budgets=FAR_BUDGETS) -> list[float]:
    """Drive-level recall at each false-alarm-rate budget (the best point not above it)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return [float(tpr[np.searchsorted(fpr, b, side="right") - 1]) for b in budgets]


def objective(
    labels: np.ndarray, scores: np.ndarray, budgets: tuple[float, ...] = FAR_BUDGETS
) -> float:
    """What the search maximises: mean recall at the false-alarm `budgets`."""
    return float(np.mean(recall_at_far(labels, scores, budgets)))


def pipeline_criterion(
    labels: np.ndarray, scores: np.ndarray, *, target_precision: float, recall_floor: float
) -> dict[str, float | bool]:
    """The operating point `threshold.tune_drive_level_threshold` picks, from drive-level
    labels and scores: the highest-recall point meeting `target_precision` at recall >=
    `recall_floor`, else the highest-precision point with recall >= `recall_floor`."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    in_range = recall >= recall_floor
    if not in_range.any():
        in_range = np.ones_like(recall, dtype=bool)
    candidates = np.flatnonzero(in_range)
    meets = candidates[precision[candidates] >= target_precision]
    if len(meets):
        best, met = int(meets[np.argmax(recall[meets])]), True
    else:
        best, met = int(candidates[np.argmax(precision[candidates])]), False
    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "target_met": met,
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def bootstrap_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    budgets: tuple[float, ...] = FAR_BUDGETS,
) -> dict[str, float]:
    """Paired drive bootstrap of `objective(a) - objective(b)` and the standard error of
    `objective(a)`. Resamples whole drives, since a drive is the unit that fails."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs, a_values = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        if labels[pick].sum() == 0:
            continue
        a = objective(labels[pick], scores_a[pick], budgets)
        b = objective(labels[pick], scores_b[pick], budgets)
        diffs.append(a - b)
        a_values.append(a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se_a": float(np.std(a_values, ddof=1)),
        "n_boot": len(diffs),
    }
