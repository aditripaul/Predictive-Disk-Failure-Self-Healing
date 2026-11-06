"""Drive-level bootstrap confidence intervals for threshold metrics.

Resamples whole drives (not drive-days), so a drive's many near-identical rows
move together, matching how `drive_level_metrics` counts them. A point estimate
without an interval cannot tell a real gain from seed or sampling noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.evaluation import drive_level_table


def drive_level_bootstrap_ci(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CIs for precision and recall at `threshold`, per drive."""
    labels, scores = drive_level_table(drive_ids, y_true, y_scores)
    flagged = (scores >= threshold).astype(np.int64)
    failing = (labels == 1).astype(np.int64)
    n = len(labels)
    if n == 0 or failing.sum() == 0:
        raise ValueError("bootstrap needs at least one failing drive")

    rng = np.random.default_rng(seed)
    precisions = np.empty(n_boot)
    recalls = np.empty(n_boot)
    for b in range(n_boot):
        # Each resampled drive is one draw; counts come from a bincount so the
        # memory cost is one length-n array, not n_boot x n.
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        caught = float((counts * flagged * failing).sum())
        alarms = float((counts * flagged).sum())
        failing_total = float((counts * failing).sum())
        precisions[b] = caught / alarms if alarms else 0.0
        recalls[b] = caught / failing_total if failing_total else 0.0

    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    point_caught = int((flagged & failing).sum())
    point_alarms = int(flagged.sum())
    return {
        "threshold": float(threshold),
        "n_boot": n_boot,
        "confidence": 1 - alpha,
        "precision": point_caught / point_alarms if point_alarms else 0.0,
        "precision_ci": [
            float(np.percentile(precisions, low)),
            float(np.percentile(precisions, high)),
        ],
        "recall": point_caught / int(failing.sum()),
        "recall_ci": [float(np.percentile(recalls, low)), float(np.percentile(recalls, high))],
    }


def paired_difference_ci(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """CI for precision(B) - precision(A) on the SAME resampled drives, so the
    shared sampling noise cancels. Use this for "does the second stage help?"
    rather than comparing two independent intervals."""
    labels, sa = drive_level_table(drive_ids, y_true, scores_a)
    _, sb = drive_level_table(drive_ids, y_true, scores_b)
    return paired_flag_difference_ci(
        labels, sa >= threshold_a, sb >= threshold_b, n_boot=n_boot, seed=seed, alpha=alpha
    )


def paired_flag_difference_ci(
    labels: np.ndarray,
    flagged_a: np.ndarray,
    flagged_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """`paired_difference_ci` on drive-level arrays already built (one entry
    per drive, aligned): precision(B) - precision(A) and recall(B) - recall(A)
    on the same resampled drives. A caller comparing many thresholds builds
    the drive tables once instead of once per comparison. The same `seed` and
    drive count give the same resamples, so comparisons sharing a seed are
    paired with each other too."""
    failing = (np.asarray(labels) == 1).astype(np.int64)
    fa = np.asarray(flagged_a).astype(np.int64)
    fb = np.asarray(flagged_b).astype(np.int64)
    n = len(failing)
    if n == 0 or failing.sum() == 0:
        raise ValueError("bootstrap needs at least one failing drive")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    recall_diffs = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        diffs[b] = _precision(counts, fb, failing) - _precision(counts, fa, failing)
        failing_total = float((counts * failing).sum())
        recall_diffs[b] = (
            float((counts * (fb - fa) * failing).sum()) / failing_total if failing_total else 0.0
        )
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    failing_count = int(failing.sum())
    return {
        "difference": _precision_point(fb, failing) - _precision_point(fa, failing),
        "difference_ci": [float(np.percentile(diffs, low)), float(np.percentile(diffs, high))],
        "recall_difference": float(((fb - fa) * failing).sum()) / failing_count,
        "recall_difference_ci": [
            float(np.percentile(recall_diffs, low)),
            float(np.percentile(recall_diffs, high)),
        ],
        "n_boot": n_boot,
        "confidence": 1 - alpha,
    }


def _precision(counts: np.ndarray, flagged: np.ndarray, failing: np.ndarray) -> float:
    alarms = float((counts * flagged).sum())
    return float((counts * flagged * failing).sum()) / alarms if alarms else 0.0


def _precision_point(flagged: np.ndarray, failing: np.ndarray) -> float:
    alarms = int(flagged.sum())
    return float((flagged & failing).sum()) / alarms if alarms else 0.0
