"""Evaluation metrics for disk-failure prediction (docs/dataset_strategy.md
section 15). AUPRC is the primary metric because failures are rare
(<1% of drive-days), making accuracy misleading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_scores))


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """docs/dataset_strategy.md section 15 "Precision at top-K risky
    drives": out of the k highest-scored rows, what fraction actually
    failed. A ranking-quality metric independent of any chosen threshold."""
    if k <= 0:
        raise ValueError("k must be positive")
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    k = min(k, len(y_scores))
    top_k_indices = np.argsort(-y_scores)[:k]
    return float(np.mean(y_true[top_k_indices]))


def precision_at_k_fractions(
    y_true: np.ndarray, y_scores: np.ndarray, fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
) -> dict[str, float]:
    """precision_at_k for the top `fraction * len(y_scores)` rows, for each
    fraction - e.g. `{"precision_at_top_1pct": ...}`."""
    n = len(y_scores)
    return {
        f"precision_at_top_{fraction * 100:g}pct": precision_at_k(
            y_true, y_scores, max(1, round(fraction * n))
        )
        for fraction in fractions
    }


def compute_calibration(
    y_true: np.ndarray, y_scores: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    """docs/dataset_strategy.md section 15 "Calibration: trustworthiness of
    predicted probabilities". Bins predictions into `n_bins` equal-width
    buckets over [0, 1] and compares each bucket's mean predicted
    probability to its observed fraction of positives; a well-calibrated
    model has these close together in every populated bin.
    `expected_calibration_error` is the bin-count-weighted mean absolute
    gap; `brier_score` is the mean squared error of the raw probabilities
    (lower is better, 0 is perfect)."""
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    n = len(y_scores)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_scores, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bins = []
    expected_calibration_error = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_record: dict[str, Any] = {
            "bin_lower": float(bin_edges[b]),
            "bin_upper": float(bin_edges[b + 1]),
            "count": count,
            "mean_predicted_probability": None,
            "observed_fraction_positive": None,
        }
        if count > 0:
            mean_predicted = float(y_scores[mask].mean())
            observed_fraction = float(y_true[mask].mean())
            bin_record["mean_predicted_probability"] = mean_predicted
            bin_record["observed_fraction_positive"] = observed_fraction
            expected_calibration_error += (count / n) * abs(mean_predicted - observed_fraction)
        bins.append(bin_record)

    return {
        "expected_calibration_error": expected_calibration_error,
        "brier_score": float(np.mean((y_scores - y_true) ** 2)),
        "bins": bins,
    }


def compute_warning_lead_time_days(
    df: pl.DataFrame,
    *,
    score_column: str,
    threshold: float,
    days_to_event_column: str = "days_to_event",
    drive_id_column: str = "drive_id",
) -> dict[str, Any]:
    """docs/dataset_strategy.md section 15 "Warning lead time: how early the
    model warns before failure". `df` must already be restricted to
    drive-days belonging to drives with a genuine failure event (e.g.
    filtered by the caller on `event_type` - see
    `src/labels/event_types.py::FAILURE_EVENT_TYPES`), with
    `days_to_event_column` = days until that drive's actual failure
    (`src/labels/labeling.py`, populated for every row of a failing drive
    regardless of label/horizon truncation). For each such drive, the lead
    time is the largest `days_to_event` among rows where the score first
    crossed `threshold` before the failure (`days_to_event > 0`) - i.e. how
    early the earliest warning came."""
    warned = df.filter((pl.col(score_column) >= threshold) & (pl.col(days_to_event_column) > 0))
    total_failed_drives = df[drive_id_column].n_unique() if df.height > 0 else 0

    if warned.height == 0:
        return {
            "mean_lead_time_days": None,
            "median_lead_time_days": None,
            "min_lead_time_days": None,
            "max_lead_time_days": None,
            "warned_drive_count": 0,
            "total_failed_drive_count": total_failed_drives,
            "warning_coverage": 0.0 if total_failed_drives else None,
        }

    lead_times = np.asarray(
        warned.group_by(drive_id_column)
        .agg(pl.col(days_to_event_column).max().alias("lead_time_days"))["lead_time_days"]
        .to_numpy(),
        dtype=float,
    )
    warned_drive_count = len(lead_times)

    return {
        "mean_lead_time_days": float(lead_times.mean()),
        "median_lead_time_days": float(np.median(lead_times)),
        "min_lead_time_days": float(lead_times.min()),
        "max_lead_time_days": float(lead_times.max()),
        "warned_drive_count": warned_drive_count,
        "total_failed_drive_count": total_failed_drives,
        "warning_coverage": (
            warned_drive_count / total_failed_drives if total_failed_drives else None
        ),
    }


def drive_level_table(
    drive_ids: np.ndarray, y_true: np.ndarray, y_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapses row-level labels/scores to one (label, score) pair per drive.

    A drive is *failing* if any of its rows has `label == 1` (it falls
    inside a failure's prediction horizon), otherwise *healthy*. Its score
    is the highest score over its positive-window rows for a failing drive
    - "was it flagged inside the warning window?" - and over all of its
    rows for a healthy one - "did it ever raise an alert?". Alerts a
    failing drive raises before its positive window are not counted for
    or against it. Returns `(drive_labels, drive_scores)`, aligned and
    ordered by drive_id (deterministic)."""
    df = pl.DataFrame({"drive_id": drive_ids, "y": y_true, "score": y_scores})
    failing = df.group_by("drive_id").agg((pl.col("y") == 1).any().alias("is_failing"))
    scored = (
        df.join(failing, on="drive_id")
        .filter(~pl.col("is_failing") | (pl.col("y") == 1))
        .group_by("drive_id")
        .agg(pl.col("score").max(), pl.col("is_failing").first())
        .sort("drive_id")
    )
    return (
        scored["is_failing"].to_numpy().astype(int),
        scored["score"].to_numpy().astype(float),
    )


def drive_level_metrics(
    drive_ids: np.ndarray, y_true: np.ndarray, y_scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """The model-goal metrics (precision >= 95%, recall 35-50%) at the
    level an operator experiences them: per DRIVE, not per drive-day. One
    failing drive contributes ~horizon near-identical positive rows, so
    row-level precision/recall over-count it (see `drive_level_table` for
    how a drive is labeled and scored). `precision` = caught failing
    drives / (caught + healthy drives that alerted); `recall` = caught /
    failing drives."""
    labels, scores = drive_level_table(drive_ids, y_true, y_scores)
    flagged = scores >= threshold
    failing = labels == 1
    caught = int((flagged & failing).sum())
    false_alarms = int((flagged & ~failing).sum())
    failing_count = int(failing.sum())
    healthy_count = int((~failing).sum())
    return {
        "failing_drive_count": failing_count,
        "healthy_drive_count": healthy_count,
        "caught_drive_count": caught,
        "false_alarm_drive_count": false_alarms,
        "precision": caught / (caught + false_alarms) if caught + false_alarms else 0.0,
        "recall": caught / failing_count if failing_count else 0.0,
        "false_alarm_rate": false_alarms / healthy_count if healthy_count else 0.0,
        "auprc": compute_auprc(labels, scores) if failing_count else 0.0,
    }


DEFAULT_RECALL_LEVELS = (0.05, 0.10, 0.20, 0.35)


def precision_at_recall_table(
    val_drive_ids: np.ndarray,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    test_drive_ids: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    *,
    recalls: tuple[float, ...] = DEFAULT_RECALL_LEVELS,
) -> list[dict[str, Any]]:
    """The model's drive-level precision at several recall levels, measured the
    way it would be deployed: for each recall level the threshold is the
    highest-precision one on VALIDATION that still catches at least that share
    of failing drives, and that threshold is then applied to TEST. One row per
    level, with the validation and test `drive_level_metrics`. `threshold` is
    None when validation cannot reach the recall level."""
    labels, scores = drive_level_table(val_drive_ids, y_val, val_scores)
    rows: list[dict[str, Any]] = []
    if labels.sum() == 0:
        return [{"target_recall": r, "threshold": None} for r in recalls]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    for target in recalls:
        ok = np.flatnonzero(recall >= target)
        if not len(ok):
            rows.append({"target_recall": target, "threshold": None})
            continue
        threshold = float(thresholds[ok[np.argmax(precision[ok])]])
        rows.append(
            {
                "target_recall": target,
                "threshold": threshold,
                "validation": drive_level_metrics(val_drive_ids, y_val, val_scores, threshold),
                "test": drive_level_metrics(test_drive_ids, y_test, test_scores, threshold),
            }
        )
    return rows


def evaluate_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "auprc": compute_auprc(y_true, y_scores),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "calibration": compute_calibration(y_true, y_scores),
        **precision_at_k_fractions(y_true, y_scores),
    }
