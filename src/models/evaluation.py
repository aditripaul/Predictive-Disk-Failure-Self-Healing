"""Evaluation metrics for disk-failure prediction (docs/dataset_strategy.md
section 15). AUPRC is the primary metric because failures are rare
(<1% of drive-days), making accuracy misleading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix


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
    warned = df.filter(
        (pl.col(score_column) >= threshold) & (pl.col(days_to_event_column) > 0)
    )
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
