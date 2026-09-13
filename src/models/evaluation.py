"""Evaluation metrics for disk-failure prediction (docs/dataset_strategy.md
section 15). AUPRC is the primary metric because failures are rare
(<1% of drive-days), making accuracy misleading.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_scores))


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
    }
