"""Precision-first threshold tuning (docs/dataset_strategy.md section 15.2 /
docs/design_goal.md section 13). Select a threshold on the validation set
such that precision >= target, maximizing recall subject to that constraint.
If the target precision is unreachable, fall back to the highest-precision
threshold available and flag it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def tune_threshold_for_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    target_precision: float = 0.95,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1;
    # drop the last (threshold=inf) precision/recall point to align arrays.
    precision, recall = precision[:-1], recall[:-1]

    meets_target = precision >= target_precision
    if meets_target.any():
        candidate_idxs = np.flatnonzero(meets_target)
        best_idx = candidate_idxs[np.argmax(recall[candidate_idxs])]
        return {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "target_met": True,
        }

    # Target unreachable: fall back to the highest-precision threshold and
    # flag it so callers can require human review / block autonomous action.
    best_idx = int(np.argmax(precision))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "target_met": False,
    }
