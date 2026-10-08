"""Precision-first threshold tuning (docs/dataset_strategy.md section 15.2 /
docs/design_goal.md section 13). Select a threshold on the validation set
such that precision >= target, maximizing recall subject to that constraint.
If the target precision is unreachable, fall back to the highest-precision
threshold available and flag it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import drive_level_table


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


def tune_drive_level_threshold(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    target_precision: float = 0.95,
    target_recall_range: tuple[float, float] = (0.35, 0.50),
) -> dict:
    """Picks the operating threshold on the DRIVE-level precision/recall
    curve (see `evaluation.drive_level_table`) - the level the model goal
    is stated at - honoring BOTH halves of it:

    1. If some threshold has drive recall >= the range's lower bound AND
       drive precision >= `target_precision`: take the one with the highest
       recall (`target_met=True`).
    2. Otherwise the goal is unreachable. Rather than falling back to the
       single most-confident drive (precision 100% at ~0% recall, useless
       for operations), report the best achievable point *inside the
       goal's recall range*: the highest precision among thresholds with
       recall >= the lower bound (`target_met=False`). The gap to the
       target is in the returned dict and the model card, and the target
       stays in config.

    `recall_floor_met` is False only if the range's lower bound is above
    anything reachable (more recall than the upper bound is never
    penalized)."""
    labels, scores = drive_level_table(drive_ids, y_true, y_scores)
    recall_low, recall_high = target_recall_range
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]

    in_range = recall >= recall_low
    if not in_range.any():  # no failing drives, or recall floor above what is reachable
        in_range = np.ones_like(recall, dtype=bool)
    candidates = np.flatnonzero(in_range)
    meets = candidates[precision[candidates] >= target_precision]
    if len(meets):
        best = int(meets[np.argmax(recall[meets])])
        target_met = True
    else:
        best = int(candidates[np.argmax(precision[candidates])])
        target_met = False
    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "target_met": target_met,
        "level": "drive",
        "target_precision": target_precision,
        "target_recall_range": [recall_low, recall_high],
        "recall_floor_met": bool(recall[best] >= recall_low),
        "precision_gap_to_target": max(0.0, target_precision - float(precision[best])),
    }


ACTION_TIERS = ("warn", "cordon", "migrate", "drain")

#: Sentinel threshold for a tier whose precision target is unreachable on
#: validation: probabilities never reach it, so the tier never fires.
UNREACHABLE_THRESHOLD = 1.01


def precision_targets_from_lift(
    lift_targets: dict[str, float], failure_rate: float
) -> dict[str, float]:
    """Precision target for each tier: lift x failure rate. The rate is that of
    the split being tuned (validation), so the conversion is exact there."""
    if not 0.0 < failure_rate < 1.0:
        raise ValueError(f"failure_rate must be in (0, 1), got {failure_rate}")
    return {tier: min(1.0, lift * failure_rate) for tier, lift in lift_targets.items()}


def tune_action_tiers(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    precision_targets: dict[str, float],
) -> dict[str, float | None]:
    """One score threshold per agent action tier (warn < cordon < migrate <
    drain), each chosen on the validation DRIVE-level curve as the
    highest-recall threshold whose drive precision is >= that tier's target.
    This replaces hand-picked probability cutoffs (configs/agent.yaml): the
    model's scores are class-weighted, so a score of 0.9 does not mean 90%
    of such drives fail - each tier's threshold is instead tied to the
    precision (the false-alarm cost) that tier's action can tolerate.

    `precision_targets` must be non-decreasing from warn to drain (a more
    drastic action needs at least as much precision); thresholds are
    forced non-decreasing too, whatever the noise in the curve. A tier
    whose target is unreachable gets `None` and should never fire."""
    targets = [precision_targets[tier] for tier in ACTION_TIERS]
    if any(b < a for a, b in zip(targets, targets[1:], strict=False)):
        raise ValueError(
            f"action tier precision targets must not decrease from warn to drain: "
            f"{dict(zip(ACTION_TIERS, targets, strict=True))}"
        )
    labels, scores = drive_level_table(drive_ids, y_true, y_scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]

    result: dict[str, float | None] = {}
    floor = float("-inf")
    for tier, target in zip(ACTION_TIERS, targets, strict=True):
        ok = np.flatnonzero(precision >= target)
        if not len(ok):
            result[tier] = None
            continue
        best = ok[np.argmax(recall[ok])]
        floor = max(floor, float(thresholds[best]))
        result[tier] = floor
    return result


def resolve_action_thresholds(
    tier_thresholds: dict[str, float | None] | None, fallback: dict[str, float]
) -> dict[str, float]:
    """The `action_thresholds` dict `determine_action_tier` expects: the
    trained model's per-tier thresholds, with an unreachable tier mapped to
    a threshold no probability reaches. `None` (no tiers recorded for the
    model) falls back to the configured thresholds."""
    if tier_thresholds is None:
        return dict(fallback)
    resolved: dict[str, float] = {}
    for tier in ACTION_TIERS:
        threshold = tier_thresholds.get(tier)
        resolved[tier] = UNREACHABLE_THRESHOLD if threshold is None else threshold
    return resolved
