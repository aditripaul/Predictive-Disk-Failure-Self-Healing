"""Precision-first threshold tuning (docs/dataset_strategy.md section 15.2 /
docs/design_goal.md section 13). Select a threshold on the validation set
such that precision >= target, maximizing recall subject to that constraint.
If the target precision is unreachable, fall back to the highest-precision
threshold available and flag it.
"""

from __future__ import annotations

from typing import Any

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


#: A family needs at least this many FAILING validation drives before it earns
#: its own thresholds. Below it, the family's drive-level curve is too coarse
#: to tune on - recall moves in steps of 1/n - and a threshold fitted to a
#: handful of drives is fitted to noise, so the family keeps the fleet-wide
#: thresholds instead.
DEFAULT_MIN_FAMILY_FAILING_DRIVES = 50


def tune_action_tiers_by_family(
    drive_ids: np.ndarray,
    drive_models: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    lift_targets: dict[str, float],
    fleet_thresholds: dict[str, float | None],
    min_failing_drives: int = DEFAULT_MIN_FAMILY_FAILING_DRIVES,
) -> dict[str, Any]:
    """One set of action-tier thresholds per drive model, tuned on that
    model's own validation drives, falling back to `fleet_thresholds`.

    A single fleet-wide threshold puts different drive models at completely
    different operating points, because their score distributions differ: on
    the 2026-10-09 build the pooled model at one threshold sat at 0.733
    precision / 0.129 recall on one model, 0.508 / 0.492 on another and
    0.360 / 0.158 on a third, against 0.394 / 0.177 fleet-wide (ADR 0002).
    Tuning per model equalises what each tier means instead of averaging it.

    Each model's thresholds come from the same `lift_targets` as the fleet's,
    converted to a precision target at THAT model's own validation failure
    rate - so a model whose drives fail more often needs correspondingly more
    precision for the same enrichment, and the tier keeps one meaning across
    a heterogeneous fleet.

    Tuning per model on few drives risks fitting the validation set rather
    than the model, which is why `min_failing_drives` gates it and why the
    result has to be judged on test (`pipelines/train_model.py` reports the
    per-family and fleet-wide thresholds side by side on test). Expect the
    gate, not the gain, to do most of the work here."""
    labels, _ = drive_level_table(drive_ids, y_true, y_scores)
    # drive_level_table collapses to one row per drive, sorted by drive id;
    # map each drive to its model the same way so the arrays line up.
    order = np.argsort(drive_ids, kind="stable")
    by_drive: dict[Any, Any] = {}
    for idx in order:
        by_drive.setdefault(drive_ids[idx], drive_models[idx])
    drive_order = np.array(sorted(by_drive))
    models_per_drive = np.array([by_drive[d] for d in drive_order])

    result: dict[str, Any] = {
        "min_failing_drives": min_failing_drives,
        "fleet_thresholds": dict(fleet_thresholds),
        "families": {},
        "skipped": {},
    }
    for model in np.unique(models_per_drive):
        in_family = models_per_drive == model
        failing = int(labels[in_family].sum())
        if failing < min_failing_drives:
            result["skipped"][str(model)] = {
                "failing_validation_drives": failing,
                "reason": "too few failing validation drives; keeps fleet thresholds",
            }
            continue
        rows = np.isin(drive_models, [model])
        family_failure_rate = float(labels[in_family].mean())
        if not 0.0 < family_failure_rate < 1.0:
            result["skipped"][str(model)] = {
                "failing_validation_drives": failing,
                "reason": "degenerate family failure rate",
            }
            continue
        precision_targets = precision_targets_from_lift(lift_targets, family_failure_rate)
        result["families"][str(model)] = {
            "failing_validation_drives": failing,
            "validation_failure_rate": family_failure_rate,
            "precision_targets": precision_targets,
            "thresholds": tune_action_tiers(
                drive_ids[rows],
                y_true[rows],
                y_scores[rows],
                precision_targets=precision_targets,
            ),
        }
    return result


def resolve_family_thresholds(plan: dict[str, Any], drive_model: str, tier: str) -> float | None:
    """The threshold `tier` should use for `drive_model`: that family's own if
    it earned one, otherwise the fleet-wide fallback. Keeping the lookup here
    means scoring, serving and evaluation cannot drift apart on the fallback
    rule."""
    family = plan.get("families", {}).get(str(drive_model))
    if family is not None and family["thresholds"].get(tier) is not None:
        return family["thresholds"][tier]
    return plan.get("fleet_thresholds", {}).get(tier)
