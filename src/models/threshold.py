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


#: Which tier set the agent acts on: the lift tiers above or the cost tiers below
#: (configs/model.yaml threshold.tier_rule).
TIER_RULES = ("lift", "cost")


def tune_cost_tiers(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    cost_ratios: dict[str, float],
) -> dict[str, float | None]:
    """One score threshold per action tier, each maximising the tier's net value on the
    validation DRIVE-level table: `r * caught - false_alarms`, where `r` is how many
    unnecessary actions of that kind one prevented failure is worth
    (configs/model.yaml threshold.action_cost_ratio). Equivalently, drives are flagged
    while the marginal precision stays at or above `1 / (1 + r)`.

    The lift tiers (`tune_action_tiers`) place each tier at a precision target without
    any cost; these place it where the action pays for itself, so a cheap action sits at
    much higher recall than a drastic one. Ratios must not increase from warn to drain (a
    more drastic action costs more per false alarm), and thresholds are forced
    non-decreasing as for the lift tiers. A tier with no threshold of positive net value
    gets `None` and should never fire.

    The optimum depends on the failure rate of the split it is tuned on: on a test period
    with half the validation failure rate, a validation-tuned cost tier flags more than is
    optimal there."""
    ratios = [float(cost_ratios[tier]) for tier in ACTION_TIERS]
    if any(r <= 0 for r in ratios):
        raise ValueError(f"cost ratios must be positive: {cost_ratios}")
    if any(b > a for a, b in zip(ratios, ratios[1:], strict=False)):
        raise ValueError(
            f"cost ratios must not increase from warn to drain: "
            f"{dict(zip(ACTION_TIERS, ratios, strict=True))}"
        )
    labels, scores = drive_level_table(drive_ids, y_true, y_scores)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    sorted_labels = np.asarray(labels, dtype=int)[order]
    caught = np.cumsum(sorted_labels)
    false_alarms = np.cumsum(1 - sorted_labels)
    # A threshold flags every drive scoring at or above it, so only the last drive of
    # each run of tied scores is a reachable cut.
    cut = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    caught, false_alarms, cut_scores = caught[cut], false_alarms[cut], sorted_scores[cut]

    result: dict[str, float | None] = {}
    floor = float("-inf")
    for tier, ratio in zip(ACTION_TIERS, ratios, strict=True):
        value = ratio * caught - false_alarms
        best = int(np.argmax(value)) if len(value) else -1
        if best < 0 or value[best] <= 0:
            result[tier] = None
            continue
        floor = max(floor, float(cut_scores[best]))
        result[tier] = floor
    return result


def cost_ratios_from_costs(
    failure_cost: float,
    action_cost: dict[str, float],
    p_save: dict[str, float],
) -> dict[str, float]:
    """The ratio `r` per tier that `tune_cost_tiers` takes, from what operations can
    actually estimate (configs/model.yaml threshold.cost_model).

    An action on a drive that fails within the horizon is worth `p_save * failure_cost -
    action_cost` (it prevents the failure's cost with probability `p_save`, and is paid
    for either way); on a drive that does not, it costs `action_cost`. Net value over a
    flagged set is then `action_cost * (r * caught - false_alarms)` with

        r = p_save * failure_cost / action_cost - 1,

    so flagging pays while the marginal precision is at least `1 / (1 + r) = action_cost
    / (p_save * failure_cost)`. Raises if an action never pays (`r <= 0`) or if the
    ratios increase from warn to drain, which `tune_cost_tiers` would also reject: a
    more drastic action must not be cheaper per prevented failure."""
    if failure_cost <= 0:
        raise ValueError(f"failure_cost must be positive, got {failure_cost}")
    ratios: dict[str, float] = {}
    for tier in ACTION_TIERS:
        cost, p = float(action_cost[tier]), float(p_save[tier])
        if cost <= 0:
            raise ValueError(f"action_cost[{tier}] must be positive, got {cost}")
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p_save[{tier}] must be in (0, 1], got {p}")
        r = p * failure_cost / cost - 1.0
        if r <= 0:
            raise ValueError(
                f"{tier} never pays: p_save x failure_cost = {p * failure_cost:g} does not "
                f"exceed its action cost {cost:g}, so even a certain failure is not worth acting on"
            )
        ratios[tier] = r
    values = [ratios[t] for t in ACTION_TIERS]
    if any(b > a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(
            "derived cost ratios increase from warn to drain "
            f"({ {t: round(v, 3) for t, v in ratios.items()} }): a more drastic action would "
            "be cheaper per prevented failure than a milder one; check the inputs"
        )
    return ratios


def resolve_cost_ratios(threshold_cfg: dict[str, Any]) -> tuple[dict[str, float], str]:
    """`(ratios, source)`: derived from `threshold.cost_model` when it is filled in,
    otherwise the placeholder `threshold.action_cost_ratio`. A partly filled cost model
    raises rather than mixing real inputs with placeholders."""
    cost_model = threshold_cfg.get("cost_model") or {}
    failure_cost = cost_model.get("failure_cost")
    action_cost = cost_model.get("action_cost") or {}
    p_save = cost_model.get("p_save") or {}
    inputs = [failure_cost] + [action_cost.get(t) for t in ACTION_TIERS]
    inputs += [p_save.get(t) for t in ACTION_TIERS]
    if all(v is None for v in inputs):
        return {t: float(threshold_cfg["action_cost_ratio"][t]) for t in ACTION_TIERS}, (
            "action_cost_ratio (placeholders)"
        )
    if any(v is None for v in inputs):
        raise ValueError(
            "threshold.cost_model is partly filled: set failure_cost and every tier's "
            "action_cost and p_save, or leave all of them null"
        )
    return cost_ratios_from_costs(float(failure_cost), action_cost, p_save), "cost_model"


def _cut_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """`(cut_scores, caught, false_alarms)` at every reachable threshold, strictest first."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    s = np.asarray(scores, dtype=float)[order]
    lab = np.asarray(labels, dtype=int)[order]
    caught = np.cumsum(lab)
    false_alarms = np.cumsum(1 - lab)
    cut = np.r_[s[1:] != s[:-1], True]
    return s[cut], caught[cut], false_alarms[cut]


def _upper_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper convex hull of `(0, 0)` and the points `(x, y)` (x non-decreasing): the
    cuts that are cost-optimal for some ratio."""
    xs = np.r_[0, x].astype(float)
    ys = np.r_[0, y].astype(float)
    # At equal x keep the highest y (the last, as y is non-decreasing along the curve).
    keep = np.r_[xs[1:] != xs[:-1], True]
    xs, ys = xs[keep], ys[keep]
    hx: list[float] = []
    hy: list[float] = []
    for px, py in zip(xs, ys, strict=True):
        while len(hx) >= 2:
            cross = (hx[-1] - hx[-2]) * (py - hy[-2]) - (hy[-1] - hy[-2]) * (px - hx[-2])
            if cross >= 0:
                hx.pop()
                hy.pop()
            else:
                break
        hx.append(px)
        hy.append(py)
    return np.array(hx), np.array(hy)


def implied_cost_ratio(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """The range of `r` for which flagging every drive scoring `>= threshold` is the
    cost-optimal cut on this drive table - what a threshold set some other way (the lift
    tiers) implicitly assumes a prevented failure is worth, in actions of that kind.

    `labels` and `scores` are one row per drive (`evaluation.drive_level_table`). The
    optimum of `r * caught - false_alarms` always lies on the upper convex hull of the
    (false alarms, caught) curve; a hull vertex is optimal for `r` between the inverse
    slopes of its two hull segments. A cut off the hull is optimal for no `r`; it gets
    the inverse slope of the hull segment above it and `on_hull: False`. `r_high` is
    `inf` past the last vertex. Marginal precision `1 / (1 + r)` is reported beside it."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    caught = int((flagged & (labels == 1)).sum())
    false_alarms = int((flagged & (labels == 0)).sum())
    _, curve_caught, curve_fa = _cut_curve(labels, scores)
    hx, hy = _upper_hull(curve_fa, curve_caught)
    slopes = np.diff(hy) / np.diff(hx)  # caught per false alarm along the hull, decreasing

    def inv(s: float) -> float:
        return float("inf") if s <= 0 else 1.0 / s

    j = int(np.searchsorted(hx, false_alarms, side="right") - 1)
    j = max(0, min(j, len(hx) - 1))
    # A cut at a vertex's false-alarm count is that vertex if it catches as many, and
    # dominated by it otherwise; either way the vertex's range applies.
    at_vertex_x = hx[j] == false_alarms
    on_vertex = bool(at_vertex_x and hy[j] == caught)
    if at_vertex_x:
        r_low = 0.0 if j == 0 else inv(slopes[j - 1])
        r_high = inv(slopes[j]) if j < len(slopes) else float("inf")
    else:
        seg = min(j, len(slopes) - 1)
        r_low = r_high = inv(slopes[seg]) if len(slopes) else float("inf")
    return {
        "threshold": float(threshold),
        "caught": caught,
        "false_alarms": false_alarms,
        "on_hull": on_vertex,
        "r_low": r_low,
        "r_high": r_high,
        "marginal_precision_high": 1.0 / (1.0 + r_low),
        "marginal_precision_low": 0.0 if r_high == float("inf") else 1.0 / (1.0 + r_high),
    }


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
