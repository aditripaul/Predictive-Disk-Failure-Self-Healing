"""Knowledge rules for the Plan node (docs/agent_knowledge_base.md section 7).

A rule takes the proposed tier, the drive's knowledge context (None without history), the
rule's parameters and the action thresholds, and returns a tier. A drive with no history is
never changed, and "no rule" (`rule_none`) is always a candidate. No rule is enabled in the
agent unless it passes the pre-registered screen (section 8,
`scripts/screen_agent_knowledge.py`).

For evaluation each tier rule is also a score transform (`apply_tier_rule`): a demotion
caps the score just below the threshold of the tier it lands in, a promotion raises it to
the threshold, so the existing point-in-time metrics apply to the transformed scores.
R1 (cohort prior) is a score transform by nature (`src/agent/family_calibration.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from data_contracts.schemas import ActionTier

#: Tiers in increasing severity, with the threshold key each one starts at.
LADDER: tuple[tuple[ActionTier, str | None], ...] = (
    (ActionTier.MONITOR, None),
    (ActionTier.WARN, "warn"),
    (ActionTier.CORDON, "cordon"),
    (ActionTier.MIGRATE, "migrate"),
    (ActionTier.DRAIN, "drain"),
)
ESCALATION = {ActionTier.CORDON, ActionTier.MIGRATE, ActionTier.DRAIN}
_BELOW = 1e-9

Rule = Callable[
    [ActionTier, Mapping[str, Any] | None, Mapping[str, Any], Mapping[str, float]], ActionTier
]


def tier_of(p: float, thresholds: Mapping[str, float]) -> ActionTier:
    """The score tier, before gates: the most severe tier whose threshold `p` reaches."""
    tier = ActionTier.MONITOR
    for candidate, key in LADDER[1:]:
        if key is not None and p >= thresholds[key]:
            tier = candidate
    return tier


def rule_none(tier, ctx, params, thresholds):  # noqa: ANN001, ARG001
    return tier


def rule_r2_chronic_demote(tier, ctx, params, thresholds):  # noqa: ANN001, ARG001
    """Cordon or above becomes warn when the drive's open episode has lasted at least N
    days and it spent at least N days above cordon in the context window."""
    n = int(params["n_days"])
    if (
        ctx is not None
        and tier in ESCALATION
        and (ctx.get("open_episode_days") or 0) >= n
        and (ctx.get("days_above_cordon_window") or 0) >= n
    ):
        return ActionTier.WARN
    return tier


def rule_r2b_first_flag_promote(tier, ctx, params, thresholds):  # noqa: ANN001
    """Warn becomes cordon for a drive's first flag: p_fail within delta of the cordon
    threshold, no earlier episode, at most one day above warn in the window, and at least
    `min_history_days` of history. Needs `p_fail` in params (the evaluator passes it)."""
    if (
        ctx is not None
        and tier == ActionTier.WARN
        and params["p_fail"] >= thresholds["cordon"] - float(params["delta"])
        and (ctx.get("prior_episodes") or 0) == 0
        and (ctx.get("days_above_warn_window") or 0) <= 1
        and (ctx.get("history_days") or 0) >= int(params.get("min_history_days", 60))
    ):
        return ActionTier.CORDON
    return tier


def rule_r3_outcome_memory(tier, ctx, params, thresholds):  # noqa: ANN001, ARG001
    """One band lower when the drive's last episode survived its horizon within M days."""
    m = int(params["m_days"])
    if (
        ctx is not None
        and tier != ActionTier.MONITOR
        and ctx.get("last_resolved_outcome") == "survived_horizon"
        and (
            ctx.get("days_since_last_resolved")
            if ctx.get("days_since_last_resolved") is not None
            else m + 1
        )
        <= m
    ):
        order = [t for t, _ in LADDER]
        return order[order.index(tier) - 1]
    return tier


RULES: dict[str, Rule] = {
    "none": rule_none,
    "r2_chronic_demote": rule_r2_chronic_demote,
    "r2b_first_flag_promote": rule_r2b_first_flag_promote,
    "r3_outcome_memory": rule_r3_outcome_memory,
}

#: The grids of section 7.1 (R1's tau grid is in the screen, with the calibrator).
GRIDS: dict[str, list[dict[str, Any]]] = {
    "r2_chronic_demote": [{"n_days": n} for n in (7, 14, 21, 28)],
    "r2b_first_flag_promote": [{"delta": d, "min_history_days": 60} for d in (0.05, 0.10)],
    "r3_outcome_memory": [{"m_days": m} for m in (14, 30, 60)],
}


def score_for_tier(p: float, tier: ActionTier, thresholds: Mapping[str, float]) -> float:
    """A score that lands in `tier`: `p` itself if it already does; otherwise just below
    the next tier's threshold (demotion) or at the tier's own threshold (promotion)."""
    if tier_of(p, thresholds) == tier:
        return p
    order = [t for t, _ in LADDER]
    i = order.index(tier)
    own, nxt = LADDER[i][1], LADDER[i + 1][1] if i + 1 < len(LADDER) else None
    lower = 0.0 if own is None else thresholds[own]
    upper = 1.01 if nxt is None else thresholds[nxt]
    return min(max(p, lower), upper - _BELOW)


def apply_tier_rule(
    rule: Rule,
    p_fail: np.ndarray,
    contexts: Sequence[Mapping[str, Any] | None],
    params: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """The rule as a score transform over observations (`contexts[i]` for `p_fail[i]`;
    None means no history and no change)."""
    out = np.asarray(p_fail, dtype=np.float64).copy()
    for i, (p, ctx) in enumerate(zip(out, contexts, strict=True)):
        if ctx is None:
            continue
        tier = tier_of(float(p), thresholds)
        new = rule(tier, ctx, {**params, "p_fail": float(p)}, thresholds)
        if new != tier:
            out[i] = score_for_tier(float(p), new, thresholds)
    return out
