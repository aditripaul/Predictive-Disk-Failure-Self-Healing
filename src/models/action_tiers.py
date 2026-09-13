"""Action-tier policy: maps a prediction plus feature confidence into an
action tier (docs/design_goal.md section 13, docs/dataset_strategy.md
section 14.3).

destructive_action_allowed =
    p_fail >= high_threshold
    AND feature_confidence >= confidence_threshold
    AND feature_maturity == MATURE
    AND telemetry is not stale
    AND guardrails pass (evaluated separately by the Guardrail Engine, Phase 7)

If confidence is low, the agent downgrades to monitor/warn/cordon, but never
migrate/drain.
"""

from __future__ import annotations

from data_contracts.schemas import ActionTier, FeatureMaturity

DESTRUCTIVE_TIERS = {ActionTier.MIGRATE, ActionTier.DRAIN}


def determine_action_tier(
    *,
    p_fail: float,
    feature_confidence: float,
    feature_maturity: FeatureMaturity,
    stale_telemetry: bool,
    action_thresholds: dict[str, float],
    min_confidence_for_destructive_action: float,
) -> ActionTier:
    """`action_thresholds` maps {"warn", "cordon", "migrate", "drain"} to
    p_fail cutoffs (ascending), per configs/agent.yaml `action_thresholds`."""
    if p_fail >= action_thresholds["drain"]:
        tier = ActionTier.DRAIN
    elif p_fail >= action_thresholds["migrate"]:
        tier = ActionTier.MIGRATE
    elif p_fail >= action_thresholds["cordon"]:
        tier = ActionTier.CORDON
    elif p_fail >= action_thresholds["warn"]:
        tier = ActionTier.WARN
    else:
        tier = ActionTier.MONITOR

    if tier not in DESTRUCTIVE_TIERS:
        return tier

    confidence_ok = feature_confidence >= min_confidence_for_destructive_action
    maturity_ok = feature_maturity == FeatureMaturity.MATURE
    if confidence_ok and maturity_ok and not stale_telemetry:
        return tier

    # Downgrade: insufficient evidence to justify a destructive action.
    return ActionTier.CORDON
