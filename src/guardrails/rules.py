"""Guardrail rule catalog (docs/design_goal.md section 14,
configs/guardrails.yaml). Each rule takes a RuleContext and returns a
GuardrailViolation if it fires, or None otherwise.
"""

from __future__ import annotations

from data_contracts.schemas import (
    ActionTier,
    FeatureMaturity,
    GuardrailSeverity,
    GuardrailViolation,
)
from src.guardrails.context import PostActionContext, RuleContext


def pred_threshold(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.is_destructive() and ctx.p_fail < ctx.prediction_threshold:
        return GuardrailViolation(
            rule_id="PRED_THRESHOLD",
            severity=GuardrailSeverity.HARD,
            message=(
                f"p_fail={ctx.p_fail:.2f} below high-confidence cutoff "
                f"{ctx.prediction_threshold:.2f}"
            ),
        )
    return None


def feature_confidence(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.is_destructive() and ctx.feature_confidence < ctx.min_feature_confidence:
        return GuardrailViolation(
            rule_id="FEATURE_CONFIDENCE",
            severity=GuardrailSeverity.HARD,
            message=(
                f"feature_confidence={ctx.feature_confidence:.2f} below "
                f"threshold {ctx.min_feature_confidence:.2f}"
            ),
        )
    return None


def telemetry_freshness(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.is_destructive() and (
        ctx.stale_telemetry or ctx.feature_maturity != FeatureMaturity.MATURE
    ):
        return GuardrailViolation(
            rule_id="TELEMETRY_FRESHNESS",
            severity=GuardrailSeverity.HARD,
            message="Telemetry stale or drive not MATURE; cannot justify a destructive action.",
        )
    return None


def no_last_node(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.proposed_action == ActionTier.DRAIN and ctx.is_last_healthy_node_in_domain:
        return GuardrailViolation(
            rule_id="HARD_NO_LAST_NODE",
            severity=GuardrailSeverity.HARD,
            message="Refusing to drain the last healthy node in its failure domain.",
        )
    return None


def quorum(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.proposed_action == ActionTier.DRAIN and not ctx.quorum_ok_after_action:
        return GuardrailViolation(
            rule_id="HARD_QUORUM",
            severity=GuardrailSeverity.HARD,
            message="Draining this drive would violate replication quorum.",
        )
    return None


def max_drains(ctx: RuleContext) -> GuardrailViolation | None:
    at_limit = ctx.concurrent_drains >= ctx.max_concurrent_drains
    if ctx.proposed_action == ActionTier.DRAIN and at_limit:
        return GuardrailViolation(
            rule_id="HARD_MAX_DRAINS",
            severity=GuardrailSeverity.HARD,
            message=f"Concurrent drain limit ({ctx.max_concurrent_drains}) reached.",
        )
    return None


def high_io(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.proposed_action == ActionTier.MIGRATE and ctx.high_io_period:
        return GuardrailViolation(
            rule_id="SOFT_HIGH_IO",
            severity=GuardrailSeverity.SOFT,
            message="Migrating during a high I/O period.",
        )
    return None


def maintenance_window(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.is_destructive() and ctx.maintenance_window_active:
        return GuardrailViolation(
            rule_id="OPS_MAINTENANCE",
            severity=GuardrailSeverity.SOFT,
            message="Destructive action requested during an active maintenance window.",
        )
    return None


def rate_limit(ctx: RuleContext) -> GuardrailViolation | None:
    if ctx.actions_in_last_hour >= ctx.max_actions_per_hour:
        return GuardrailViolation(
            rule_id="OPS_RATE_LIMIT",
            severity=GuardrailSeverity.SOFT,
            message=f"Action rate limit ({ctx.max_actions_per_hour}/hour) reached.",
        )
    return None


def post_data_integrity(ctx: PostActionContext) -> GuardrailViolation | None:
    if not ctx.data_integrity_ok:
        return GuardrailViolation(
            rule_id="POST_DATA_INTEGRITY",
            severity=GuardrailSeverity.HARD,
            message=f"Post-action data integrity check failed for {ctx.action_id}.",
        )
    return None


def post_service_continuity(ctx: PostActionContext) -> GuardrailViolation | None:
    if not ctx.service_continuity_ok:
        return GuardrailViolation(
            rule_id="POST_SERVICE_CONTINUITY",
            severity=GuardrailSeverity.HARD,
            message=f"Post-action service continuity check failed for {ctx.action_id}.",
        )
    return None


#: Evaluated in this order; hard rules are listed first so, combined with the
#: engine sorting violations by severity, "safety > operational > efficiency"
#: conflict resolution (docs/design_goal.md section 14) is easy to audit.
ALL_RULES = (
    pred_threshold,
    feature_confidence,
    telemetry_freshness,
    no_last_node,
    quorum,
    max_drains,
    high_io,
    maintenance_window,
    rate_limit,
)

#: Evaluated in the Validate node, after an action has been executed.
POST_ACTION_RULES = (
    post_data_integrity,
    post_service_continuity,
)
