"""Adapts the GuardrailEngine to the agent's dict-based GuardrailEvaluator
seam (src/agent/deps.py), so Plan nodes can call it uniformly regardless of
whether guardrails or the fleet simulator are in play yet.
"""

from __future__ import annotations

from typing import Any

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.config import GuardrailSettings
from src.guardrails.context import PostActionContext, RuleContext
from src.guardrails.engine import GuardrailEngine
from src.guardrails.operational_state import InMemoryOperationalState

REQUIRED_PROPOSAL_KEYS = (
    "action_id",
    "drive_id",
    "proposed_action",
    "p_fail",
    "feature_confidence",
    "feature_maturity",
    "stale_telemetry",
)


def build_guardrail_evaluator(
    *,
    engine: GuardrailEngine | None = None,
    operational_state: InMemoryOperationalState | None = None,
    prediction_threshold: float | None = None,
    min_feature_confidence: float | None = None,
    max_concurrent_drains: int | None = None,
    max_actions_per_hour: int | None = None,
):
    """Thresholds default to configs/guardrails.yaml + configs/agent.yaml
    (via GuardrailSettings.load()) rather than hardcoded literals, resolved
    lazily at call time so callers/tests can override individual values
    without needing the config files present."""
    engine = engine or GuardrailEngine()
    operational_state = operational_state or InMemoryOperationalState()
    # A config-file read is cheap; always loading it (rather than only when
    # some argument is None) keeps the fallback resolution below simple and
    # fully type-safe, with no Optional narrowing across the closure below
    # (mypy cannot carry an `if x is None: x = ...` narrowing of an outer-
    # scope variable across a nested function boundary).
    defaults = GuardrailSettings.load()
    resolved_prediction_threshold: float = (
        prediction_threshold if prediction_threshold is not None else defaults.prediction_threshold
    )
    resolved_min_feature_confidence: float = (
        min_feature_confidence
        if min_feature_confidence is not None
        else defaults.min_feature_confidence
    )
    resolved_max_concurrent_drains: int = (
        max_concurrent_drains
        if max_concurrent_drains is not None
        else defaults.max_concurrent_drains
    )
    resolved_max_actions_per_hour: int = (
        max_actions_per_hour if max_actions_per_hour is not None else defaults.max_actions_per_hour
    )

    def evaluate(proposal: dict[str, Any]) -> dict[str, Any]:
        missing = [k for k in REQUIRED_PROPOSAL_KEYS if k not in proposal]
        if missing:
            raise ValueError(f"Guardrail proposal missing required keys: {missing}")

        ctx = RuleContext(
            action_id=proposal["action_id"],
            drive_id=proposal["drive_id"],
            proposed_action=ActionTier(proposal["proposed_action"]),
            p_fail=proposal["p_fail"],
            feature_confidence=proposal["feature_confidence"],
            feature_maturity=FeatureMaturity(proposal["feature_maturity"]),
            stale_telemetry=proposal["stale_telemetry"],
            is_last_healthy_node_in_domain=proposal.get("is_last_healthy_node_in_domain", False),
            quorum_ok_after_action=proposal.get("quorum_ok_after_action", True),
            concurrent_drains=operational_state.concurrent_drains(),
            high_io_period=proposal.get("high_io_period", False),
            maintenance_window_active=proposal.get("maintenance_window_active", False),
            actions_in_last_hour=operational_state.actions_in_last_hour(),
            prediction_threshold=resolved_prediction_threshold,
            min_feature_confidence=resolved_min_feature_confidence,
            max_concurrent_drains=resolved_max_concurrent_drains,
            max_actions_per_hour=resolved_max_actions_per_hour,
        )

        result = engine.evaluate(ctx)
        if result.passed:
            operational_state.record_action(ctx.action_id, ctx.proposed_action)

        return _guardrail_result_to_dict(result)

    return evaluate


def build_post_action_guardrail_evaluator(*, engine: GuardrailEngine | None = None):
    """Adapts GuardrailEngine.evaluate_post_action to the agent's
    PostActionGuardrailEvaluator seam (src/agent/deps.py), called from the
    Validate node after execution."""
    engine = engine or GuardrailEngine()

    def evaluate(
        execution_result: dict[str, Any], validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        ctx = PostActionContext(
            action_id=execution_result.get("action_id", ""),
            proposed_action=ActionTier(execution_result["proposed_action"]),
            data_integrity_ok=validation_result.get("data_integrity_ok", True),
            service_continuity_ok=validation_result.get("service_continuity_ok", True),
        )
        result = engine.evaluate_post_action(ctx)
        return _guardrail_result_to_dict(result)

    return evaluate


def _guardrail_result_to_dict(result) -> dict[str, Any]:
    return {
        "action_id": result.action_id,
        "passed": result.passed,
        "violations": [v.model_dump() for v in result.violations],
        "final_action": result.final_action.value,
        "evaluation_latency_ms": result.evaluation_latency_ms,
    }
