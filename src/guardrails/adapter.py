"""Adapts the GuardrailEngine to the agent's dict-based GuardrailEvaluator
seam (src/agent/deps.py), so Plan nodes can call it uniformly regardless of
whether guardrails or the fleet simulator are in play yet.
"""

from __future__ import annotations

from typing import Any

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.guardrails.context import RuleContext
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
    prediction_threshold: float = 0.80,
    min_feature_confidence: float = 0.80,
    max_concurrent_drains: int = 1,
    max_actions_per_hour: int = 5,
):
    engine = engine or GuardrailEngine()
    operational_state = operational_state or InMemoryOperationalState()

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
            prediction_threshold=prediction_threshold,
            min_feature_confidence=min_feature_confidence,
            max_concurrent_drains=max_concurrent_drains,
            max_actions_per_hour=max_actions_per_hour,
        )

        result = engine.evaluate(ctx)
        if result.passed:
            operational_state.record_action(ctx.action_id, ctx.proposed_action)

        return {
            "action_id": result.action_id,
            "passed": result.passed,
            "violations": [v.model_dump() for v in result.violations],
            "final_action": result.final_action.value,
            "evaluation_latency_ms": result.evaluation_latency_ms,
        }

    return evaluate
