"""Guardrail evaluation latency benchmark (docs/project_plan.md Phase 7
exit criteria and Phase 11 evaluation matrix: <500ms per evaluation)."""

from __future__ import annotations

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.guardrails.context import RuleContext
from src.guardrails.engine import GuardrailEngine

N_EVALUATIONS = 2000
LATENCY_BUDGET_MS = 500.0


def test_guardrail_engine_latency_stays_within_budget():
    engine = GuardrailEngine()
    ctx = RuleContext(
        action_id="a1",
        drive_id="D-1",
        proposed_action=ActionTier.DRAIN,
        p_fail=0.95,
        feature_confidence=0.95,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
    )

    latencies = [engine.evaluate(ctx).evaluation_latency_ms for _ in range(N_EVALUATIONS)]

    assert max(latencies) < LATENCY_BUDGET_MS
    assert (sum(latencies) / len(latencies)) < LATENCY_BUDGET_MS / 10
