from data_contracts.schemas import ActionTier, FeatureMaturity, GuardrailSeverity
from src.guardrails.adapter import build_guardrail_evaluator
from src.guardrails.context import RuleContext
from src.guardrails.engine import GuardrailEngine
from src.guardrails.operational_state import InMemoryOperationalState


def _drain_ctx(**overrides) -> RuleContext:
    defaults = dict(
        action_id="a1",
        drive_id="D-1",
        proposed_action=ActionTier.DRAIN,
        p_fail=0.95,
        feature_confidence=0.95,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
    )
    defaults.update(overrides)
    return RuleContext(**defaults)


def test_engine_passes_healthy_drain_proposal():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx())
    assert result.passed is True
    assert result.violations == []
    assert result.evaluation_latency_ms < 500


def test_engine_blocks_low_confidence_destructive_action():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx(feature_confidence=0.1))
    assert result.passed is False
    assert any(v.rule_id == "FEATURE_CONFIDENCE" for v in result.violations)


def test_engine_blocks_stale_telemetry_destructive_action():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx(stale_telemetry=True))
    assert result.passed is False
    assert any(v.rule_id == "TELEMETRY_FRESHNESS" for v in result.violations)


def test_engine_blocks_last_healthy_node_drain():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx(is_last_healthy_node_in_domain=True))
    assert result.passed is False
    assert any(v.rule_id == "HARD_NO_LAST_NODE" for v in result.violations)


def test_engine_blocks_quorum_violation():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx(quorum_ok_after_action=False))
    assert result.passed is False
    assert any(v.rule_id == "HARD_QUORUM" for v in result.violations)


def test_engine_blocks_max_concurrent_drains():
    engine = GuardrailEngine()
    result = engine.evaluate(_drain_ctx(concurrent_drains=1, max_concurrent_drains=1))
    assert result.passed is False
    assert any(v.rule_id == "HARD_MAX_DRAINS" for v in result.violations)


def test_soft_violation_does_not_block_but_is_recorded():
    engine = GuardrailEngine()
    ctx = _drain_ctx(
        proposed_action=ActionTier.MIGRATE, high_io_period=True, quorum_ok_after_action=True
    )
    result = engine.evaluate(ctx)
    assert result.passed is True
    assert any(
        v.rule_id == "SOFT_HIGH_IO" and v.severity == GuardrailSeverity.SOFT
        for v in result.violations
    )


def test_hard_violations_sort_before_soft_violations():
    engine = GuardrailEngine()
    ctx = _drain_ctx(feature_confidence=0.1, maintenance_window_active=True)
    result = engine.evaluate(ctx)
    assert result.violations[0].severity == GuardrailSeverity.HARD


def test_monitor_action_never_triggers_destructive_rules():
    engine = GuardrailEngine()
    ctx = _drain_ctx(
        proposed_action=ActionTier.MONITOR,
        feature_confidence=0.0,
        stale_telemetry=True,
        quorum_ok_after_action=False,
    )
    result = engine.evaluate(ctx)
    assert result.passed is True


def test_adapter_enforces_rate_limit_across_calls():
    operational_state = InMemoryOperationalState()
    evaluate = build_guardrail_evaluator(
        operational_state=operational_state, max_actions_per_hour=1, max_concurrent_drains=5
    )
    proposal = {
        "action_id": "a1",
        "drive_id": "D-1",
        "proposed_action": ActionTier.CORDON.value,
        "p_fail": 0.5,
        "feature_confidence": 0.9,
        "feature_maturity": FeatureMaturity.MATURE.value,
        "stale_telemetry": False,
    }
    first = evaluate(proposal)
    assert first["passed"] is True  # cordon isn't destructive; rate limit is soft anyway

    second = evaluate({**proposal, "action_id": "a2"})
    assert any(v["rule_id"] == "OPS_RATE_LIMIT" for v in second["violations"])


def test_adapter_raises_on_missing_required_keys():
    evaluate = build_guardrail_evaluator()
    try:
        evaluate({"action_id": "a1"})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "missing required keys" in str(exc)
