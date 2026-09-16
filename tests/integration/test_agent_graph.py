from langgraph.types import Command

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies, InMemoryActionLedger, default_guardrail_pass_all
from src.agent.graph import compiled_agent
from src.guardrails.adapter import build_guardrail_evaluator
from tests.support import IMMEDIATE_ESCALATION_PLAN_KWARGS

HIGH_RISK_DRIVE = {
    "drive_id": "D-1",
    "p_fail": 0.95,
    "feature_confidence": 0.95,
    "feature_maturity": FeatureMaturity.MATURE,
    "stale_telemetry": False,
}


def _fleet_state_provider():
    return {"fleet_snapshot_id": "snap-1", "drives": [HIGH_RISK_DRIVE]}


def _predictor(snapshot):
    return {"drives": snapshot["drives"]}


def _make_executor(call_log: list[str]):
    def executor(action):
        call_log.append(action["action_id"])
        return {
            "action_id": action["action_id"],
            "drive_id": action["drive_id"],
            "proposed_action": action["proposed_action"],
            "success": True,
        }

    return executor


def _validator(execution_result):
    return {
        "data_integrity_ok": True,
        "service_continuity_ok": True,
        "quorum_ok": True,
    }


def _build_deps(call_log: list[str], ledger=None) -> AgentDependencies:
    return AgentDependencies(
        fleet_state_provider=_fleet_state_provider,
        predictor=_predictor,
        guardrail_evaluator=default_guardrail_pass_all,
        executor=_make_executor(call_log),
        validator=_validator,
        action_ledger=ledger or InMemoryActionLedger(),
    )


def test_full_mape_k_cycle_executes_high_risk_drive():
    call_log: list[str] = []
    deps = _build_deps(call_log)

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "fleet-1"}})

    assert result["proposed_action"]["proposed_action"] == "drain"
    assert result["execution_result"]["success"] is True
    assert result["validation_result"]["quorum_ok"] is True
    assert result["decision_record_id"]
    assert len(call_log) == 1


def test_low_confidence_prediction_downgrades_to_cordon_and_still_executes():
    call_log: list[str] = []
    low_confidence_drive = {**HIGH_RISK_DRIVE, "feature_confidence": 0.1}

    def fleet_state_provider():
        return {"fleet_snapshot_id": "snap-2", "drives": [low_confidence_drive]}

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=default_guardrail_pass_all,
        executor=_make_executor(call_log),
        validator=_validator,
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "fleet-2"}})

    assert result["proposed_action"]["proposed_action"] == "cordon"
    assert result["human_review_required"] is False


def test_guardrail_block_routes_to_human_review_without_executing():
    call_log: list[str] = []

    def blocking_guardrail(proposal):
        return {
            "action_id": proposal["action_id"],
            "passed": False,
            "violations": [{"rule_id": "HARD_QUORUM", "severity": "hard", "message": "blocked"}],
            "final_action": proposal["proposed_action"],
        }

    deps = _build_deps(call_log)
    deps.guardrail_evaluator = blocking_guardrail

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "fleet-3"}})

    assert result["human_review_required"] is True
    assert "execution_result" not in result or result.get("execution_result") is None
    assert call_log == []


def test_crash_recovery_replay_does_not_duplicate_execution():
    """Re-invoking the same thread_id resumes from the persisted checkpoint,
    so `run_id` (and therefore the deterministic action_id) is unchanged
    across cycles. This models "crash mid-cycle, restart the same logical
    run": the idempotency ledger must prevent a second real execution."""
    call_log: list[str] = []
    ledger = InMemoryActionLedger()
    deps = _build_deps(call_log, ledger=ledger)

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        config = {"configurable": {"thread_id": "fleet-crash"}}
        first = app.invoke({}, config=config)
        second = app.invoke({}, config=config)

    assert first["proposed_action"]["action_id"] == second["proposed_action"]["action_id"]
    assert len(call_log) == 1  # second cycle replayed the ledger result, did not re-execute
    assert second["execution_result"] == first["execution_result"]


def test_real_guardrail_engine_blocks_last_healthy_node_drain():
    """End-to-end with the Phase 7 GuardrailEngine (not the pass-all stub):
    action_tiers already allows a high-confidence drain, but the guardrail
    engine must still veto draining the last healthy node in its domain."""
    call_log: list[str] = []
    last_node_drive = {**HIGH_RISK_DRIVE, "is_last_healthy_node_in_domain": True}

    def fleet_state_provider():
        return {"fleet_snapshot_id": "snap-4", "drives": [last_node_drive]}

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "fleet-real-guardrail"}})

    assert result["proposed_action"]["proposed_action"] == "drain"
    assert result["guardrail_result"]["passed"] is False
    assert any(
        v["rule_id"] == "HARD_NO_LAST_NODE" for v in result["guardrail_result"]["violations"]
    )
    assert result["human_review_required"] is True
    assert call_log == []
    assert "__interrupt__" in result


def test_human_review_approval_resumes_and_executes_the_action():
    """The core Phase 10 exit criterion: approve/reject must actually
    resume the LangGraph workflow, not dead-end at a terminal node."""

    call_log: list[str] = []
    last_node_drive = {**HIGH_RISK_DRIVE, "is_last_healthy_node_in_domain": True}

    def fleet_state_provider():
        return {"fleet_snapshot_id": "snap-5", "drives": [last_node_drive]}

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )

    config = {"configurable": {"thread_id": "fleet-resume-approve"}}
    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        first = app.invoke({}, config=config)
        assert "__interrupt__" in first
        assert call_log == []

        second = app.invoke(
            Command(resume={"approved": True, "operator_id": "op-1", "reason_code": "CONFIRMED"}),
            config=config,
        )

    assert second["human_decision"] == "approved"
    assert second["human_review_operator_id"] == "op-1"
    assert second["execution_result"]["success"] is True
    assert second["validation_result"]["quorum_ok"] is True
    assert len(call_log) == 1


def test_human_review_rejection_resumes_without_executing():
    call_log: list[str] = []
    last_node_drive = {**HIGH_RISK_DRIVE, "is_last_healthy_node_in_domain": True}

    def fleet_state_provider():
        return {"fleet_snapshot_id": "snap-6", "drives": [last_node_drive]}

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )


    config = {"configurable": {"thread_id": "fleet-resume-reject"}}
    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        app.invoke({}, config=config)
        second = app.invoke(
            Command(
                resume={"approved": False, "operator_id": "op-1", "reason_code": "FALSE_POSITIVE"}
            ),
            config=config,
        )

    assert second["human_decision"] == "rejected"
    assert second.get("execution_result") is None
    assert call_log == []


def test_real_hysteresis_config_caps_first_cycle_and_escalates_on_the_second():
    """No plan_kwargs override here - this exercises the actual
    configs/agent.yaml default (hysteresis_cycles_required=2), proving
    docs/design_goal.md section 13 ("a drive must remain high-risk for
    multiple cycles before escalation") is real, not just a config value
    that AgentSettings loads and nothing reads."""
    call_log: list[str] = []
    deps = AgentDependencies(
        fleet_state_provider=_fleet_state_provider,
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )

    config = {"configurable": {"thread_id": "fleet-hysteresis"}}
    with compiled_agent(deps) as app:  # real AgentSettings defaults, no override
        first = app.invoke({}, config=config)
        second = app.invoke({}, config=config)

    # cycle 1: real p_fail/confidence would justify DRAIN, but hysteresis
    # caps it at WARN since this is the drive's first high-risk cycle.
    assert first["proposed_action"]["proposed_action"] == "warn"
    assert first["execution_result"]["proposed_action"] == "warn"
    # cycle 2: sustained for hysteresis_cycles_required=2, now let through.
    assert second["proposed_action"]["proposed_action"] == "drain"
    assert second["execution_result"]["proposed_action"] == "drain"
    assert second["execution_result"]["success"] is True
    assert call_log == [
        first["proposed_action"]["action_id"],
        second["proposed_action"]["action_id"],
    ]


def test_cooldown_prevents_re_drain_on_the_cycle_immediately_after_one():
    """With hysteresis satisfied (cycle 3+), a drive that was just drained
    should not be proposed for another destructive action on the very next
    cycle - the default 3600s cooldown should downgrade it to WARN."""
    call_log: list[str] = []
    deps = AgentDependencies(
        fleet_state_provider=_fleet_state_provider,
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )

    config = {"configurable": {"thread_id": "fleet-cooldown"}}
    with compiled_agent(
        deps, plan_kwargs={"hysteresis_cycles_required": 1}
    ) as app:  # real cooldown default, only hysteresis relaxed
        first = app.invoke({}, config=config)
        second = app.invoke({}, config=config)

    assert first["proposed_action"]["proposed_action"] == "drain"
    assert first["execution_result"]["success"] is True
    # cooldown downgrades the very next cycle's proposal to WARN, so the
    # drive is never drained twice back-to-back on the same evidence.
    assert second["proposed_action"]["proposed_action"] == "warn"
    assert second["execution_result"]["proposed_action"] == "warn"
    drain_calls = [
        action_id
        for action_id in call_log
        if action_id == first["proposed_action"]["action_id"]
    ]
    assert len(drain_calls) == 1  # the drain itself was never executed twice
