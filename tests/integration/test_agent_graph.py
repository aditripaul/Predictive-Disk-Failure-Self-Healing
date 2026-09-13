from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies, InMemoryActionLedger, default_guardrail_pass_all
from src.agent.graph import compiled_agent

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
        return {"action_id": action["action_id"], "success": True}

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

    with compiled_agent(deps) as app:
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

    with compiled_agent(deps) as app:
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

    with compiled_agent(deps) as app:
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

    with compiled_agent(deps) as app:
        config = {"configurable": {"thread_id": "fleet-crash"}}
        first = app.invoke({}, config=config)
        second = app.invoke({}, config=config)

    assert first["proposed_action"]["action_id"] == second["proposed_action"]["action_id"]
    assert len(call_log) == 1  # second cycle replayed the ledger result, did not re-execute
    assert second["execution_result"] == first["execution_result"]
