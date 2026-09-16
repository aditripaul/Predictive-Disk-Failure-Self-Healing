"""End-to-end MAPE-K cycle against the Phase 8 fleet simulator, with the
real Phase 7 guardrail engine, exercising the full vertical slice described
in docs/design_goal.md section 7.
"""

from __future__ import annotations

from functools import partial

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies
from src.agent.graph import compiled_agent
from src.guardrails.adapter import build_guardrail_evaluator
from src.simulator.actions import apply_action, validate_action
from src.simulator.fleet import Drive, DriveState, FleetSimulator, Node, ReplicationGroup
from tests.support import IMMEDIATE_ESCALATION_PLAN_KWARGS


def _build_fleet() -> FleetSimulator:
    # node-a and node-b share failure domain rack-1, so draining D-1 does not
    # make node-a "the last healthy node in its domain" while node-b (D-2)
    # is still healthy.
    drives = [
        Drive("D-1", "node-a", "group-1"),
        Drive("D-2", "node-b", "group-1"),
        Drive("D-3", "node-c", "group-1"),
    ]
    nodes = [
        Node("node-a", failure_domain="rack-1", drive_ids=["D-1"]),
        Node("node-b", failure_domain="rack-1", drive_ids=["D-2"]),
        Node("node-c", failure_domain="rack-2", drive_ids=["D-3"]),
    ]
    groups = [ReplicationGroup("group-1", drive_ids=["D-1", "D-2", "D-3"], min_healthy_drives=1)]
    return FleetSimulator(drives, nodes, groups)


def _predictor_from_snapshot(snapshot: dict) -> dict:
    """Stand-in for the Phase 5 model: treats a hardcoded target drive as
    high-risk and everything else as healthy, and folds in fleet-topology
    context the guardrail engine needs."""
    drives = []
    for d in snapshot["drives"]:
        is_target = d["drive_id"] == "D-1"
        drives.append(
            {
                "drive_id": d["drive_id"],
                "p_fail": 0.95 if is_target else 0.05,
                "feature_confidence": 0.95,
                "feature_maturity": FeatureMaturity.MATURE,
                "stale_telemetry": d["stale_telemetry"],
            }
        )
    return {"drives": drives}


def test_agent_drains_high_risk_drive_via_simulator_end_to_end():
    fleet = _build_fleet()

    def fleet_state_provider():
        snapshot = fleet.snapshot()
        for d in snapshot["drives"]:
            d["is_last_healthy_node_in_domain"] = fleet.is_last_healthy_node_in_domain(
                d["drive_id"]
            )
            d["quorum_ok_after_action"] = fleet.quorum_ok_after_drain(d["drive_id"])
        return snapshot

    def predictor(snapshot):
        result = _predictor_from_snapshot(snapshot)
        # carry the fleet-topology fields through to the proposal
        by_id = {d["drive_id"]: d for d in snapshot["drives"]}
        for drive in result["drives"]:
            drive["is_last_healthy_node_in_domain"] = by_id[drive["drive_id"]][
                "is_last_healthy_node_in_domain"
            ]
            drive["quorum_ok_after_action"] = by_id[drive["drive_id"]]["quorum_ok_after_action"]
        return result

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=partial(apply_action, fleet),
        validator=partial(validate_action, fleet),
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "sim-fleet-1"}})

    assert result["proposed_action"]["drive_id"] == "D-1"
    assert result["execution_result"]["success"] is True
    assert fleet.drives["D-1"].state == DriveState.DRAINED
    assert result["validation_result"]["quorum_ok"] is True
    assert result["validation_result"]["data_integrity_ok"] is True


def test_agent_blocked_by_guardrail_when_draining_would_break_quorum():
    fleet = _build_fleet()
    # Fail two of the three drives in the group up front: only D-1 remains
    # healthy, so draining it would leave 0 < min_healthy=1.
    fleet.drives["D-2"].state = DriveState.FAILED
    fleet.drives["D-3"].state = DriveState.FAILED

    def fleet_state_provider():
        snapshot = fleet.snapshot()
        for d in snapshot["drives"]:
            d["is_last_healthy_node_in_domain"] = fleet.is_last_healthy_node_in_domain(
                d["drive_id"]
            )
            d["quorum_ok_after_action"] = fleet.quorum_ok_after_drain(d["drive_id"])
        return snapshot

    def predictor(snapshot):
        result = _predictor_from_snapshot(snapshot)
        by_id = {d["drive_id"]: d for d in snapshot["drives"]}
        for drive in result["drives"]:
            drive["is_last_healthy_node_in_domain"] = by_id[drive["drive_id"]][
                "is_last_healthy_node_in_domain"
            ]
            drive["quorum_ok_after_action"] = by_id[drive["drive_id"]]["quorum_ok_after_action"]
        return result

    deps = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=partial(apply_action, fleet),
        validator=partial(validate_action, fleet),
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "sim-fleet-2"}})

    assert result["guardrail_result"]["passed"] is False
    assert any(v["rule_id"] == "HARD_QUORUM" for v in result["guardrail_result"]["violations"])
    assert result["human_review_required"] is True
    assert fleet.drives["D-1"].state == DriveState.HEALTHY  # never executed
