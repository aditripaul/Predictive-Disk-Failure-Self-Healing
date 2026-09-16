"""Chaos scenario tests against the full agent + guardrail + simulator
stack (docs/design_goal.md section 22, docs/project_plan.md Phase 8/11 exit
criteria: zero hard-guardrail violations executed, compensating actions
verified, human-review timeout fallback tested).
"""

from __future__ import annotations

import datetime as dt
from functools import partial

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.agent.deps import AgentDependencies
from src.agent.graph import compiled_agent
from src.agent.human_review_timeout import resolve_timeout_fallback
from src.guardrails.adapter import build_guardrail_evaluator
from src.simulator.actions import apply_action, validate_action
from src.simulator.chaos import (
    inject_correlated_failure,
    inject_stale_telemetry,
    make_action_timeout_hook,
)
from src.simulator.fleet import Drive, DriveState, FleetSimulator, Node, ReplicationGroup
from tests.support import IMMEDIATE_ESCALATION_PLAN_KWARGS

HIGH_RISK_DRIVE_ID = "D-1"


def _build_fleet() -> FleetSimulator:
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
    groups = [ReplicationGroup("group-1", drive_ids=["D-2", "D-3", "D-1"], min_healthy_drives=1)]
    return FleetSimulator(drives, nodes, groups)


def _predictor(snapshot: dict) -> dict:
    drives = []
    for d in snapshot["drives"]:
        is_target = d["drive_id"] == HIGH_RISK_DRIVE_ID
        drives.append(
            {
                "drive_id": d["drive_id"],
                "p_fail": 0.95 if is_target else 0.05,
                "feature_confidence": 0.95,
                "feature_maturity": FeatureMaturity.MATURE,
                "stale_telemetry": d["stale_telemetry"],
                "is_last_healthy_node_in_domain": d["is_last_healthy_node_in_domain"],
                "quorum_ok_after_action": d["quorum_ok_after_action"],
            }
        )
    return {"drives": drives}


def _fleet_state_provider(fleet: FleetSimulator) -> dict:
    snapshot = fleet.snapshot()
    for d in snapshot["drives"]:
        d["is_last_healthy_node_in_domain"] = fleet.is_last_healthy_node_in_domain(d["drive_id"])
        d["quorum_ok_after_action"] = fleet.quorum_ok_after_drain(d["drive_id"])
    return snapshot


def test_stale_telemetry_downgrades_destructive_action_and_no_data_loss():
    """Chaos: the high-risk drive's telemetry has gone stale (agent crashed
    or drive stopped reporting). Defense-in-depth: the Plan node's action-
    tier policy (Phase 5) already downgrades a stale-telemetry destructive
    proposal to cordon before guardrails even run, so the drive must never
    be drained. (If that downgrade were ever removed, TELEMETRY_FRESHNESS
    provides the second layer of defense - see test_guardrails.py.)"""
    fleet = _build_fleet()
    inject_stale_telemetry(fleet, [HIGH_RISK_DRIVE_ID])

    deps = AgentDependencies(
        fleet_state_provider=partial(_fleet_state_provider, fleet),
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=partial(apply_action, fleet),
        validator=partial(validate_action, fleet),
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "chaos-stale"}})

    assert result["proposed_action"]["proposed_action"] == "cordon"
    assert result["guardrail_result"]["passed"] is True
    assert fleet.drives[HIGH_RISK_DRIVE_ID].state == DriveState.CORDONED
    assert fleet.drives[HIGH_RISK_DRIVE_ID].state != DriveState.DRAINED  # zero data-loss risk


def test_correlated_multi_drive_failure_triggers_quorum_guardrail():
    """Chaos: two of the three drives in the replication group fail
    simultaneously (e.g. a rack power event). Any further drain proposal in
    that group must be blocked by the quorum guardrail."""
    fleet = _build_fleet()
    failed = inject_correlated_failure(fleet, "group-1", count=2)
    assert HIGH_RISK_DRIVE_ID not in failed  # D-1 remains the one flagged as high-risk

    deps = AgentDependencies(
        fleet_state_provider=partial(_fleet_state_provider, fleet),
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=partial(apply_action, fleet),
        validator=partial(validate_action, fleet),
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "chaos-correlated"}})

    assert result["guardrail_result"]["passed"] is False
    assert any(v["rule_id"] == "HARD_QUORUM" for v in result["guardrail_result"]["violations"])
    assert fleet.drives[HIGH_RISK_DRIVE_ID].state == DriveState.HEALTHY


def test_action_timeout_triggers_compensating_action_and_zero_data_loss():
    """Chaos: the execute step crashes/times out mid-action. The simulator
    must roll back to the drive's prior state (compensating action) rather
    than leave it in an inconsistent DRAINED state, and validation must
    reflect the failure."""
    fleet = _build_fleet()

    def flaky_executor(action: dict) -> dict:
        hook = make_action_timeout_hook({action["action_id"]})
        return apply_action(fleet, action, fail_hook=hook)

    deps = AgentDependencies(
        fleet_state_provider=partial(_fleet_state_provider, fleet),
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=flaky_executor,
        validator=partial(validate_action, fleet),
    )

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "chaos-timeout"}})

    assert result["guardrail_result"]["passed"] is True  # guardrails were fine
    assert result["execution_result"]["success"] is False
    assert result["execution_result"]["compensating_action_triggered"] is True
    assert fleet.drives[HIGH_RISK_DRIVE_ID].state == DriveState.HEALTHY  # rolled back, no data loss
    assert result["validation_result"]["service_continuity_ok"] is False
    assert result["validation_result"]["data_integrity_ok"] is True  # compensated, not corrupted


def test_human_review_timeout_falls_back_to_safe_action():
    """Chaos: a blocked action sits in the approval queue past its critical
    SLA with no operator response. The system must resolve to the safest
    non-destructive fallback (cordon), never silently execute the original
    destructive action."""
    created_at = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
    two_hours_later = created_at + dt.timedelta(hours=2)

    still_pending = resolve_timeout_fallback(
        created_at=created_at, is_critical=True, now=created_at, critical_sla_hours=1
    )
    assert still_pending is None

    fallback = resolve_timeout_fallback(
        created_at=created_at, is_critical=True, now=two_hours_later, critical_sla_hours=1
    )
    assert fallback == ActionTier.CORDON
