"""Tests the agent <-> API store wiring (src/agent/orchestrator.py): a
blocked action must appear in the approval queue, and approving/rejecting it
through the store must actually resume the paused LangGraph thread."""

from __future__ import annotations

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies
from src.agent.graph import compiled_agent
from src.agent.orchestrator import AgentOrchestrator
from src.api.store import InMemoryAuditStore
from src.guardrails.adapter import build_guardrail_evaluator
from tests.support import IMMEDIATE_ESCALATION_PLAN_KWARGS

LAST_NODE_DRIVE = {
    "drive_id": "D-1",
    "p_fail": 0.95,
    "feature_confidence": 0.95,
    "feature_maturity": FeatureMaturity.MATURE,
    "stale_telemetry": False,
    "is_last_healthy_node_in_domain": True,
}

HEALTHY_DRIVE = {
    "drive_id": "D-1",
    "p_fail": 0.05,
    "feature_confidence": 0.95,
    "feature_maturity": FeatureMaturity.MATURE,
    "stale_telemetry": False,
}


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
    return {"data_integrity_ok": True, "service_continuity_ok": True, "quorum_ok": True}


def test_blocked_action_appears_in_pending_queue_and_run_cycle_reports_pending():
    call_log: list[str] = []
    deps = AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "snap-1", "drives": [LAST_NODE_DRIVE]},
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )
    store = InMemoryAuditStore()

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        orchestrator = AgentOrchestrator(app=app, store=store)
        outcome = orchestrator.run_cycle("orch-thread-1")

        assert outcome["status"] == "pending_review"
        pending = store.list_pending_actions()
        assert len(pending) == 1
        assert pending[0].drive_id == "D-1"
        assert pending[0].thread_id == "orch-thread-1"
        assert call_log == []

        action_id = outcome["action_id"]
        resumed = orchestrator.resume_after_decision(
            action_id, approve=True, operator_id="op-1", reason_code="CONFIRMED_TRAJECTORY"
        )

    assert resumed["status"] == "completed"
    assert call_log == [action_id]
    assert store.get_action(action_id).status == "approved"
    assert store.get_action(action_id).operator_id == "op-1"

    decisions = store.list_decisions()
    assert len(decisions) == 1
    assert decisions[0]["action_id"] == action_id
    assert decisions[0]["human_decision"] == "approved"
    assert decisions[0]["execution_result"]["success"] is True
    # The original guardrail evaluation hit a hard violation (HARD_NO_LAST_NODE);
    # a human choosing to override and execute anyway doesn't retroactively
    # make that a trustworthy automated decision (docs/design_goal.md section
    # 16 veto semantics apply to the decision that was made, not any
    # subsequent human override of it).
    assert decisions[0]["trust_score_provisional"] == 0.0


def test_rejecting_a_pending_action_never_executes_it():
    call_log: list[str] = []
    deps = AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "snap-2", "drives": [LAST_NODE_DRIVE]},
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )
    store = InMemoryAuditStore()

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        orchestrator = AgentOrchestrator(app=app, store=store)
        outcome = orchestrator.run_cycle("orch-thread-2")
        action_id = outcome["action_id"]

        resumed = orchestrator.resume_after_decision(
            action_id, approve=False, operator_id="op-2", reason_code="FALSE_POSITIVE"
        )

    assert resumed["status"] == "completed"
    assert call_log == []
    assert store.get_action(action_id).status == "rejected"

    # Audit completeness (docs/design_goal.md section 6 invariant table):
    # even a human-rejected, never-executed proposal still gets a decision
    # record — it just carries no execution_result.
    decisions = store.list_decisions()
    assert len(decisions) == 1
    assert decisions[0]["human_decision"] == "rejected"
    assert decisions[0]["execution_result"] is None


def test_low_risk_cycle_completes_with_monitor_tier_and_is_still_audited():
    """A healthy drive still yields a "monitor" proposal (docs/design_goal.md
    action tiers: Monitor = "no action" operationally, but every decision
    must still have an audit record per the section 6 invariant table)."""
    call_log: list[str] = []
    deps = AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "snap-3", "drives": [HEALTHY_DRIVE]},
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_make_executor(call_log),
        validator=_validator,
    )
    store = InMemoryAuditStore()

    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as app:
        orchestrator = AgentOrchestrator(app=app, store=store)
        outcome = orchestrator.run_cycle("orch-thread-3")

    assert outcome["status"] == "completed"
    assert store.list_pending_actions() == []
    decisions = store.list_decisions()
    assert len(decisions) == 1
    assert decisions[0]["trust_score_provisional"] == 0.05
