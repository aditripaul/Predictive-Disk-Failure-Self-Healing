"""Entry point for `make agent-demo`.

Runs one MAPE-K cycle against a hardcoded in-memory fleet snapshot with the
pass-all guardrail from Phase 6, so the agent core can be exercised before
the real guardrail engine (Phase 7) and fleet simulator (Phase 8) exist.
"""

from __future__ import annotations

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies, default_guardrail_pass_all
from src.agent.graph import compiled_agent

DEMO_DRIVES = [
    {
        "drive_id": "D-1042",
        "p_fail": 0.93,
        "feature_confidence": 0.91,
        "feature_maturity": FeatureMaturity.MATURE,
        "stale_telemetry": False,
    },
    {
        "drive_id": "D-2001",
        "p_fail": 0.20,
        "feature_confidence": 0.80,
        "feature_maturity": FeatureMaturity.MATURE,
        "stale_telemetry": False,
    },
]


def _fleet_state_provider() -> dict:
    return {"fleet_snapshot_id": "demo-snapshot-1", "drives": DEMO_DRIVES}


def _predictor(snapshot: dict) -> dict:
    return {"drives": snapshot["drives"]}


def _executor(action: dict) -> dict:
    print(f"[executor] performing {action['proposed_action']} on {action['drive_id']}")
    return {"action_id": action["action_id"], "success": True}


def _validator(execution_result: dict) -> dict:
    return {"data_integrity_ok": True, "service_continuity_ok": True, "quorum_ok": True}


def main() -> None:
    deps = AgentDependencies(
        fleet_state_provider=_fleet_state_provider,
        predictor=_predictor,
        guardrail_evaluator=default_guardrail_pass_all,
        executor=_executor,
        validator=_validator,
    )

    with compiled_agent(deps) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "demo-fleet"}})

    print(f"proposed_action: {result['proposed_action']}")
    print(f"guardrail_result: {result['guardrail_result']}")
    print(f"execution_result: {result['execution_result']}")
    print(f"validation_result: {result['validation_result']}")
    print(f"decision_record_id: {result['decision_record_id']}")


if __name__ == "__main__":
    main()
