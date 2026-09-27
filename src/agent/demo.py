"""Entry point for `make agent-demo`.

Runs one MAPE-K cycle against a hardcoded in-memory fleet snapshot with the
real Phase 7 guardrail engine, so the agent core can be exercised before the
real fleet simulator (Phase 8) exists (fleet-topology context like quorum
and last-node status defaults to safe assumptions until then).
"""

from __future__ import annotations

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import ActionLedger, AgentDependencies, SqliteActionLedger
from src.agent.graph import compiled_agent
from src.config import AgentSettings
from src.guardrails.adapter import build_guardrail_evaluator
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

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
    logger.info(
        "demo_executor_performing_action",
        proposed_action=action["proposed_action"],
        drive_id=action["drive_id"],
    )
    return {
        "action_id": action["action_id"],
        "drive_id": action["drive_id"],
        "proposed_action": action["proposed_action"],
        "success": True,
    }


def _validator(execution_result: dict) -> dict:
    return {"data_integrity_ok": True, "service_continuity_ok": True, "quorum_ok": True}


def build_demo_dependencies(*, action_ledger: ActionLedger | None = None) -> AgentDependencies:
    """Reusable by anything that needs a runnable-but-not-real agent: the
    CLI demo below, and the FastAPI service's default wiring (src/api/main.py)
    until a real fleet simulator/model are plugged in for a given
    deployment. Defaults to the persistent SqliteActionLedger (not the
    in-memory one) so idempotent crash recovery is real, not merely
    available - pass `action_ledger=InMemoryActionLedger()` (e.g. in tests)
    to avoid ever touching the real project sqlite file on disk."""
    resolved_ledger = action_ledger or SqliteActionLedger(AgentSettings.load().action_ledger_path)
    return AgentDependencies(
        fleet_state_provider=_fleet_state_provider,
        predictor=_predictor,
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_executor,
        validator=_validator,
        action_ledger=resolved_ledger,
    )


def main() -> None:
    configure_logging()
    deps = build_demo_dependencies()
    checkpoint_path = AgentSettings.load().checkpoint_path

    with compiled_agent(deps, checkpoint_path=checkpoint_path) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "demo-fleet"}})

    logger.info(
        "demo_cycle_completed",
        proposed_action=result["proposed_action"],
        guardrail_result=result["guardrail_result"],
        execution_result=result["execution_result"],
        validation_result=result["validation_result"],
        decision_record_id=result["decision_record_id"],
    )


if __name__ == "__main__":
    main()
