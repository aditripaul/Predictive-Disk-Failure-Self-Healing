"""Action execution against the fleet simulator, with idempotency and
compensating actions on simulated failure (docs/design_goal.md sections
10, 14, 21).
"""

from __future__ import annotations

from data_contracts.schemas import ActionTier
from src.simulator.fleet import Drive, DriveState, FleetSimulator

_ACTION_TO_STATE = {
    ActionTier.CORDON: DriveState.CORDONED,
    ActionTier.MIGRATE: DriveState.MIGRATED,
    ActionTier.DRAIN: DriveState.DRAINED,
}


class ExecutionError(Exception):
    """Raised by an injected chaos hook to simulate an action-execution
    failure (crash, timeout, partial migration failure)."""


def apply_action(
    simulator: FleetSimulator,
    action: dict,
    *,
    fail_hook: callable | None = None,
) -> dict:
    """Executes cordon/migrate/drain against the simulator. `fail_hook`, if
    provided, is called with `action` and may raise ExecutionError to
    simulate a chaos-injected execution failure, triggering a compensating
    action (revert to the drive's prior serving state)."""
    drive_id = action["drive_id"]
    tier = ActionTier(action["proposed_action"])
    drive = simulator.drives[drive_id]
    new_state = _ACTION_TO_STATE.get(tier)

    if new_state is None:
        # monitor/warn/human_review: no fleet-state change.
        return {
            "action_id": action["action_id"],
            "drive_id": drive_id,
            "proposed_action": tier.value,
            "success": True,
            "compensating_action_triggered": False,
            "error": None,
        }

    previous_state = drive.state
    try:
        if fail_hook is not None:
            fail_hook(action)
        drive.state = new_state
        success, compensating, error = True, False, None
    except ExecutionError as exc:
        # Compensating action: roll back to a safe, previously-observed state
        # rather than leaving the drive in an inconsistent one.
        drive.state = previous_state
        success, compensating, error = False, True, str(exc)

    return {
        "action_id": action["action_id"],
        "drive_id": drive_id,
        "proposed_action": tier.value,
        "success": success,
        "compensating_action_triggered": compensating,
        "error": error,
    }


def validate_action(simulator: FleetSimulator, execution_result: dict) -> dict:
    """Takes only `execution_result` (which carries drive_id/proposed_action
    from apply_action above), so it composes directly with the agent's
    Validator seam (src/agent/deps.py), which only forwards execution_result."""
    drive_id = execution_result["drive_id"]
    tier = ActionTier(execution_result["proposed_action"])

    quorum_ok = True
    if tier == ActionTier.DRAIN and execution_result["success"]:
        # Post-hoc check: quorum must still hold now that the drive is drained.
        group = simulator.replication_groups.get(
            simulator.drives[drive_id].replication_group_id
        )
        if group is not None:
            healthy_count = sum(
                1
                for did in group.drive_ids
                if simulator.drives[did].state in {DriveState.HEALTHY, DriveState.DEGRADED}
            )
            quorum_ok = healthy_count >= group.min_healthy_drives

    data_integrity_ok = (
        execution_result["success"] or execution_result["compensating_action_triggered"]
    )
    return {
        "data_integrity_ok": data_integrity_ok,
        "service_continuity_ok": execution_result["success"],
        "quorum_ok": quorum_ok,
    }


__all__ = ["Drive", "ExecutionError", "apply_action", "validate_action"]
