"""Chaos scenario injectors (docs/design_goal.md section 22,
docs/dataset_strategy.md section 3.3).
"""

from __future__ import annotations

from src.simulator.actions import ExecutionError
from src.simulator.fleet import DriveState, FleetSimulator


def inject_stale_telemetry(simulator: FleetSimulator, drive_ids: list[str]) -> None:
    """Marks drives as having stale telemetry, without changing fleet state,
    so downstream guardrails (TELEMETRY_FRESHNESS) can be exercised."""
    simulator.stale_drive_ids.update(drive_ids)


def clear_stale_telemetry(simulator: FleetSimulator, drive_ids: list[str]) -> None:
    simulator.stale_drive_ids.difference_update(drive_ids)


def inject_correlated_failure(
    simulator: FleetSimulator, group_id: str, *, count: int
) -> list[str]:
    """Simultaneously fails `count` drives in the same replication group, to
    exercise quorum guardrails."""
    group = simulator.replication_groups[group_id]
    affected = group.drive_ids[:count]
    for drive_id in affected:
        simulator.drives[drive_id].state = DriveState.FAILED
    return affected


def make_action_timeout_hook(action_ids_to_fail: set[str]):
    """Returns a `fail_hook` for src.simulator.actions.apply_action that
    raises ExecutionError for any action_id in `action_ids_to_fail`,
    simulating an action timeout or node crash mid-execution."""

    def fail_hook(action: dict) -> None:
        if action["action_id"] in action_ids_to_fail:
            raise ExecutionError(f"Simulated action timeout for {action['action_id']}")

    return fail_hook


def make_flaky_hook(fail_probability: float, *, rng) -> callable:
    """Returns a `fail_hook` that raises ExecutionError with the given
    probability on each call, using the supplied `numpy.random.Generator`
    for determinism in tests."""

    def fail_hook(action: dict) -> None:
        if rng.random() < fail_probability:
            raise ExecutionError(f"Simulated random execution failure for {action['action_id']}")

    return fail_hook
