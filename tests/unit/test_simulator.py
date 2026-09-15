import numpy as np

from data_contracts.schemas import ActionTier
from src.simulator.actions import apply_action, validate_action
from src.simulator.chaos import (
    inject_correlated_failure,
    inject_stale_telemetry,
    make_action_timeout_hook,
    make_flaky_hook,
)
from src.simulator.fleet import Drive, DriveState, FleetSimulator, Node, ReplicationGroup


def _two_node_fleet() -> FleetSimulator:
    drives = [
        Drive("D-1", "node-a", "group-1"),
        Drive("D-2", "node-b", "group-1"),
        Drive("D-3", "node-b", "group-1"),
    ]
    nodes = [
        Node("node-a", failure_domain="rack-1", drive_ids=["D-1"]),
        Node("node-b", failure_domain="rack-2", drive_ids=["D-2", "D-3"]),
    ]
    groups = [ReplicationGroup("group-1", drive_ids=["D-1", "D-2", "D-3"], min_healthy_drives=2)]
    return FleetSimulator(drives, nodes, groups)


def test_snapshot_reflects_current_state():
    sim = _two_node_fleet()
    snapshot = sim.snapshot()
    assert len(snapshot["drives"]) == 3
    assert all(d["state"] == "HEALTHY" for d in snapshot["drives"])


def test_is_last_healthy_node_in_domain():
    sim = _two_node_fleet()
    # node-a is alone in rack-1 -> it's the last healthy node in its domain.
    assert sim.is_last_healthy_node_in_domain("D-1") is True
    # node-b shares no domain with another node, but has 2 drives of its own;
    # still the only node in rack-2, so also "last" by this domain definition.
    assert sim.is_last_healthy_node_in_domain("D-2") is True


def test_quorum_ok_after_drain():
    sim = _two_node_fleet()
    # 3 healthy drives, min_healthy=2: draining one leaves 2 -> quorum ok.
    assert sim.quorum_ok_after_drain("D-1") is True

    sim.drives["D-2"].state = DriveState.FAILED
    # now only D-1 and D-3 healthy; draining D-1 leaves 1 < min_healthy=2.
    assert sim.quorum_ok_after_drain("D-1") is False


def test_apply_action_cordon_updates_state():
    sim = _two_node_fleet()
    action = {"action_id": "a1", "drive_id": "D-1", "proposed_action": ActionTier.CORDON.value}
    result = apply_action(sim, action)
    assert result["success"] is True
    assert sim.drives["D-1"].state == DriveState.CORDONED


def test_apply_action_with_fail_hook_triggers_compensating_action():
    sim = _two_node_fleet()
    hook = make_action_timeout_hook({"a1"})
    action = {"action_id": "a1", "drive_id": "D-1", "proposed_action": ActionTier.DRAIN.value}

    result = apply_action(sim, action, fail_hook=hook)

    assert result["success"] is False
    assert result["compensating_action_triggered"] is True
    # Rolled back to the prior state, not left DRAINED.
    assert sim.drives["D-1"].state == DriveState.HEALTHY


def test_validate_action_detects_quorum_violation_after_drain():
    sim = _two_node_fleet()
    sim.drives["D-2"].state = DriveState.FAILED  # only D-1, D-3 healthy now

    action = {"action_id": "a1", "drive_id": "D-3", "proposed_action": ActionTier.DRAIN.value}
    exec_result = apply_action(sim, action)
    validation = validate_action(sim, exec_result)

    assert exec_result["success"] is True
    assert validation["quorum_ok"] is False  # only D-1 left, min_healthy=2


def test_flaky_hook_is_deterministic_with_seeded_rng():
    sim = _two_node_fleet()
    rng = np.random.default_rng(0)
    hook = make_flaky_hook(1.0, rng=rng)  # always fails
    action = {"action_id": "a1", "drive_id": "D-1", "proposed_action": ActionTier.DRAIN.value}

    result = apply_action(sim, action, fail_hook=hook)
    assert result["success"] is False
    assert result["compensating_action_triggered"] is True


def test_inject_stale_telemetry_flags_snapshot():
    sim = _two_node_fleet()
    inject_stale_telemetry(sim, ["D-1"])
    snapshot = sim.snapshot()
    by_id = {d["drive_id"]: d for d in snapshot["drives"]}
    assert by_id["D-1"]["stale_telemetry"] is True
    assert by_id["D-2"]["stale_telemetry"] is False


def test_inject_correlated_failure_fails_multiple_drives():
    sim = _two_node_fleet()
    affected = inject_correlated_failure(sim, "group-1", count=2)
    assert len(affected) == 2
    for drive_id in affected:
        assert sim.drives[drive_id].state == DriveState.FAILED
