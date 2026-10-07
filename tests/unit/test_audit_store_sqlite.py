import pytest

from src.api.store import PendingAction, SQLiteAuditStore


def _action(action_id: str = "a1") -> PendingAction:
    return PendingAction(
        action_id=action_id,
        drive_id="drv-1",
        proposed_action="drain",
        p_fail=0.97,
        guardrail_violations=[{"rule": "score_cutoff", "severity": "hard"}],
        fleet_snapshot_id="snap-1",
        created_at="2026-10-07T00:00:00+00:00",
    )


def test_pending_decision_and_trail_survive_restart(tmp_path):
    path = tmp_path / "audit.sqlite"
    store = SQLiteAuditStore(path)
    store.add_pending_action(_action())
    store.add_decision(
        {
            "action_id": "a1",
            "timestamp": "2026-10-07T00:01:00+00:00",
            "guardrail_result": {"violations": [{"rule": "score_cutoff"}]},
            "trust_score_provisional": 0.4,
        }
    )
    store.decide_action("a1", approve=True, operator_id="op-1", reason_code="confirmed")
    store.close()

    reopened = SQLiteAuditStore(path)
    action = reopened.get_action("a1")
    assert action is not None
    assert action.status == "approved"
    assert action.operator_id == "op-1"
    assert len(reopened.list_decisions()) == 1
    assert reopened.list_guardrail_violations()[0]["action_id"] == "a1"
    reopened.close()


def test_double_decision_is_refused_and_not_persisted(tmp_path):
    path = tmp_path / "audit.sqlite"
    store = SQLiteAuditStore(path)
    store.add_pending_action(_action())
    store.decide_action("a1", approve=False, operator_id="op-1", reason_code="false_alarm")
    with pytest.raises(ValueError):
        store.decide_action("a1", approve=True, operator_id="op-2", reason_code="x")
    store.close()

    reopened = SQLiteAuditStore(path)
    assert reopened.get_action("a1").status == "rejected"
    assert reopened.get_action("a1").operator_id == "op-1"
    reopened.close()


def test_fleet_state_is_not_persisted(tmp_path):
    path = tmp_path / "audit.sqlite"
    store = SQLiteAuditStore(path)
    store.set_fleet_state({"nodes": 3})
    store.close()
    assert SQLiteAuditStore(path).get_fleet_state() is None
