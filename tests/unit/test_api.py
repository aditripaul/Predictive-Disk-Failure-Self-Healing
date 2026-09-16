import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.store import InMemoryAuditStore, PendingAction, get_store


@pytest.fixture
def client():
    store = InMemoryAuditStore()
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), store
    app.dependency_overrides.clear()


def test_fleet_state_404_before_any_report(client):
    test_client, _ = client
    response = test_client.get("/api/v1/fleet/state")
    assert response.status_code == 404


def test_fleet_state_roundtrip(client):
    test_client, store = client
    store.set_fleet_state({"fleet_snapshot_id": "snap-1", "drives": []})
    response = test_client.get("/api/v1/fleet/state")
    assert response.status_code == 200
    assert response.json()["fleet_snapshot_id"] == "snap-1"


def test_latest_predictions_defaults_to_empty(client):
    test_client, _ = client
    response = test_client.get("/api/v1/predictions/latest")
    assert response.json() == {"predictions": []}


def test_pending_actions_lists_only_pending(client):
    test_client, store = client
    store.add_pending_action(
        PendingAction(
            action_id="a1",
            drive_id="D-1",
            proposed_action="drain",
            p_fail=0.9,
            guardrail_violations=[],
            fleet_snapshot_id="snap-1",
            created_at="2024-01-01T00:00:00Z",
        )
    )
    response = test_client.get("/api/v1/actions/pending")
    assert len(response.json()["actions"]) == 1


def test_approve_action_requires_operator_and_reason_code(client):
    test_client, store = client
    store.add_pending_action(
        PendingAction(
            action_id="a1",
            drive_id="D-1",
            proposed_action="drain",
            p_fail=0.9,
            guardrail_violations=[],
            fleet_snapshot_id="snap-1",
            created_at="2024-01-01T00:00:00Z",
        )
    )

    missing_fields = test_client.post("/api/v1/actions/a1/approve", json={})
    assert missing_fields.status_code == 422

    response = test_client.post(
        "/api/v1/actions/a1/approve",
        json={"operator_id": "op-1", "reason_code": "CONFIRMED_TRAJECTORY"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["operator_id"] == "op-1"

    # no longer in the pending list
    pending = test_client.get("/api/v1/actions/pending").json()["actions"]
    assert pending == []


def test_reject_unknown_action_returns_404(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/actions/unknown/reject",
        json={"operator_id": "op-1", "reason_code": "FALSE_POSITIVE"},
    )
    assert response.status_code == 404


def test_deciding_twice_returns_409(client):
    test_client, store = client
    store.add_pending_action(
        PendingAction(
            action_id="a1",
            drive_id="D-1",
            proposed_action="drain",
            p_fail=0.9,
            guardrail_violations=[],
            fleet_snapshot_id="snap-1",
            created_at="2024-01-01T00:00:00Z",
        )
    )
    body = {"operator_id": "op-1", "reason_code": "CONFIRMED_TRAJECTORY"}
    test_client.post("/api/v1/actions/a1/approve", json=body)
    second = test_client.post("/api/v1/actions/a1/approve", json=body)
    assert second.status_code == 409


def test_audit_decisions_and_trust_trend(client):
    test_client, store = client
    store.add_decision(
        {
            "action_id": "a1",
            "timestamp": "2024-01-01T00:00:00Z",
            "trust_score_provisional": 0.9,
            "trust_score_final": None,
            "guardrail_result": {"violations": [{"rule_id": "HARD_QUORUM", "severity": "hard"}]},
        }
    )
    decisions = test_client.get("/api/v1/audit/decisions").json()["decisions"]
    assert len(decisions) == 1

    trend = test_client.get("/api/v1/reliability/trust-trend").json()["trust_trend"]
    assert trend[0]["trust_score_provisional"] == 0.9

    violations = test_client.get("/api/v1/guardrails/violations").json()["violations"]
    assert violations[0]["rule_id"] == "HARD_QUORUM"
