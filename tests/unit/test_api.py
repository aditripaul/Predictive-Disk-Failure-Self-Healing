import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from data_contracts.schemas import FeatureMaturity
from src.agent.deps import AgentDependencies
from src.agent.graph import build_agent_graph
from src.agent.orchestrator import AgentOrchestrator
from src.api.main import app, get_orchestrator
from src.api.store import InMemoryAuditStore, PendingAction, get_store
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
    "drive_id": "D-2",
    "p_fail": 0.05,
    "feature_confidence": 0.95,
    "feature_maturity": FeatureMaturity.MATURE,
    "stale_telemetry": False,
}


def _executor(action: dict) -> dict:
    return {
        "action_id": action["action_id"],
        "drive_id": action["drive_id"],
        "proposed_action": action["proposed_action"],
        "success": True,
    }


def _validator(execution_result: dict) -> dict:
    return {"data_integrity_ok": True, "service_continuity_ok": True, "quorum_ok": True}


def _build_orchestrator(store: InMemoryAuditStore, drives: list[dict]) -> AgentOrchestrator:
    deps = AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "snap-1", "drives": drives},
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=_executor,
        validator=_validator,
    )
    graph = build_agent_graph(
        deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS
    ).compile(checkpointer=InMemorySaver())
    return AgentOrchestrator(app=graph, store=store)


@pytest.fixture
def client():
    store = InMemoryAuditStore()
    orchestrator = _build_orchestrator(store, [LAST_NODE_DRIVE])
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    yield TestClient(app), store, orchestrator
    app.dependency_overrides.clear()


def test_fleet_state_404_before_any_report(client):
    test_client, _, _ = client
    response = test_client.get("/api/v1/fleet/state")
    assert response.status_code == 404


def test_fleet_state_roundtrip(client):
    test_client, store, _ = client
    store.set_fleet_state({"fleet_snapshot_id": "snap-1", "drives": []})
    response = test_client.get("/api/v1/fleet/state")
    assert response.status_code == 200
    assert response.json()["fleet_snapshot_id"] == "snap-1"


def test_latest_predictions_defaults_to_empty(client):
    test_client, _, _ = client
    response = test_client.get("/api/v1/predictions/latest")
    assert response.json() == {"predictions": []}


def test_run_cycle_blocked_action_appears_in_pending_queue(client):
    test_client, _, _ = client
    response = test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t1"})
    assert response.status_code == 200
    assert response.json()["status"] == "pending_review"

    pending = test_client.get("/api/v1/actions/pending").json()["actions"]
    assert len(pending) == 1
    assert pending[0]["drive_id"] == "D-1"

    fleet_state = test_client.get("/api/v1/fleet/state").json()
    assert fleet_state["fleet_snapshot_id"] == "snap-1"


def test_run_cycle_low_risk_completes_and_is_audited():
    store = InMemoryAuditStore()
    orchestrator = _build_orchestrator(store, [HEALTHY_DRIVE])
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        test_client = TestClient(app)
        response = test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t2"})
        assert response.json()["status"] == "completed"
        decisions = test_client.get("/api/v1/audit/decisions").json()["decisions"]
        assert len(decisions) == 1
    finally:
        app.dependency_overrides.clear()


def test_approve_action_requires_operator_and_reason_code(client):
    test_client, _, _ = client
    test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t3"})
    action_id = test_client.get("/api/v1/actions/pending").json()["actions"][0]["action_id"]

    missing_fields = test_client.post(f"/api/v1/actions/{action_id}/approve", json={})
    assert missing_fields.status_code == 422

    response = test_client.post(
        f"/api/v1/actions/{action_id}/approve",
        json={"operator_id": "op-1", "reason_code": "CONFIRMED_TRAJECTORY"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["status"] == "approved"
    assert body["decision"]["operator_id"] == "op-1"
    assert body["resume_outcome"]["status"] == "completed"

    # no longer in the pending list
    pending = test_client.get("/api/v1/actions/pending").json()["actions"]
    assert pending == []

    # and the action was actually executed against the fleet, not just marked
    decisions = test_client.get("/api/v1/audit/decisions").json()["decisions"]
    assert decisions[0]["execution_result"]["success"] is True


def test_reject_action_does_not_execute(client):
    test_client, _, _ = client
    test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t4"})
    action_id = test_client.get("/api/v1/actions/pending").json()["actions"][0]["action_id"]

    response = test_client.post(
        f"/api/v1/actions/{action_id}/reject",
        json={"operator_id": "op-1", "reason_code": "FALSE_POSITIVE"},
    )
    assert response.status_code == 200
    assert response.json()["decision"]["status"] == "rejected"

    decisions = test_client.get("/api/v1/audit/decisions").json()["decisions"]
    assert decisions[0]["execution_result"] is None


def test_reject_unknown_action_returns_404(client):
    test_client, _, _ = client
    response = test_client.post(
        "/api/v1/actions/unknown/reject",
        json={"operator_id": "op-1", "reason_code": "FALSE_POSITIVE"},
    )
    assert response.status_code == 404


def test_deciding_twice_returns_409(client):
    test_client, _, _ = client
    test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t5"})
    action_id = test_client.get("/api/v1/actions/pending").json()["actions"][0]["action_id"]

    body = {"operator_id": "op-1", "reason_code": "CONFIRMED_TRAJECTORY"}
    test_client.post(f"/api/v1/actions/{action_id}/approve", json=body)
    second = test_client.post(f"/api/v1/actions/{action_id}/approve", json=body)
    assert second.status_code == 409


def test_audit_decisions_and_trust_trend_read_endpoints(client):
    test_client, store, _ = client
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


def test_export_decisions_csv_and_parquet(client):
    test_client, store, _ = client
    store.add_decision(
        {
            "action_id": "a1",
            "drive_id": "D-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "trust_score_provisional": 0.9,
            "trust_score_final": None,
            "execution_result": {"success": True},
            "guardrail_result": {"violations": [{"rule_id": "HARD_QUORUM", "severity": "hard"}]},
        }
    )

    csv_response = test_client.get("/api/v1/audit/decisions/export?format=csv")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert b"a1" in csv_response.content
    assert b"success" in csv_response.content
    assert b"true" in csv_response.content.lower()

    parquet_response = test_client.get("/api/v1/audit/decisions/export?format=parquet")
    assert parquet_response.status_code == 200
    assert parquet_response.headers["content-disposition"].endswith('.parquet"')

    unsupported = test_client.get("/api/v1/audit/decisions/export?format=xml")
    assert unsupported.status_code == 400


def test_analytics_failure_rate_by_model_family_reads_gold_parquet(client, tmp_path, monkeypatch):
    import datetime as dt

    import polars as pl

    import src.api.main as api_main

    test_client, _, _ = client
    gold_dir = tmp_path / "gold"
    (gold_dir / "features").mkdir(parents=True)
    (gold_dir / "labels").mkdir(parents=True)

    pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "model_family": ["Seagate HDD", "WD HDD"],
        }
    ).write_parquet(gold_dir / "features" / "part.parquet")
    pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "horizon_days": [14, 14],
            "label": [1, 0],
        }
    ).write_parquet(gold_dir / "labels" / "part.parquet")

    monkeypatch.setattr(
        api_main, "load_yaml", lambda name: {"gold_dir": str(gold_dir)}
    )

    response = test_client.get("/api/v1/analytics/failure-rate-by-model-family?horizon_days=14")
    assert response.status_code == 200
    body = response.json()
    assert body["horizon_days"] == 14
    by_family = {row["model_family"]: row for row in body["failure_rate_by_model_family"]}
    assert by_family["Seagate HDD"]["failure_rate"] == 1.0
    assert by_family["WD HDD"]["failure_rate"] == 0.0


def test_analytics_failure_rate_returns_empty_before_gold_data_exists(
    client, tmp_path, monkeypatch
):
    import src.api.main as api_main

    test_client, _, _ = client
    monkeypatch.setattr(
        api_main, "load_yaml", lambda name: {"gold_dir": str(tmp_path / "nonexistent_gold")}
    )

    response = test_client.get("/api/v1/analytics/failure-rate-by-model-family?horizon_days=14")
    assert response.status_code == 200
    assert response.json()["failure_rate_by_model_family"] == []


def test_pending_actions_lists_only_pending(client):
    test_client, store, _ = client
    store.add_pending_action(
        PendingAction(
            action_id="a1",
            drive_id="D-1",
            proposed_action="drain",
            p_fail=0.9,
            guardrail_violations=[],
            fleet_snapshot_id="snap-1",
            created_at="2024-01-01T00:00:00Z",
            thread_id="unused",
        )
    )
    response = test_client.get("/api/v1/actions/pending")
    assert len(response.json()["actions"]) == 1


def test_run_cycle_populates_fleet_state_drives():
    store = InMemoryAuditStore()
    orchestrator = _build_orchestrator(store, [HEALTHY_DRIVE])
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    try:
        test_client = TestClient(app)
        test_client.post("/api/v1/agent/run-cycle", json={"thread_id": "t-fleet"})
        fleet_state = test_client.get("/api/v1/fleet/state").json()
    finally:
        app.dependency_overrides.clear()
    assert [d["drive_id"] for d in fleet_state["drives"]] == [HEALTHY_DRIVE["drive_id"]]
