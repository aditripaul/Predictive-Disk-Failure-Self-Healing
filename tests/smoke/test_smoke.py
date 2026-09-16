"""Smoke tests: fast, shallow checks that the system *as shipped* actually
boots and does something end-to-end, using the real default wiring
(`build_demo_dependencies`) rather than hand-rolled fixtures — as opposed to
`tests/golden` (dataset-generation regression) or the deeper unit/
integration/chaos suites. Run via `make smoke`.

These never touch the real project runtime files under `mlflow/`: the
action ledger is swapped for an in-memory one and the LangGraph checkpoint
uses the default `:memory:` path, so running `make smoke` repeatedly (e.g.
in CI) has zero on-disk side effects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.demo import build_demo_dependencies
from src.agent.deps import InMemoryActionLedger
from src.agent.graph import build_agent_graph, compiled_agent
from src.agent.orchestrator import AgentOrchestrator
from src.api.main import app, get_orchestrator
from src.api.store import InMemoryAuditStore, get_store
from tests.support import IMMEDIATE_ESCALATION_PLAN_KWARGS

DOCUMENTED_GET_ENDPOINTS = [
    "/api/v1/fleet/state",
    "/api/v1/predictions/latest",
    "/api/v1/actions/pending",
    "/api/v1/audit/decisions",
    "/api/v1/reliability/trust-trend",
    "/api/v1/guardrails/violations",
]


def _isolated_demo_deps():
    return build_demo_dependencies(action_ledger=InMemoryActionLedger())


def test_agent_demo_wiring_completes_one_cycle_end_to_end():
    """The exact dependency wiring `make agent-demo` and the API's default
    lifespan use, run through one real MAPE-K cycle."""
    deps = _isolated_demo_deps()
    with compiled_agent(deps, plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS) as agent_app:
        result = agent_app.invoke({}, config={"configurable": {"thread_id": "smoke"}})

    assert result["decision_record_id"]
    assert result["proposed_action"] is not None
    assert result["guardrail_result"] is not None


@pytest.fixture
def smoke_client():
    store = InMemoryAuditStore()
    graph = build_agent_graph(
        _isolated_demo_deps(), plan_kwargs=IMMEDIATE_ESCALATION_PLAN_KWARGS
    ).compile(checkpointer=InMemorySaver())
    orchestrator = AgentOrchestrator(app=graph, store=store)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_boots_and_serves_every_documented_endpoint(smoke_client):
    """One linear walk through every endpoint in docs/developer_guide.md's
    and docs/user_guide.md's API tables, against a live run's output."""
    run = smoke_client.post("/api/v1/agent/run-cycle", json={"thread_id": "smoke"})
    assert run.status_code == 200
    assert run.json()["status"] in {"completed", "pending_review"}

    for path in DOCUMENTED_GET_ENDPOINTS:
        response = smoke_client.get(path)
        assert response.status_code == 200, f"{path} returned {response.status_code}"


def test_dashboard_module_has_valid_syntax():
    """Streamlit apps aren't meaningfully exercisable under pytest (no
    browser session), but a syntax error in it should still fail CI."""
    source = Path("src/dashboards/app.py").read_text()
    ast.parse(source)
