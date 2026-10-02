"""FastAPI approval and audit service (README.md section 20,
docs/design_goal.md section 15). Human review requires a mandatory
operator id and reason code (docs/design_goal.md section 15 "Human Review
Requirements").

The compiled agent graph is opened once for the process lifetime (via the
lifespan handler below) rather than per-request, since LangGraph threads
must persist across the "propose -> pending review -> approve/reject ->
resume" request sequence. It defaults to the same demo wiring as
`make agent-demo` (src/agent/demo.py) until a real fleet simulator/model are
plugged in for a given deployment; override `app.state.orchestrator` (or the
`get_orchestrator` dependency, in tests) to use real ones.
"""

from __future__ import annotations

from contextlib import ExitStack, asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from src.agent.demo import build_demo_dependencies
from src.agent.graph import build_agent_graph
from src.agent.orchestrator import AgentOrchestrator
from src.api.store import InMemoryAuditStore, get_store
from src.config import AgentSettings, ModelSettings, load_yaml
from src.reliability.analytics import failure_rate_by_model_family
from src.reliability.audit_export import export_decisions_to_csv, export_decisions_to_parquet


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deliberately no apply_memory_limit_from_config() call here: this
    # module (and its `app` object) is imported directly by the test
    # suite via FastAPI's TestClient, which really executes this
    # lifespan - applying a memory cap here would cap the *test runner's
    # own process*, not a real deployment's. `make api` applies the limit
    # in pipelines/run_api.py instead, which the test suite never imports.
    #
    # A real (not :memory:) checkpoint path, so a paused (pending-review)
    # thread survives an API process restart, not just a fresh request.
    checkpoint_path = AgentSettings.load().checkpoint_path
    with ExitStack() as stack:
        checkpointer = stack.enter_context(SqliteSaver.from_conn_string(checkpoint_path))
        deps = build_demo_dependencies()
        compiled = build_agent_graph(deps).compile(checkpointer=checkpointer)
        app.state.orchestrator = AgentOrchestrator(app=compiled, store=get_store())
        yield


app = FastAPI(
    title="Disk Failure Self-Healing Agent — Approval & Audit API", lifespan=lifespan
)


def get_orchestrator() -> AgentOrchestrator:
    """FastAPI dependency; overridden in tests to avoid the real demo graph."""
    return app.state.orchestrator


class ActionDecisionRequest(BaseModel):
    operator_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    comment: str | None = None


class RunCycleRequest(BaseModel):
    thread_id: str = Field(min_length=1, default="default-fleet")


@app.post("/api/v1/agent/run-cycle")
def run_cycle(
    body: RunCycleRequest = RunCycleRequest(),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    store: InMemoryAuditStore = Depends(get_store),
):
    """Runs one MAPE-K cycle and records its outcome (pending review, or a
    trust-scored decision) to the audit/approval store. Also refreshes the
    fleet-state/predictions read endpoints below."""
    outcome = orchestrator.run_cycle(body.thread_id)
    state = outcome.get("state", {})
    prediction_drives = state.get("prediction_summary", {}).get("drives", [])
    if "fleet_snapshot_id" in state:
        store.set_fleet_state(
            {"fleet_snapshot_id": state["fleet_snapshot_id"], "drives": prediction_drives}
        )
    if "prediction_summary" in state:
        store.set_latest_predictions(prediction_drives)
    return outcome


@app.get("/api/v1/fleet/state")
def get_fleet_state(store: InMemoryAuditStore = Depends(get_store)):
    state = store.get_fleet_state()
    if state is None:
        raise HTTPException(status_code=404, detail="No fleet state has been reported yet.")
    return state


@app.get("/api/v1/predictions/latest")
def get_latest_predictions(store: InMemoryAuditStore = Depends(get_store)):
    return {"predictions": store.get_latest_predictions()}


@app.get("/api/v1/actions/pending")
def get_pending_actions(store: InMemoryAuditStore = Depends(get_store)):
    return {"actions": [vars(a) for a in store.list_pending_actions()]}


@app.post("/api/v1/actions/{action_id}/approve")
def approve_action(
    action_id: str,
    body: ActionDecisionRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    return _decide(orchestrator, action_id, approve=True, body=body)


@app.post("/api/v1/actions/{action_id}/reject")
def reject_action(
    action_id: str,
    body: ActionDecisionRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    return _decide(orchestrator, action_id, approve=False, body=body)


def _decide(
    orchestrator: AgentOrchestrator,
    action_id: str,
    *,
    approve: bool,
    body: ActionDecisionRequest,
):
    try:
        outcome = orchestrator.resume_after_decision(
            action_id,
            approve=approve,
            operator_id=body.operator_id,
            reason_code=body.reason_code,
            comment=body.comment,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    action = orchestrator.store.get_action(action_id)
    return {"decision": vars(action), "resume_outcome": outcome}


@app.get("/api/v1/audit/decisions")
def get_decisions(store: InMemoryAuditStore = Depends(get_store)):
    return {"decisions": store.list_decisions()}


@app.get("/api/v1/audit/decisions/export")
def export_decisions(
    format: str = "csv",  # noqa: A002 - matches the query param name in docs/user_guide.md
    store: InMemoryAuditStore = Depends(get_store),
):
    """Exports the full decision audit trail (docs/project_plan.md Phase 10
    Key Task 6). `format` is `csv` (default) or `parquet`."""
    decisions = store.list_decisions()
    if format == "csv":
        body = export_decisions_to_csv(decisions)
        media_type = "text/csv"
        filename = "audit_decisions.csv"
    elif format == "parquet":
        body = export_decisions_to_parquet(decisions)
        media_type = "application/octet-stream"
        filename = "audit_decisions.parquet"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format!r}")

    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/analytics/failure-rate-by-model-family")
def get_failure_rate_by_model_family(horizon_days: int | None = None):
    """Reads analytics from DuckDB (docs/project_plan.md Phase 10 Key Task
    3), querying the gold features/labels Parquet files directly on disk
    rather than through the in-memory audit store. Returns `[]` before
    `make build-features`/`make build-labels` have ever been run."""
    data_config = load_yaml("data.yaml")
    gold_dir = Path(data_config["gold_dir"])
    resolved_horizon = (
        horizon_days if horizon_days is not None else ModelSettings.load().primary_horizon_days
    )
    rows = failure_rate_by_model_family(
        gold_dir / "features" / "part.parquet",
        gold_dir / "labels" / "part.parquet",
        horizon_days=resolved_horizon,
    )
    return {"horizon_days": resolved_horizon, "failure_rate_by_model_family": rows}


@app.get("/api/v1/reliability/trust-trend")
def get_trust_trend(store: InMemoryAuditStore = Depends(get_store)):
    return {"trust_trend": store.trust_trend()}


@app.get("/api/v1/guardrails/violations")
def get_guardrail_violations(store: InMemoryAuditStore = Depends(get_store)):
    return {"violations": store.list_guardrail_violations()}
