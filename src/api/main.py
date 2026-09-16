"""FastAPI approval and audit service (README.md section 20,
docs/design_goal.md section 15). Human review requires a mandatory
operator id and reason code (docs/design_goal.md section 15 "Human Review
Requirements").
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.store import InMemoryAuditStore, get_store

app = FastAPI(title="Disk Failure Self-Healing Agent — Approval & Audit API")


class ActionDecisionRequest(BaseModel):
    operator_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    comment: str | None = None


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
    store: InMemoryAuditStore = Depends(get_store),
):
    return _decide(store, action_id, approve=True, body=body)


@app.post("/api/v1/actions/{action_id}/reject")
def reject_action(
    action_id: str,
    body: ActionDecisionRequest,
    store: InMemoryAuditStore = Depends(get_store),
):
    return _decide(store, action_id, approve=False, body=body)


def _decide(
    store: InMemoryAuditStore, action_id: str, *, approve: bool, body: ActionDecisionRequest
):
    try:
        action = store.decide_action(
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
    return vars(action)


@app.get("/api/v1/audit/decisions")
def get_decisions(store: InMemoryAuditStore = Depends(get_store)):
    return {"decisions": store.list_decisions()}


@app.get("/api/v1/reliability/trust-trend")
def get_trust_trend(store: InMemoryAuditStore = Depends(get_store)):
    return {"trust_trend": store.trust_trend()}


@app.get("/api/v1/guardrails/violations")
def get_guardrail_violations(store: InMemoryAuditStore = Depends(get_store)):
    return {"violations": store.list_guardrail_violations()}
