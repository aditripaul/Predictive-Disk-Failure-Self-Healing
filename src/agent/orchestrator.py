"""Connects the compiled MAPE-K agent graph to the FastAPI audit/approval
store (src/api/store.py): every cycle's outcome — a pending human-review
action, or a completed decision with its provisional trust score — is
recorded so the dashboards and approval queue reflect real agent activity,
and approve/reject genuinely resumes the paused LangGraph thread rather
than only updating the store.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.api.store import InMemoryAuditStore, PendingAction
from src.config import ModelSettings
from src.reliability.audit import build_provisional_assessment


@dataclass
class AgentOrchestrator:
    app: CompiledStateGraph
    store: InMemoryAuditStore
    horizon_days: int = field(default_factory=lambda: ModelSettings.load().primary_horizon_days)

    def run_cycle(self, thread_id: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        # LangGraph's own examples use this exact {} + configurable-dict
        # invoke() shape; mypy's overload resolution for Pregel.invoke is
        # overly strict about the plain-dict config literal here.
        result = self.app.invoke({}, config=config)  # type: ignore[call-overload]
        return self._handle_result(thread_id, result)

    def resume_after_decision(
        self,
        action_id: str,
        *,
        approve: bool,
        operator_id: str,
        reason_code: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        action = self.store.get_action(action_id)
        if action is None:
            raise KeyError(f"Unknown action_id: {action_id}")

        # Record the human decision in the store first: if the resumed graph
        # invocation then raises, the audit trail still reflects that a
        # decision was made (fail-safe over fail-silent).
        self.store.decide_action(
            action_id,
            approve=approve,
            operator_id=operator_id,
            reason_code=reason_code,
            comment=comment,
        )

        config = {"configurable": {"thread_id": action.thread_id}}
        result = self.app.invoke(  # type: ignore[call-overload]
            Command(
                resume={
                    "approved": approve,
                    "operator_id": operator_id,
                    "reason_code": reason_code,
                    "comment": comment,
                }
            ),
            config=config,
        )
        return self._handle_result(action.thread_id, result)

    def _handle_result(self, thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            proposal = payload["proposal"]
            guardrail_result = payload.get("guardrail_result") or {}
            self.store.add_pending_action(
                PendingAction(
                    action_id=proposal["action_id"],
                    drive_id=proposal["drive_id"],
                    proposed_action=proposal["proposed_action"],
                    p_fail=proposal["p_fail"],
                    guardrail_violations=guardrail_result.get("violations", []),
                    fleet_snapshot_id=payload.get("fleet_snapshot_id") or "",
                    created_at=dt.datetime.now(dt.UTC).isoformat(),
                    thread_id=thread_id,
                )
            )
            return {"status": "pending_review", "action_id": proposal["action_id"], "state": result}

        assessment = build_provisional_assessment(result)
        if assessment["trust_score"] is not None:
            trust_score = assessment["trust_score"]
            timestamp = result.get("timestamp")
            # The prediction's "as of" drive-day date, used to join this
            # decision back to its resolved label once the horizon
            # completes (src/reliability/batch.py). This system doesn't yet
            # track a distinct observation date separate from wall-clock
            # time, so the cycle's timestamp date is used as an approximation.
            observed_date = (
                dt.datetime.fromisoformat(timestamp).date() if timestamp else dt.date.today()
            )
            self.store.add_decision(
                {
                    "decision_id": assessment["decision_id"],
                    "action_id": trust_score.action_id,
                    "drive_id": result["proposed_action"]["drive_id"],
                    "date": observed_date,
                    "horizon_days": self.horizon_days,
                    "timestamp": timestamp,
                    "proposed_action": result["proposed_action"]["proposed_action"],
                    "human_review_required": result.get("human_review_required", False),
                    "human_decision": result.get("human_decision"),
                    "override_reason_code": result.get("human_review_reason_code"),
                    "execution_result": result.get("execution_result"),
                    "validation_result": result.get("validation_result"),
                    "trust_score_provisional": trust_score.trust_score_provisional,
                    "trust_score_final": trust_score.trust_score_final,
                    "safety_violation": trust_score.safety_violation,
                    "guardrail_severity": trust_score.guardrail_severity.value,
                    "guardrail_result": result.get("guardrail_result"),
                    "post_action_guardrail_result": result.get("post_action_guardrail_result"),
                    "feature_explanations": assessment["explanation"],
                }
            )
        return {"status": "completed", "result": result, "state": result}
