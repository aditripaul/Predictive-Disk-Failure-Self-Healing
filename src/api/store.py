"""In-memory audit/approval store backing the FastAPI service
(docs/design_goal.md section 15, 19). A real deployment would back this
with Redis (hot operational state) and DuckDB/Postgres (durable audit
trail); this in-memory version keeps Phase 10 fully testable without
standing up either.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any


@dataclass
class PendingAction:
    action_id: str
    drive_id: str
    proposed_action: str
    p_fail: float
    guardrail_violations: list[dict]
    fleet_snapshot_id: str
    created_at: str
    status: str = "pending"  # pending | approved | rejected
    operator_id: str | None = None
    reason_code: str | None = None
    comment: str | None = None
    decided_at: str | None = None


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._fleet_state: dict[str, Any] | None = None
        self._latest_predictions: list[dict] = []
        self._pending_actions: dict[str, PendingAction] = {}
        self._decisions: list[dict] = []
        self._guardrail_violations: list[dict] = []

    # --- fleet / predictions -------------------------------------------------

    def set_fleet_state(self, fleet_state: dict[str, Any]) -> None:
        self._fleet_state = fleet_state

    def get_fleet_state(self) -> dict[str, Any] | None:
        return self._fleet_state

    def set_latest_predictions(self, predictions: list[dict]) -> None:
        self._latest_predictions = predictions

    def get_latest_predictions(self) -> list[dict]:
        return self._latest_predictions

    # --- approval queue -------------------------------------------------------

    def add_pending_action(self, action: PendingAction) -> None:
        self._pending_actions[action.action_id] = action

    def list_pending_actions(self) -> list[PendingAction]:
        return [a for a in self._pending_actions.values() if a.status == "pending"]

    def get_action(self, action_id: str) -> PendingAction | None:
        return self._pending_actions.get(action_id)

    def decide_action(
        self,
        action_id: str,
        *,
        approve: bool,
        operator_id: str,
        reason_code: str,
        comment: str | None = None,
    ) -> PendingAction:
        action = self._pending_actions.get(action_id)
        if action is None:
            raise KeyError(f"Unknown action_id: {action_id}")
        if action.status != "pending":
            raise ValueError(f"Action {action_id} already {action.status}")

        action.status = "approved" if approve else "rejected"
        action.operator_id = operator_id
        action.reason_code = reason_code
        action.comment = comment
        action.decided_at = dt.datetime.now(dt.UTC).isoformat()
        return action

    # --- audit trail ------------------------------------------------------

    def add_decision(self, decision: dict) -> None:
        self._decisions.append(decision)
        for v in decision.get("guardrail_result", {}).get("violations", []):
            self._guardrail_violations.append({**v, "action_id": decision.get("action_id")})

    def list_decisions(self) -> list[dict]:
        return list(self._decisions)

    def list_guardrail_violations(self) -> list[dict]:
        return list(self._guardrail_violations)

    def trust_trend(self) -> list[dict]:
        return [
            {
                "action_id": d.get("action_id"),
                "timestamp": d.get("timestamp"),
                "trust_score_provisional": d.get("trust_score_provisional"),
                "trust_score_final": d.get("trust_score_final"),
            }
            for d in self._decisions
            if "trust_score_provisional" in d
        ]


_default_store = InMemoryAuditStore()


def get_store() -> InMemoryAuditStore:
    """FastAPI dependency: returns the process-wide store singleton."""
    return _default_store
