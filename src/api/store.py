"""In-memory audit/approval store backing the FastAPI service
(docs/design_goal.md section 15, 19). A real deployment would back this
with Redis (hot operational state) and DuckDB/Postgres (durable audit
trail); this in-memory version keeps Phase 10 fully testable without
standing up either.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
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
    thread_id: str = ""  # LangGraph thread to resume on approve/reject
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
        violations = list((decision.get("guardrail_result") or {}).get("violations", []))
        violations += (decision.get("post_action_guardrail_result") or {}).get("violations", [])
        for v in violations:
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


_default_store: InMemoryAuditStore | None = None


def get_store() -> InMemoryAuditStore:
    """FastAPI dependency: returns the process-wide store singleton, backed
    by SQLite at AUDIT_DB_PATH (default data/audit/audit.sqlite)."""
    global _default_store
    if _default_store is None:
        _default_store = SQLiteAuditStore(default_audit_db_path())
    return _default_store


class SQLiteAuditStore(InMemoryAuditStore):
    """`InMemoryAuditStore` with the audit-critical state written through to
    SQLite: pending actions, their decisions, the decision trail and the
    guardrail violations. Those survive a restart. Fleet state and latest
    predictions stay in memory, since each agent cycle rebuilds them.

    Reads are served from memory, loaded from the database at construction,
    so the in-memory methods keep their behaviour exactly."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pending_actions (
                action_id TEXT PRIMARY KEY,
                body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT,
                body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS guardrail_violations (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT,
                body TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        for (body,) in self._conn.execute("SELECT body FROM pending_actions ORDER BY rowid"):
            action = PendingAction(**json.loads(body))
            self._pending_actions[action.action_id] = action
        for (body,) in self._conn.execute("SELECT body FROM decisions ORDER BY seq"):
            self._decisions.append(json.loads(body))
        for (body,) in self._conn.execute("SELECT body FROM guardrail_violations ORDER BY seq"):
            self._guardrail_violations.append(json.loads(body))

    def _save_action(self, action: PendingAction) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO pending_actions (action_id, body) VALUES (?, ?) "
                "ON CONFLICT(action_id) DO UPDATE SET body = excluded.body",
                (action.action_id, json.dumps(asdict(action), sort_keys=True)),
            )

    def add_pending_action(self, action: PendingAction) -> None:
        super().add_pending_action(action)
        self._save_action(action)

    def decide_action(
        self,
        action_id: str,
        *,
        approve: bool,
        operator_id: str,
        reason_code: str,
        comment: str | None = None,
    ) -> PendingAction:
        action = super().decide_action(
            action_id,
            approve=approve,
            operator_id=operator_id,
            reason_code=reason_code,
            comment=comment,
        )
        self._save_action(action)
        return action

    def add_decision(self, decision: dict) -> None:
        before = len(self._guardrail_violations)
        super().add_decision(decision)
        with self._conn:
            self._conn.execute(
                "INSERT INTO decisions (action_id, body) VALUES (?, ?)",
                (decision.get("action_id"), json.dumps(decision, sort_keys=True, default=str)),
            )
            for violation in self._guardrail_violations[before:]:
                self._conn.execute(
                    "INSERT INTO guardrail_violations (action_id, body) VALUES (?, ?)",
                    (
                        violation.get("action_id"),
                        json.dumps(violation, sort_keys=True, default=str),
                    ),
                )

    def close(self) -> None:
        self._conn.close()


def default_audit_db_path() -> Path:
    return Path(os.environ.get("AUDIT_DB_PATH", "data/audit/audit.sqlite"))
