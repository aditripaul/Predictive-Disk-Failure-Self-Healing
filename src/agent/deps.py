"""Injectable dependencies for the MAPE-K agent nodes.

Phase 6 defines these seams and ships simple in-memory defaults so the graph
is fully testable end-to-end. Phase 7 supplies the real guardrail evaluator,
Phase 8 supplies the real fleet simulator (state provider + executor).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

FleetStateProvider = Callable[[], dict[str, Any]]
Predictor = Callable[[dict[str, Any]], dict[str, Any]]
GuardrailEvaluator = Callable[[dict[str, Any]], dict[str, Any]]
PostActionGuardrailEvaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
#: Executor's return dict MUST include "drive_id" and "proposed_action"
#: (mirroring the fields on the action it was given), in addition to
#: "success"/"compensating_action_triggered"/"error" - the Validate node's
#: post-action guardrail check depends on both being present on
#: execution_result, not just on the original proposal.
Executor = Callable[[dict[str, Any]], dict[str, Any]]
Validator = Callable[[dict[str, Any]], dict[str, Any]]


class ActionLedger(Protocol):
    """Idempotency ledger: prevents duplicate execution on crash-recovery
    replay (docs/design_goal.md section 10, invariant table section 6)."""

    def is_executed(self, action_id: str) -> bool: ...

    def mark_executed(self, action_id: str, result: dict[str, Any]) -> None: ...

    def get_result(self, action_id: str) -> dict[str, Any] | None: ...


@dataclass
class InMemoryActionLedger:
    _executed: dict[str, dict[str, Any]] = field(default_factory=dict)

    def is_executed(self, action_id: str) -> bool:
        return action_id in self._executed

    def mark_executed(self, action_id: str, result: dict[str, Any]) -> None:
        self._executed[action_id] = result

    def get_result(self, action_id: str) -> dict[str, Any] | None:
        return self._executed.get(action_id)


class SqliteActionLedger:
    """Persistent idempotency ledger, backed by SQLite. Unlike
    InMemoryActionLedger, this survives a process crash/restart, which is
    the entire point of idempotent action IDs (docs/design_goal.md section
    5.4/10): a crash-recovery replay must be checked against durable state,
    not process memory that the crash itself just wiped out."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS action_ledger "
            "(action_id TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )
        self._conn.commit()

    def is_executed(self, action_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM action_ledger WHERE action_id = ?", (action_id,)
        ).fetchone()
        return row is not None

    def mark_executed(self, action_id: str, result: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO action_ledger (action_id, result_json) VALUES (?, ?)",
            (action_id, json.dumps(result)),
        )
        self._conn.commit()

    def get_result(self, action_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT result_json FROM action_ledger WHERE action_id = ?", (action_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()


def default_guardrail_pass_all(action_proposal: dict[str, Any]) -> dict[str, Any]:
    """Trivial pass-through guardrail used before Phase 7's real engine is
    wired in. Never use this for anything but tests/demos."""
    return {
        "action_id": action_proposal["action_id"],
        "passed": True,
        "violations": [],
        "final_action": action_proposal["proposed_action"],
    }


def _default_post_action_guardrail_evaluator() -> PostActionGuardrailEvaluator:
    # Local import to avoid a src.agent <-> src.guardrails import cycle at
    # module-load time (guardrails never imports agent).
    from src.guardrails.adapter import build_post_action_guardrail_evaluator

    return build_post_action_guardrail_evaluator()


@dataclass
class AgentDependencies:
    fleet_state_provider: FleetStateProvider
    predictor: Predictor
    guardrail_evaluator: GuardrailEvaluator
    executor: Executor
    validator: Validator
    action_ledger: ActionLedger = field(default_factory=InMemoryActionLedger)
    post_action_guardrail_evaluator: PostActionGuardrailEvaluator = field(
        default_factory=_default_post_action_guardrail_evaluator
    )
