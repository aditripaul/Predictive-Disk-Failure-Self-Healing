"""Injectable dependencies for the MAPE-K agent nodes.

Phase 6 defines these seams and ships simple in-memory defaults so the graph
is fully testable end-to-end. Phase 7 supplies the real guardrail evaluator,
Phase 8 supplies the real fleet simulator (state provider + executor).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

FleetStateProvider = Callable[[], dict[str, Any]]
Predictor = Callable[[dict[str, Any]], dict[str, Any]]
GuardrailEvaluator = Callable[[dict[str, Any]], dict[str, Any]]
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


def default_guardrail_pass_all(action_proposal: dict[str, Any]) -> dict[str, Any]:
    """Trivial pass-through guardrail used before Phase 7's real engine is
    wired in. Never use this for anything but tests/demos."""
    return {
        "action_id": action_proposal["action_id"],
        "passed": True,
        "violations": [],
        "final_action": action_proposal["proposed_action"],
    }


@dataclass
class AgentDependencies:
    fleet_state_provider: FleetStateProvider
    predictor: Predictor
    guardrail_evaluator: GuardrailEvaluator
    executor: Executor
    validator: Validator
    action_ledger: ActionLedger = field(default_factory=InMemoryActionLedger)
