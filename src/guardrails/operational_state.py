"""In-memory operational state for guardrail counters (docs/design_goal.md
section 14, "Redis-backed operational state"). This is the Phase 7 stand-in
for the real Redis-backed hot counters; the interface is small enough that a
Redis-backed implementation can replace it without touching the engine.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from data_contracts.schemas import ActionTier


@dataclass
class InMemoryOperationalState:
    _active_drains: set[str] = field(default_factory=set)
    _action_timestamps: list[dt.datetime] = field(default_factory=list)

    def concurrent_drains(self) -> int:
        return len(self._active_drains)

    def actions_in_last_hour(self, *, now: dt.datetime | None = None) -> int:
        now = now or dt.datetime.now(dt.UTC)
        cutoff = now - dt.timedelta(hours=1)
        self._action_timestamps = [t for t in self._action_timestamps if t >= cutoff]
        return len(self._action_timestamps)

    def record_action(
        self, action_id: str, proposed_action: ActionTier, *, now: dt.datetime | None = None
    ) -> None:
        self._action_timestamps.append(now or dt.datetime.now(dt.UTC))
        if proposed_action == ActionTier.DRAIN:
            self._active_drains.add(action_id)

    def complete_drain(self, action_id: str) -> None:
        self._active_drains.discard(action_id)
