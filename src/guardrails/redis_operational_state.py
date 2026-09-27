"""Redis-backed operational state (docs/design_goal.md section 14;
docs/project_plan.md: "Fleet-state cache | Redis | Live fleet state;
guardrail counters; cooldowns; active-action tracking | Low-latency
operational checks"). Implements the same interface as
`InMemoryOperationalState` (`src/guardrails/operational_state.py`) so
`build_guardrail_evaluator` can swap between them without touching the
engine - this is what makes the drain/rate-limit counters real shared
state across multiple API/agent process instances, rather than resetting
per process the way `InMemoryOperationalState` does. `redis` was a
declared but entirely unused dependency before this.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, cast

from data_contracts.schemas import ActionTier

ACTIVE_DRAINS_KEY = "guardrails:active_drains"
ACTION_TIMESTAMPS_KEY = "guardrails:action_timestamps"


class RedisLike(Protocol):
    """The narrow subset of the `redis.Redis` client this module needs -
    lets tests inject a lightweight fake instead of a real Redis server."""

    def scard(self, name: str) -> int: ...
    def sadd(self, name: str, *values: str) -> int: ...
    def srem(self, name: str, *values: str) -> int: ...
    def zadd(self, name: str, mapping: dict[str, float]) -> int: ...
    def zremrangebyscore(self, name: str, min: float | str, max: float | str) -> int: ...
    def zcard(self, name: str) -> int: ...


class RedisOperationalState:
    def __init__(self, redis_url: str | None = None, *, client: RedisLike | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            if redis_url is None:
                raise ValueError("redis_url is required when no client is provided")
            import redis

            self._client = cast(
                RedisLike, redis.Redis.from_url(redis_url, decode_responses=True)
            )

    def concurrent_drains(self) -> int:
        return int(self._client.scard(ACTIVE_DRAINS_KEY))

    def actions_in_last_hour(self, *, now: dt.datetime | None = None) -> int:
        now = now or dt.datetime.now(dt.UTC)
        cutoff = (now - dt.timedelta(hours=1)).timestamp()
        self._client.zremrangebyscore(ACTION_TIMESTAMPS_KEY, "-inf", cutoff)
        return int(self._client.zcard(ACTION_TIMESTAMPS_KEY))

    def record_action(
        self, action_id: str, proposed_action: ActionTier, *, now: dt.datetime | None = None
    ) -> None:
        now = now or dt.datetime.now(dt.UTC)
        # A timestamp-only member would collide for two actions recorded in
        # the same instant; suffixing with action_id keeps every entry
        # distinct so none are silently dropped by the sorted set.
        member = f"{now.timestamp()}:{action_id}"
        self._client.zadd(ACTION_TIMESTAMPS_KEY, {member: now.timestamp()})
        if proposed_action == ActionTier.DRAIN:
            self._client.sadd(ACTIVE_DRAINS_KEY, action_id)

    def complete_drain(self, action_id: str) -> None:
        self._client.srem(ACTIVE_DRAINS_KEY, action_id)
