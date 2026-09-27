import datetime as dt

from data_contracts.schemas import ActionTier
from src.guardrails.redis_operational_state import (
    ACTION_TIMESTAMPS_KEY,
    ACTIVE_DRAINS_KEY,
    RedisOperationalState,
)


class FakeRedisClient:
    """A minimal in-memory stand-in for the subset of the real `redis.Redis`
    client RedisOperationalState uses, so these tests exercise the actual
    Redis command sequence/semantics without needing a real Redis server."""

    def __init__(self) -> None:
        self._sets: dict[str, set[str]] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}

    def scard(self, name: str) -> int:
        return len(self._sets.get(name, set()))

    def sadd(self, name: str, *values: str) -> int:
        self._sets.setdefault(name, set()).update(values)
        return len(values)

    def srem(self, name: str, *values: str) -> int:
        members = self._sets.get(name, set())
        removed = sum(1 for v in values if v in members)
        members.difference_update(values)
        return removed

    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        zset = self._sorted_sets.setdefault(name, {})
        added = sum(1 for k in mapping if k not in zset)
        zset.update(mapping)
        return added

    def zremrangebyscore(self, name: str, min, max) -> int:  # noqa: A002
        zset = self._sorted_sets.get(name, {})
        lower = float("-inf") if min == "-inf" else float(min)
        upper = float("inf") if max == "+inf" else float(max)
        to_remove = [k for k, score in zset.items() if lower <= score <= upper]
        for k in to_remove:
            del zset[k]
        return len(to_remove)

    def zcard(self, name: str) -> int:
        return len(self._sorted_sets.get(name, {}))


def test_concurrent_drains_reflects_active_drains_set():
    client = FakeRedisClient()
    state = RedisOperationalState(client=client)
    assert state.concurrent_drains() == 0

    state.record_action("a1", ActionTier.DRAIN)
    state.record_action("a2", ActionTier.DRAIN)
    assert state.concurrent_drains() == 2

    state.complete_drain("a1")
    assert state.concurrent_drains() == 1


def test_record_action_only_tracks_drains_in_active_drains_set():
    client = FakeRedisClient()
    state = RedisOperationalState(client=client)
    state.record_action("a1", ActionTier.WARN)
    state.record_action("a2", ActionTier.CORDON)
    assert state.concurrent_drains() == 0
    assert client.scard(ACTIVE_DRAINS_KEY) == 0


def test_actions_in_last_hour_prunes_entries_older_than_the_window():
    client = FakeRedisClient()
    state = RedisOperationalState(client=client)
    now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

    state.record_action("old", ActionTier.WARN, now=now - dt.timedelta(hours=2))
    state.record_action("recent", ActionTier.WARN, now=now - dt.timedelta(minutes=30))

    assert state.actions_in_last_hour(now=now) == 1
    # the stale entry should actually be gone from the sorted set, not just
    # excluded from the count
    assert client.zcard(ACTION_TIMESTAMPS_KEY) == 1


def test_actions_in_last_hour_counts_zero_with_no_recorded_actions():
    client = FakeRedisClient()
    state = RedisOperationalState(client=client)
    assert state.actions_in_last_hour() == 0


def test_record_action_never_drops_two_actions_at_the_same_instant():
    client = FakeRedisClient()
    state = RedisOperationalState(client=client)
    now = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    state.record_action("a1", ActionTier.WARN, now=now)
    state.record_action("a2", ActionTier.WARN, now=now)
    assert state.actions_in_last_hour(now=now) == 2


def test_missing_redis_url_and_no_client_raises_a_clear_error():
    import pytest

    with pytest.raises(ValueError, match="redis_url"):
        RedisOperationalState()
