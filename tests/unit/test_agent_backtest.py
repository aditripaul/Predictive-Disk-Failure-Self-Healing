"""Advancing the candidate pool, and scoring what the agent did.

The pool fix is a bug fix with a specific failure: the plan node takes
`max(drives, key=p_fail)` from a snapshot that never changes, so the same
drive is chosen every cycle forever. It escalates once, enters cooldown,
then caps at WARN on identical evidence while every lower-scoring drive is
never reached - and because the action ledger keys on (run_id, drive_id,
tier), the repeats are no-ops the executor never even sees.

The scorer exists because the model's 57.1% / 12.0% describe a threshold on
scores, not the agent. These tests pin that the agent's own numbers are
computed from actions actually taken, and that the two populations the
scorer must not confuse - control-arm drives and drives never reached - are
counted separately rather than as policy failures.
"""

import datetime as dt

import polars as pl
import pytest

from data_contracts.schemas import ActionTier
from pipelines.run_agent_fleet import advance_pool
from src.agent.backtest import failing_drives, load_labels, score_decisions

WINDOW = (dt.date(2026, 4, 1), dt.date(2026, 5, 31))


def _pool(n: int = 5) -> list[dict]:
    return [{"drive_id": f"d{i}", "p_fail": 0.9 - i * 0.1} for i in range(n)]


def _proposal(drive_id: str, action: str) -> dict:
    return {"drive_id": drive_id, "proposed_action": action}


def test_an_acted_drive_leaves_the_pool():
    """A cordoned drive is not a candidate next cycle - that is what happens
    in a live loop, and what lets the run reach drive #2."""
    pool, selections = _pool(), {}
    reason = advance_pool(
        pool, _proposal("d0", ActionTier.DRAIN.value), selections, max_selections=3
    )
    assert reason == "acted"
    assert [d["drive_id"] for d in pool] == ["d1", "d2", "d3", "d4"]


def test_a_warned_drive_stays_so_hysteresis_can_escalate_it():
    """Dropping on the first non-action would mean no drive ever reaches a
    destructive tier, because hysteresis caps the first cycle at WARN."""
    pool, selections = _pool(), {}
    reason = advance_pool(
        pool, _proposal("d0", ActionTier.WARN.value), selections, max_selections=3
    )
    assert reason is None
    assert [d["drive_id"] for d in pool] == ["d0", "d1", "d2", "d3", "d4"]


def test_a_permanently_blocked_drive_is_dropped_rather_than_stalling_the_run():
    """A drive the confidence floor or maturity gate will never let through
    stays the maximum forever, blocking every lower-scoring drive behind
    it."""
    pool, selections = _pool(), {}
    reasons = [
        advance_pool(pool, _proposal("d0", ActionTier.WARN.value), selections, max_selections=3)
        for _ in range(3)
    ]
    assert reasons == [None, None, "blocked"]
    assert [d["drive_id"] for d in pool] == ["d1", "d2", "d3", "d4"]


def test_the_run_walks_the_whole_pool_instead_of_spinning_on_one_drive():
    """The regression itself: without advancing, a fixed snapshot yields one
    decided drive no matter how many cycles run."""
    pool, selections, decided = _pool(4), {}, []
    for _ in range(20):
        if not pool:
            break
        top = max(pool, key=lambda d: d["p_fail"])
        decided.append(top["drive_id"])
        advance_pool(
            pool, _proposal(top["drive_id"], ActionTier.DRAIN.value), selections, max_selections=3
        )
    assert decided == ["d0", "d1", "d2", "d3"]
    assert pool == []


def _labels(failing: list[str], healthy: list[str]) -> pl.DataFrame:
    rows = [{"drive_id": d, "date": dt.date(2026, 4, 15), "label": 1} for d in failing] + [
        {"drive_id": d, "date": dt.date(2026, 4, 15), "label": 0} for d in healthy
    ]
    return pl.DataFrame(rows)


def test_agent_precision_and_recall_come_from_actions_actually_taken():
    labels = _labels(failing=["d0", "d1", "d2", "d3"], healthy=["h0", "h1"])
    decisions = [
        {"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value},
        {"drive_id": "h0", "proposed_action": ActionTier.CORDON.value},
        {"drive_id": "d1", "proposed_action": ActionTier.WARN.value},  # not an action
    ]
    scored = score_decisions(decisions, labels, candidate_count=10)
    assert scored["drives_acted_on"] == 2
    assert scored["caught"] == 1
    assert scored["agent_precision"] == pytest.approx(0.5)
    # One of four failing drives reached - WARN on d1 is not an action.
    assert scored["agent_recall"] == pytest.approx(0.25)


def test_a_drive_decided_twice_counts_once_at_its_strongest_action():
    """Hysteresis means a drive is typically WARNed before it is drained;
    counting both would double it and understate precision."""
    labels = _labels(failing=["d0"], healthy=["h0"])
    decisions = [
        {"drive_id": "d0", "proposed_action": ActionTier.WARN.value},
        {"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value},
    ]
    scored = score_decisions(decisions, labels, candidate_count=2)
    assert scored["drives_decided"] == 1
    assert scored["drives_acted_on"] == 1
    assert scored["agent_precision"] == pytest.approx(1.0)


def test_control_arm_drives_are_counted_separately_not_as_failures():
    """They were withheld on purpose; scoring them as misses would punish
    the mechanism that keeps their labels uncensored."""
    labels = _labels(failing=["d0", "d1"], healthy=[])
    decisions = [
        {
            "drive_id": "d0",
            "proposed_action": ActionTier.MONITOR.value,
            "control_arm": True,
            "withheld_action": ActionTier.DRAIN.value,
        },
        {"drive_id": "d1", "proposed_action": ActionTier.DRAIN.value},
    ]
    scored = score_decisions(decisions, labels, candidate_count=2)
    assert scored["attrition"]["withheld_by_control_arm"] == 1
    assert scored["drives_acted_on"] == 1
    assert scored["agent_precision"] == pytest.approx(1.0)


def test_drives_never_reached_are_attrition_not_policy_misses():
    """One action per cycle means --cycles bounds reach independently of any
    decision rule, so it must not read as the guardrails rejecting them."""
    labels = _labels(failing=["d0"], healthy=[])
    scored = score_decisions(
        [{"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value}],
        labels,
        candidate_count=100,
    )
    assert scored["attrition"]["never_selected"] == 99


def test_a_drive_without_an_observable_label_is_scored_neither_way():
    """An unobservable outcome is not evidence the action was wrong."""
    labels = _labels(failing=["d0"], healthy=[])
    decisions = [
        {"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value},
        {"drive_id": "unknown", "proposed_action": ActionTier.DRAIN.value},
    ]
    scored = score_decisions(decisions, labels, candidate_count=2)
    assert scored["drives_acted_on"] == 2
    assert scored["drives_acted_on_with_labels"] == 1
    assert scored["agent_precision"] == pytest.approx(1.0)


def test_censored_labels_are_dropped_rather_than_read_as_negatives(tmp_path):
    path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "drive_id": ["a", "b", "c"],
            "date": [dt.date(2026, 4, 10)] * 3,
            "horizon_days": [30, 30, 30],
            "label": [1, 0, None],
        },
        schema_overrides={"label": pl.Int8},
    ).write_parquet(path)
    labels = load_labels(path, horizon_days=30, start_date=WINDOW[0], end_date=WINDOW[1])
    assert sorted(labels["drive_id"].to_list()) == ["a", "b"]
    assert failing_drives(labels) == {"a"}


def test_only_the_configured_horizon_is_loaded(tmp_path):
    path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "drive_id": ["a", "a"],
            "date": [dt.date(2026, 4, 10)] * 2,
            "horizon_days": [7, 30],
            "label": [0, 1],
        },
        schema_overrides={"label": pl.Int8},
    ).write_parquet(path)
    labels = load_labels(path, horizon_days=30, start_date=WINDOW[0], end_date=WINDOW[1])
    assert labels.height == 1
    assert failing_drives(labels) == {"a"}


def test_truncated_drives_are_reported_separately_from_rejected_ones():
    """The defect this closes: a capped run reported never_selected=0, which
    reads as the filters rejecting nothing rather than the agent never
    having been shown the rest of the fleet."""
    labels = _labels(failing=["d0"], healthy=["h0"])
    scored = score_decisions(
        [{"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value}],
        labels,
        candidate_count=100,
        above_threshold_count=4000,
    )
    assert scored["drives_above_threshold"] == 4000
    assert scored["truncated_by_max_candidates"] == 3900
    # Within the pool the agent did reach everything but 99.
    assert scored["attrition"]["never_selected"] == 99


def test_the_pool_recall_ceiling_is_reported():
    """With fewer candidates than failing drives, recall is bounded by the
    pool before any decision rule applies - so the ceiling has to be stated
    next to the recall, or the number reads as a model limitation."""
    labels = _labels(failing=[f"d{i}" for i in range(300)], healthy=["h0"])
    scored = score_decisions(
        [{"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value}],
        labels,
        candidate_count=100,
        above_threshold_count=4000,
    )
    assert scored["recall_ceiling_from_pool"] == pytest.approx(100 / 300)
    assert scored["agent_recall"] <= scored["recall_ceiling_from_pool"]


def test_an_uncapped_pool_has_no_truncation_and_no_ceiling_below_one():
    labels = _labels(failing=["d0", "d1"], healthy=[])
    scored = score_decisions(
        [{"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value}],
        labels,
        candidate_count=500,
        above_threshold_count=500,
    )
    assert scored["truncated_by_max_candidates"] == 0
    assert scored["recall_ceiling_from_pool"] == pytest.approx(1.0)


def test_the_pre_cap_count_defaults_to_the_pool_when_not_supplied():
    """Older reports have no above_threshold_count; the scorer must not
    invent truncation that did not happen."""
    labels = _labels(failing=["d0"], healthy=[])
    scored = score_decisions(
        [{"drive_id": "d0", "proposed_action": ActionTier.DRAIN.value}],
        labels,
        candidate_count=10,
    )
    assert scored["drives_above_threshold"] == 10
    assert scored["truncated_by_max_candidates"] == 0
