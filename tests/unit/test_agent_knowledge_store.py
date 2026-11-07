"""src/agent/knowledge/store.py: the time-indexed knowledge store (docs/agent_knowledge_base.md
section 5). Self-checks (a), (b), (c), (f) and (j) of section 5.5 are tested here."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.agent.knowledge.store import (
    KnowledgeBase,
    KnowledgeConflict,
    derive_episodes,
    episode_outcome,
    outcome_as_of,
)

D0 = dt.date(2026, 1, 1)
META = {
    "run_id": "stub",
    "thresholds": {"warn": 0.6, "cordon": 0.8, "primary": 0.95},
    "candidate_threshold": 0.3,
    "horizon_days": 10,
    "episode_gap_days": 3,
    "context_window_days": 30,
}


def _fleet(n_drives=40, n_days=40, seed=0, fail_day=None):
    """Per day: (scored frame, presence frame, failures frame). Drive d0 is chronically
    high, d1 fails on `fail_day` (default day 25), d2 joins on day 10, d3 leaves on day 20."""
    rng = np.random.default_rng(seed)
    fail_day = 25 if fail_day is None else fail_day
    days = []
    for i in range(n_days):
        day = D0 + dt.timedelta(days=i)
        ids, ps = [], []
        for k in range(n_drives):
            if k == 2 and i < 10 or k == 3 and i >= 20 or k == 1 and i > fail_day:
                continue
            p = rng.uniform(0.0, 0.5)
            if k == 0:
                p = rng.uniform(0.7, 0.9)
            if k == 1 and fail_day - 6 <= i <= fail_day:
                p = rng.uniform(0.65, 0.99)
            ids.append(f"d{k}")
            ps.append(float(p))
        scored = pl.DataFrame({"drive_id": ids, "p_fail": ps})
        presence = scored.select(
            "drive_id",
            pl.lit("M1").alias("drive_model"),
            pl.lit("F1").alias("model_family"),
        )
        failures = pl.DataFrame(
            {"drive_id": ["d1"] if i == fail_day else []}, schema={"drive_id": pl.String}
        )
        days.append((day, scored, presence, failures))
    return days


def _build(days, kb=None, order=None):
    kb = kb or KnowledgeBase.in_memory(**META)
    for idx in order if order is not None else range(len(days)):
        day, scored, presence, failures = days[idx]
        kb.ingest_snapshot(day, scored, source="test", presence=presence, failures=failures)
    return kb


def test_only_candidate_days_are_stored_and_presence_is_kept_for_all():
    days = _fleet()
    kb = _build(days[:1])
    counts = kb.row_counts()
    scored = days[0][1]
    assert counts["candidate_days"] == int((scored["p_fail"] >= 0.3).sum())
    assert counts["presence_weeks"] == scored.height and counts["drive_info"] == scored.height


def test_a_store_refuses_a_different_scorer():
    with pytest.raises(ValueError, match="needs"):
        KnowledgeBase.in_memory(run_id="x")


def test_reopening_a_file_with_other_settings_is_refused(tmp_path):
    path = tmp_path / "kb.sqlite"
    KnowledgeBase(path, **META).close()
    KnowledgeBase(path).close()  # reopening without settings is fine
    with pytest.raises(ValueError, match="built with"):
        KnowledgeBase(path, **{**META, "horizon_days": 30})


def test_incremental_equals_batch_in_any_order():  # check (a)
    days = _fleet()
    forward = _build(days)
    rng = np.random.default_rng(3)
    shuffled = _build(days, order=list(rng.permutation(len(days))))
    assert forward.content_hash() == shuffled.content_hash()
    as_of = D0 + dt.timedelta(days=39)
    ids = [f"d{k}" for k in range(40)]
    assert forward.contexts(ids, as_of=as_of) == shuffled.contexts(ids, as_of=as_of)


def test_a_view_as_of_a_date_ignores_later_days():  # check (b)
    days = _fleet()
    ids = [f"d{k}" for k in range(40)]
    kb = KnowledgeBase.in_memory(**META)
    snapshots = {}
    for day, scored, presence, failures in days:
        kb.ingest_snapshot(day, scored, source="test", presence=presence, failures=failures)
        snapshots[day] = kb.contexts(ids, as_of=day)
    for day, before in snapshots.items():
        assert kb.contexts(ids, as_of=day) == before, day


def test_moving_a_failure_later_changes_nothing_before_it():  # check (c)
    early = _build(_fleet(fail_day=25))
    late = _build(_fleet(fail_day=25), kb=None)
    # rebuild with the failure recorded later: same scores, failure on day 33 instead
    days = _fleet(fail_day=25)
    days = [
        (d, s, p, pl.DataFrame({"drive_id": []}, schema={"drive_id": pl.String}))
        for d, s, p, _ in days
    ]
    day33 = days[33]
    days[33] = (day33[0], day33[1], day33[2], pl.DataFrame({"drive_id": ["d1"]}))
    late = _build(days)
    ids = [f"d{k}" for k in range(40)]
    for i in range(25):
        as_of = D0 + dt.timedelta(days=i)
        assert early.contexts(ids, as_of=as_of) == late.contexts(ids, as_of=as_of)


def test_reingesting_a_day_changes_nothing():  # check (f)
    days = _fleet()
    kb = _build(days)
    before = kb.content_hash()
    day, scored, presence, failures = days[17]
    stats = kb.ingest_snapshot(day, scored, source="test", presence=presence, failures=failures)
    assert stats.stored == 0 and stats.conflicts == 0
    assert kb.content_hash() == before


def test_sqlite_and_memory_stores_agree(tmp_path):  # check (j)
    days = _fleet()
    on_disk = _build(days, kb=KnowledgeBase(tmp_path / "kb.sqlite", **META))
    in_memory = _build(days)
    assert on_disk.content_hash() == in_memory.content_hash()


def test_a_changed_score_from_the_same_source_is_a_conflict():
    days = _fleet()
    kb = _build(days[:1])
    day, scored, presence, failures = days[0]
    changed = scored.with_columns(
        pl.when(pl.col("drive_id") == "d0").then(0.99).otherwise(pl.col("p_fail")).alias("p_fail")
    )
    with pytest.raises(KnowledgeConflict):
        kb.ingest_snapshot(day, changed, source="test", presence=presence)
    stats = kb.ingest_snapshot(day, changed, source="other", presence=presence)
    assert stats.conflicts == 1 and stats.stored == 0


def test_outcomes_follow_the_label_definition():
    d = dt.date(2026, 3, 1)
    late = dt.date(2026, 4, 1)
    assert outcome_as_of(d, 10, failure_date=dt.date(2026, 3, 5), last_presence=None, as_of=late)[
        :2
    ] == (1, "failure")
    assert outcome_as_of(d, 10, failure_date=None, last_presence=dt.date(2026, 3, 11), as_of=late)[
        :2
    ] == (0, "presence")
    assert outcome_as_of(d, 10, failure_date=None, last_presence=dt.date(2026, 3, 9), as_of=late)[
        :2
    ] == (None, "censored_removed")
    assert (
        outcome_as_of(
            d, 10, failure_date=None, last_presence=dt.date(2026, 3, 9), as_of=dt.date(2026, 3, 12)
        )[1]
        == "pending"
    )
    # a failure known only after as_of is not used
    assert (
        outcome_as_of(
            d, 10, failure_date=dt.date(2026, 3, 5), last_presence=None, as_of=dt.date(2026, 3, 4)
        )[1]
        == "pending"
    )
    # a failure after the horizon is not a positive
    assert (
        outcome_as_of(
            d, 10, failure_date=dt.date(2026, 3, 20), last_presence=dt.date(2026, 3, 19), as_of=late
        )[0]
        == 0
    )


def test_episodes_split_on_gaps_and_close_on_failure():
    day = lambda i: D0 + dt.timedelta(days=i)  # noqa: E731
    flags = [(day(0), 0.7), (day(1), 0.85), (day(3), 0.7), (day(10), 0.9)]
    eps = derive_episodes(flags, cordon=0.8, gap_days=3, failure_date=None)
    assert [(e.first_flag, e.last_flag, e.days_flagged, e.days_above_cordon) for e in eps] == [
        (day(0), day(3), 3, 1),
        (day(10), day(10), 1, 1),
    ]
    assert eps[0].closed_as_of == day(7)
    eps = derive_episodes(flags, cordon=0.8, gap_days=3, failure_date=day(12))
    assert eps[-1].closed_as_of == day(12)


def test_context_of_a_chronic_drive_and_a_new_drive():
    kb = _build(_fleet())
    as_of = D0 + dt.timedelta(days=39)
    ctx = kb.contexts(["d0", "d2", "nope"], as_of=as_of)
    assert ctx["d0"].open_episode_days == 39 and ctx["d0"].prior_episodes == 0
    assert ctx["d0"].days_above_warn_window == 30  # the 30-day window
    assert ctx["d2"].history_days == 29 and ctx["d2"].known
    assert not ctx["nope"].known and ctx["nope"].history_days == 0
    assert ctx["d0"].drive_model == "M1" and ctx["d0"].cohort_rate_window is not None


def test_the_failed_drives_episode_resolves_as_failed():
    kb = _build(_fleet())
    ctx = kb.context_for("d1", as_of=D0 + dt.timedelta(days=39))
    assert ctx.last_resolved_outcome == "failed" and ctx.open_episode_days is None


def test_resolve_outcomes_writes_known_labels_and_leaves_pending_ones():
    kb = _build(_fleet())
    first = kb.resolve_outcomes(D0 + dt.timedelta(days=39))
    assert first["failure"] > 0 and first["presence"] > 0 and first["pending"] > 0
    # d3 left on day 20: its earlier rows are censored, counted but not written
    assert first["censored_removed"] > 0
    assert not kb.db.execute("SELECT 1 FROM outcomes WHERE basis = 'censored_removed'").fetchall()
    again = kb.resolve_outcomes(D0 + dt.timedelta(days=39))
    # nothing new to write; censored rows are reconsidered each call, as they may resolve
    assert again["failure"] == 0 and again["presence"] == 0
    assert again["censored_removed"] == first["censored_removed"]
    labels = dict(
        ((d, day), label)
        for d, day, label in kb.db.execute(
            "SELECT drive_id, date, label FROM outcomes WHERE drive_id = 'd1'"
        )
    )
    # d1 fails on day 25 with a 10-day horizon: its candidate days from day 15 on are
    # positives, earlier ones (more than 10 days before the failure) are not
    fail = D0 + dt.timedelta(days=25)
    assert labels
    for (_, day), label in labels.items():
        gap = (fail - dt.date.fromisoformat(day)).days
        assert label == (1 if 0 < gap <= 10 else 0), (day, label)


def test_decisions_feed_final_trust_scores():
    kb = _build(_fleet())
    day = D0 + dt.timedelta(days=22)
    state = {
        "run_id": "r1",
        "timestamp": day.isoformat() + "T00:00:00+00:00",
        "decision_record_id": "dec-1",
        "human_review_required": False,
        "guardrail_result": {"passed": True},
        "proposed_action": {
            "action_id": "a1",
            "drive_id": "d1",
            "proposed_action": "cordon",
            "p_fail": 0.9,
            "as_of": day.isoformat(),
            "control_arm": False,
            "withheld_action": None,
            "knowledge_context": {"k": 1},
        },
    }
    assert kb.record_cycle(state, cycle=1) == "dec-1"
    kb.resolve_outcomes(D0 + dt.timedelta(days=39))
    assert kb.resolve_trust_scores() == 1
    (score,) = kb.db.execute(
        "SELECT trust_score_final FROM decisions WHERE decision_id = 'dec-1'"
    ).fetchone()
    assert score is not None and score > 0
    assert kb.record_cycle({"proposed_action": None}) is None


def test_annotate_coverage_and_drift():
    kb = _build(_fleet())
    as_of = D0 + dt.timedelta(days=39)
    out = kb.annotate([{"drive_id": "d0", "as_of": as_of.isoformat()}, {"drive_id": "d5"}])
    assert (
        out[0]["knowledge_context"]["open_episode_days"] == 39
        and out[1]["knowledge_context"] is None
    )
    cov = kb.coverage(as_of=as_of, fleet_drive_ids=["d0", "d2", "new"])
    assert cov["with_any_history"] == 2 and cov["share_with_any_history"] == pytest.approx(2 / 3)
    report = kb.drift_report(
        reference=(D0, D0 + dt.timedelta(days=19)), current=(D0 + dt.timedelta(days=20), as_of)
    )
    assert {"candidates_share", "warn_share", "cordon_share", "candidate_p_fail"} <= set(
        report["features"]
    )
    shares = kb.daily_shares(D0, as_of)
    assert shares.height == 40 and (shares["candidates_share"] > 0).all()


def test_an_episode_ending_on_its_failure_day_is_failed():
    day = lambda i: D0 + dt.timedelta(days=i)  # noqa: E731
    (episode,) = derive_episodes(
        [(day(0), 0.7), (day(5), 0.9)], cordon=0.8, gap_days=7, failure_date=day(5)
    )
    label, basis, resolved = episode_outcome(
        episode, 10, failure_date=day(5), last_presence=day(5), as_of=day(30)
    )
    assert (label, basis, resolved) == (1, "failure", day(5))
    # a failure before the episode began does not count for it
    label, basis, _ = episode_outcome(
        episode, 10, failure_date=day(0), last_presence=day(20), as_of=day(30)
    )
    assert (label, basis) == (0, "presence")


def test_a_censored_row_resolves_when_the_drive_comes_back():
    """A drive silent past its horizon's end plus the grace period is censored as of then,
    but a telemetry gap is not a removal: when it reports again the row is a 0."""
    kb = KnowledgeBase.in_memory(**META)
    flagged = pl.DataFrame({"drive_id": ["g"], "p_fail": [0.7]})
    kb.ingest_snapshot(D0, flagged, source="test")
    for i in range(1, 9):  # seen until day 8, then silent until day 30
        kb.ingest_snapshot(
            D0 + dt.timedelta(days=i),
            flagged.with_columns(pl.lit(0.1).alias("p_fail")),
            source="test",
        )
    as_of = D0 + dt.timedelta(days=25)
    assert kb.resolve_outcomes(as_of)["censored_removed"] >= 1
    ctx = kb.context_for("g", as_of=as_of)
    assert ctx.last_resolved_outcome == "censored"
    kb.ingest_snapshot(
        D0 + dt.timedelta(days=30), flagged.with_columns(pl.lit(0.1).alias("p_fail")), source="test"
    )
    later = D0 + dt.timedelta(days=30)
    kb.resolve_outcomes(later)
    (label,) = kb.db.execute(
        "SELECT label FROM outcomes WHERE drive_id = 'g' AND date = ?", (D0.isoformat(),)
    ).fetchone()
    assert label == 0
    assert kb.context_for("g", as_of=later).last_resolved_outcome == "survived_horizon"
    # the earlier view is unchanged: as of day 25 the drive was still missing
    assert kb.context_for("g", as_of=as_of).last_resolved_outcome == "censored"
