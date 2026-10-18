"""False-positive profiling, against fleets small enough to count by hand.

The point of `src/agent/false_positives.py` is to decide whether a policy
could remove false positives the score cannot. Each measurement has a way to
mislead, and these tests pin those: late hits must only be counted where days
31-60 are fully observed, small groups must not masquerade as targets, and a
streak must not read as "first-time" merely because history is missing.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.agent.false_positives import (
    age_bucket,
    attribute_profile,
    flag_streaks,
    flagged_at,
    late_hit_dates,
    late_hits,
    precision_by_group,
)
from src.agent.point_in_time import COMBINERS, compare_looks

T = dt.date(2026, 4, 5)


def _obs(rows, day=T):
    """rows: (drive_id, p_fail, label, extras...)"""
    base = {
        "drive_id": [r[0] for r in rows],
        "date": [day] * len(rows),
        "p_fail": [r[1] for r in rows],
        "label": [r[2] for r in rows],
    }
    return pl.DataFrame(base, schema_overrides={"label": pl.Int8})


# --- late hits -----------------------------------------------------------------


def test_late_hits_are_only_counted_where_days_31_to_60_are_observed():
    dates = [dt.date(2026, 4, 5), dt.date(2026, 5, 1), dt.date(2026, 5, 2), dt.date(2026, 5, 31)]
    exact = late_hit_dates(dates, data_end=dt.date(2026, 6, 30), horizon_days=30)
    assert exact == [dt.date(2026, 4, 5), dt.date(2026, 5, 1)]


def test_a_false_alarm_that_fails_in_days_31_to_60_is_a_late_hit():
    flagged = _obs([("hit", 0.9, 1), ("late", 0.8, 0), ("wrong", 0.7, 0), ("wrong2", 0.6, 0)])
    # "late" is positive at T+30, i.e. fails in (T+30, T+60].
    future = pl.DataFrame({"drive_id": ["late"], "date": [T + dt.timedelta(days=30)]})
    result = late_hits(flagged, future, horizon_days=30)
    assert result["caught"] == 1
    assert result["false_alarms"] == 3
    assert result["late_hits"] == 1
    assert result["precision_30d"] == pytest.approx(0.25)
    assert result["precision_with_late_credit"] == pytest.approx(0.5)


def test_a_true_positive_is_not_double_counted_as_a_late_hit():
    """A drive positive at T AND at T+30 is a hit, not a hit plus a late hit."""
    flagged = _obs([("hit", 0.9, 1), ("miss", 0.5, 0)])
    future = pl.DataFrame({"drive_id": ["hit"], "date": [T + dt.timedelta(days=30)]})
    result = late_hits(flagged, future, horizon_days=30)
    assert result["late_hits"] == 0
    assert result["precision_with_late_credit"] == pytest.approx(0.5)


def test_a_late_positive_on_the_wrong_date_does_not_count():
    flagged = _obs([("x", 0.9, 0)])
    future = pl.DataFrame({"drive_id": ["x"], "date": [T + dt.timedelta(days=31)]})
    assert late_hits(flagged, future, horizon_days=30)["late_hits"] == 0


# --- composition -----------------------------------------------------------------


def test_precision_by_group_ranks_by_false_alarm_count_and_folds_small_groups():
    rows = (
        [(f"a{i}", 0.9, 1 if i < 10 else 0) for i in range(50)]  # A: 10/50
        + [(f"b{i}", 0.9, 1 if i < 20 else 0) for i in range(40)]  # B: 20/40
        + [(f"c{i}", 0.9, 0) for i in range(5)]  # C: 0/5, too small to trust
    )
    flagged = _obs(rows).with_columns(pl.Series("drive_model", ["A"] * 50 + ["B"] * 40 + ["C"] * 5))
    table = precision_by_group(flagged, "drive_model", min_flagged=30)
    assert table["drive_model"].to_list() == ["A", "B", "(other)"]
    a = table.row(0, named=True)
    assert (a["flagged"], a["caught"], a["false_alarms"]) == (50, 10, 40)
    assert a["precision"] == pytest.approx(0.2)
    assert a["share_of_fp"] == pytest.approx(40 / 65)


def test_a_missing_group_column_yields_an_empty_table():
    assert precision_by_group(_obs([("a", 0.9, 1)]), "drive_model").is_empty()


def test_age_buckets_from_power_on_hours():
    flagged = _obs([("a", 0.9, 1), ("b", 0.9, 0), ("c", 0.9, 0), ("d", 0.9, 0)]).with_columns(
        pl.Series("power_on_hours", [100.0, 8760 * 2.5, 8760 * 9.0, None])
    )
    assert age_bucket(flagged)["age_years"].to_list() == ["0-1y", "2-3y", "6y+", "unknown"]


# --- streaks ---------------------------------------------------------------------


def test_streaks_count_consecutive_earlier_flags_and_skip_dates_without_history():
    days = [T + dt.timedelta(days=7 * i) for i in range(5)]
    chronic = [("chronic", 0.9, 0)]  # flagged every week, never fails
    by_date = {
        days[0]: _obs(chronic + [("new", 0.1, 0)], days[0]),
        days[1]: _obs(chronic + [("new", 0.1, 0)], days[1]),
        days[2]: _obs(chronic + [("new", 0.1, 0)], days[2]),
        days[3]: _obs(chronic + [("new", 0.9, 1)], days[3]),  # newly flagged, fails
        days[4]: _obs(chronic + [("new", 0.9, 1)], days[4]),
    }
    streaks = flag_streaks(by_date, 0.5, cap=3)
    # Only days[3] and days[4] have 3 earlier snapshots.
    assert sorted(streaks["date"].unique().to_list()) == [days[3], days[4]]
    rows = {(r["drive_id"], r["date"]): r["streak"] for r in streaks.iter_rows(named=True)}
    assert rows[("chronic", days[3])] == 3
    assert rows[("new", days[3])] == 0
    assert rows[("new", days[4])] == 1


# --- attributes --------------------------------------------------------------------


def test_attribute_profile_compares_true_and_false_positives():
    flagged = _obs([("a", 0.9, 1), ("b", 0.9, 1), ("c", 0.9, 0), ("d", 0.9, 0)]).with_columns(
        pl.Series("reallocated_sector_count", [5.0, 0.0, 0.0, 0.0]),
        pl.Series("command_timeout", [0.0, 0.0, 3.0, np.nan]),
    )
    profile = {
        r["attribute"]: r
        for r in attribute_profile(
            flagged, ("reallocated_sector_count", "command_timeout", "absent")
        ).iter_rows(named=True)
    }
    assert set(profile) == {"reallocated_sector_count", "command_timeout"}
    assert profile["reallocated_sector_count"]["nonzero_in_true_positives"] == pytest.approx(0.5)
    assert profile["reallocated_sector_count"]["nonzero_in_false_positives"] == 0.0
    # NaN is missing, not zero: one of the one non-missing false positives.
    assert profile["command_timeout"]["nonzero_in_false_positives"] == pytest.approx(1.0)


def test_flagged_at_is_inclusive_of_the_threshold():
    assert flagged_at(_obs([("a", 0.5, 1), ("b", 0.49, 0)]), 0.5)["drive_id"].to_list() == ["a"]


# --- combiners ---------------------------------------------------------------------


def _pairs():
    rng = np.random.default_rng(0)
    n = 3000
    fails = rng.random(n) < 0.1
    cur = np.where(fails, rng.beta(6, 2, n), rng.beta(1.5, 6, n))
    prev = np.clip(cur - np.where(fails, 0.2, 0.0) + rng.normal(0, 0.05, n), 0, 1)
    return pl.DataFrame(
        {
            "drive_id": [f"d{i}" for i in range(n)],
            "p_prev": prev,
            "p_cur": cur,
            "label": fails.astype(np.int8),
        }
    )


@pytest.mark.parametrize("combiner", sorted(COMBINERS))
def test_every_combiner_runs_and_labels_its_rows(combiner):
    rows = compare_looks(_pairs(), combiner=combiner, budgets=(1e-2,), n_boot=20)
    assert rows[0]["combiner"] == combiner
    lo, hi = rows[0]["difference_ci"]
    assert lo <= rows[0]["difference"] <= hi + 1e-12


def test_a_rising_score_is_rewarded_where_min_penalises_it():
    """Failing drives here climb 0.2 between looks. `min` uses the older,
    lower score; `rising` adds the climb. So rising beats min."""
    pairs = _pairs()
    rising = compare_looks(pairs, combiner="rising", budgets=(1e-2,), n_boot=20)[0]
    minimum = compare_looks(pairs, combiner="min", budgets=(1e-2,), n_boot=20)[0]
    assert rising["two_look_recall"] > minimum["two_look_recall"]


def test_an_unknown_combiner_is_rejected():
    with pytest.raises(ValueError, match="unknown combiner"):
        compare_looks(_pairs(), combiner="median")
