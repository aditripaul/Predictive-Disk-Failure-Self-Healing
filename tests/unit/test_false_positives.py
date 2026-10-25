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


def test_lift_separates_a_weak_model_from_a_rare_failure():
    """Two families with the same 20% precision. In R the family fails
    rarely (2%), so 20% is ten times the base rate: the model ranks it well.
    In W the family fails often (20%), so 20% is no better than chance: the
    model has nothing there. Precision alone cannot tell them apart."""

    def family(name, n, failing, flagged, caught):
        rows = []
        for i in range(n):
            fails = i < failing
            # The first `flagged` rows are flagged; arrange `caught` of them to fail.
            if i < flagged:
                fails = i < caught
            elif i < flagged + (failing - caught):
                fails = True
            else:
                fails = False
            rows.append((f"{name}{i}", 0.9 if i < flagged else 0.1, int(fails)))
        return rows

    obs = _obs(family("r", 1000, 20, 50, 10) + family("w", 250, 50, 50, 10)).with_columns(
        pl.Series("drive_model", ["R"] * 1000 + ["W"] * 250)
    )
    table = precision_by_group(flagged_at(obs, 0.5), "drive_model", population=obs, min_flagged=10)
    rows = {r["drive_model"]: r for r in table.iter_rows(named=True)}
    assert rows["R"]["precision"] == pytest.approx(0.2)
    assert rows["W"]["precision"] == pytest.approx(0.2)
    assert rows["R"]["base_rate"] == pytest.approx(0.02)
    assert rows["W"]["base_rate"] == pytest.approx(0.2)
    assert rows["R"]["lift"] == pytest.approx(10.0)
    assert rows["W"]["lift"] == pytest.approx(1.0)
    assert rows["R"]["recall"] == pytest.approx(0.5)


def test_without_a_population_there_is_no_lift_column():
    flagged = _obs([("a", 0.9, 1)] * 40).with_columns(pl.Series("drive_model", ["A"] * 40))
    assert "lift" not in precision_by_group(flagged, "drive_model", min_flagged=10).columns


def test_plr_matches_a_hand_count_and_ignores_the_base_rate():
    """R: 10 caught of 20 failing (recall 0.5), 40 false alarms of 980
    healthy. W: 10 of 50 (0.2), 40 of 200. PLR = recall / FAR."""

    def rows(name, n, failing, flagged, caught):
        out = []
        for i in range(n):
            if i < flagged:
                fails = i < caught
            else:
                fails = i < flagged + (failing - caught)
            out.append((f"{name}{i}", 0.9 if i < flagged else 0.1, int(fails)))
        return out

    obs = _obs(rows("r", 1000, 20, 50, 10) + rows("w", 250, 50, 50, 10)).with_columns(
        pl.Series("drive_model", ["R"] * 1000 + ["W"] * 250)
    )
    table = precision_by_group(flagged_at(obs, 0.5), "drive_model", population=obs, min_flagged=10)
    plr = {r["drive_model"]: r["plr"] for r in table.iter_rows(named=True)}
    assert plr["R"] == pytest.approx(0.5 / (40 / 980))
    assert plr["W"] == pytest.approx(0.2 / (40 / 200))
