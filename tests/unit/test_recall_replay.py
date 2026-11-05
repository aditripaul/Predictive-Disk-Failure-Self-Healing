"""The recall-first policy replay: thresholds, streaks, lead-time catches, tiers, late hits."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.agent.recall_replay import (
    first_dates,
    policy_metrics,
    streak_lengths,
    warn_threshold_for_recall,
)

D0 = dt.date(2026, 3, 1)


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n - 1)  # day(1) = 2026-03-01


def _daily(scores: dict[str, dict[int, float]], days: int = 25) -> pl.DataFrame:
    """Every drive scored on every day 1..days; `scores[drive]` lists the non-zero days."""
    rows = [
        (drive, day(n), scores[drive].get(n, 0.0))
        for drive in sorted(scores)
        for n in range(1, days + 1)
    ]
    return pl.DataFrame(rows, schema=["drive_id", "date", "s"], orient="row").sort(
        "drive_id", "date"
    )


def _failures(**dates: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"drive_id": list(dates), "failure_date": [day(n) for n in dates.values()]},
        schema={"drive_id": pl.String, "failure_date": pl.Date},
    )


TIERS = {"cordon": 0.8, "migrate": 0.9, "drain": 0.95}


def _metrics(daily, failures, *, k=1, warn=0.5, horizon=10, data_end=day(31)):
    return policy_metrics(
        daily,
        failures,
        warn=warn,
        tiers=TIERS,
        k=k,
        horizon_days=horizon,
        window=(day(1), day(25)),
        data_end=data_end,
    )


def test_warn_threshold_gives_at_least_the_target_recall() -> None:
    maxima = np.array([0.9, 0.8, 0.7, 0.2])
    assert warn_threshold_for_recall(maxima, 0.5) == 0.8
    assert warn_threshold_for_recall(maxima, 0.75) == 0.7
    assert warn_threshold_for_recall(maxima, 1.0) == 0.2
    for target in (0.1, 0.5, 0.6, 0.9):
        t = warn_threshold_for_recall(maxima, target)
        assert (maxima >= t).mean() >= target
    with pytest.raises(ValueError):
        warn_threshold_for_recall(np.array([]), 0.6)


def test_streak_lengths_restart_at_gaps_and_at_drive_boundaries() -> None:
    above = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    codes = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    assert streak_lengths(above, codes).tolist() == [1, 2, 0, 1, 2, 3, 1, 0]


def test_a_warning_on_the_failure_day_is_not_a_catch_and_lead_is_graded() -> None:
    daily = _daily(
        {
            "A": {19: 0.9},  # warned 1 day before: caught at lead 1, not at lead 7
            "B": {20: 0.9},  # warned only on the failure day: not caught
            "C": {10: 0.9, 19: 0.9},  # warned 10 days before: caught at both leads
            "H": {5: 0.9},  # healthy but warned once: a false alarm
        }
    )
    out = _metrics(daily, _failures(A=20, B=20, C=20))
    assert out["failing_drives_judged"] == 3
    assert out["warned_drives"] == 4  # A, B, C and H
    assert out["caught_drives"] == 2
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["recall_lead_7"] == pytest.approx(1 / 3)
    assert out["warn_precision"] == pytest.approx(2 / 4)
    assert out["median_lead_days"] == pytest.approx(5.5)  # leads 1 and 10


def test_failing_drive_with_no_scored_day_in_its_window_is_not_judged() -> None:
    daily = _daily({"A": {19: 0.9}, "Z": {}}, days=25)
    daily = daily.filter(~((pl.col("drive_id") == "Z") & (pl.col("date") >= day(5))))
    out = _metrics(daily, _failures(A=20, Z=20))
    assert out["failing_drives_in_range"] == 2 and out["failing_drives_judged"] == 1


def test_hysteresis_needs_consecutive_scored_days_above_the_tier() -> None:
    daily = _daily({"D": {11: 0.9, 15: 0.9, 16: 0.9}})
    assert first_dates(daily, 0.8, 1).row(0, named=True)["first"] == day(11)
    assert first_dates(daily, 0.8, 2).row(0, named=True)["first"] == day(16)
    out1 = _metrics(daily, _failures(D=18), k=1)
    out2 = _metrics(daily, _failures(D=18), k=2)
    assert out1["tiers"]["cordon"]["escalated"] == out2["tiers"]["cordon"]["escalated"] == 1
    assert out1["tiers"]["cordon"]["caught"] == out2["tiers"]["cordon"]["caught"] == 1
    assert out1["tiers"]["migrate"]["escalated"] == 1  # 0.9 clears the migrate cut-off too
    assert out1["tiers"]["drain"]["escalated"] == 0  # but not drain at 0.95
    assert out1["caught_then_escalated"] == 1.0


def test_escalation_after_the_failure_is_not_counted_as_caught_then_escalated() -> None:
    daily = _daily({"E": {10: 0.6, 21: 0.9}})  # warned at day 10, only escalated after failing
    out = _metrics(daily, _failures(E=20))
    assert out["caught_drives"] == 1 and out["caught_then_escalated"] == 0.0
    assert out["tiers"]["cordon"]["caught"] == 0


def test_late_hits_are_counted_only_where_sixty_days_are_observable() -> None:
    daily = _daily({"L": {5: 0.9}, "M": {25: 0.9}})
    failures = _failures(L=20)  # fails 15 days after its last warning
    out = _metrics(daily, failures, data_end=day(31), horizon=10)
    # cutoff = day 31 - 20 = day 11: L's last warning (day 5) is judged, M's (day 25) is not
    assert out["late_hits"] == {"judged_warned_drives": 1, "late_hits": 1}


def test_watchlist_and_daily_load() -> None:
    daily = _daily(
        {"A": {3: 0.6}, "B": {3: 0.6}, "C": {10: 0.6}}
    )  # warn (0.5) but not cordon (0.8)
    out = _metrics(daily, _failures(), horizon=5)
    assert out["carried_in_at_start"] == 0
    assert out["new_warns_peak_day"] == 2
    assert out["new_warns_per_day"] == pytest.approx(3 / 24)  # days 2..25
    assert out["watchlist_peak"] == 2  # A and B, still listed on days 6 and 7 (after warm-up)


def test_a_drive_escalated_on_the_day_it_is_warned_is_never_on_the_watchlist() -> None:
    daily = _daily({"A": {3: 0.99}, "B": {3: 0.6}})  # A clears cordon (0.8) at once, B only warns
    out = _metrics(daily, _failures(), horizon=5, k=1)
    assert out["watchlist_peak"] == 1  # B alone, days 3..7
    out2 = _metrics(daily, _failures(), horizon=5, k=2)
    assert out2["watchlist_peak"] == 2  # with k = 2 A is not escalated, so it stays on the list


def test_drives_above_warn_on_the_first_day_are_carried_in_not_new() -> None:
    daily = _daily({"X": dict.fromkeys(range(1, 26), 0.9), "Y": {4: 0.9}})
    out = _metrics(daily, _failures(), horizon=5)
    assert out["carried_in_at_start"] == 1  # X was already warned when the window opened
    assert out["new_warns_per_day"] == pytest.approx(1 / 24)  # only Y is a new warning
    assert out["new_warns_peak_day"] == 1


def test_tier_precision_uses_the_same_caught_rule_as_recall_so_tiers_nest() -> None:
    """A drive warned early AND again just before failing is caught for warn and for cordon."""
    daily = _daily({"P": {2: 0.9, 18: 0.9}, "Q": {18: 0.6}})
    out = _metrics(daily, _failures(P=20, Q=20), horizon=10)
    assert out["caught_drives"] == 2  # P (day 18) and Q (day 18) are both warned in the window
    cordon = out["tiers"]["cordon"]
    assert cordon["escalated"] == 1 and cordon["caught"] == 1  # P only, on day 18
    assert cordon["precision"] == 1.0
    assert out["tiers"]["cordon"]["caught"] <= out["caught_drives"]
    assert out["caught_then_escalated"] == pytest.approx(1 / 2)
