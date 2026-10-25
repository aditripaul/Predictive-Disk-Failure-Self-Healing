"""Missed-failure profiling, against fleets small enough to count by hand.

The point of `src/agent/missed_failures.py` is a recall ceiling, so each test
pins a way it could overstate or understate one: a null counter must not read
as a silent zero, a drive must take its best snapshot rather than its first,
and only positive snapshots may decide whether a failing drive was caught.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.agent.missed_failures import (
    class_base_rates,
    class_by_family,
    failing_drives,
    false_alarm_rate_to_catch,
    far_quantiles,
    recall_by_class,
    recall_ceiling,
    signal_class,
    window_columns,
)

ATTRS = ("reallocated_sector_count", "current_pending_sector_count")
T = dt.date(2026, 5, 3)


def _frame(rows):
    """rows: (drive_id, p_fail, label, (realloc max, range), (pending max, range))"""
    return pl.DataFrame(
        {
            "drive_id": [r[0] for r in rows],
            "date": [T] * len(rows),
            "p_fail": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "reallocated_sector_count_30d_max": [r[3][0] for r in rows],
            "reallocated_sector_count_30d_range": [r[3][1] for r in rows],
            "current_pending_sector_count_30d_max": [r[4][0] for r in rows],
            "current_pending_sector_count_30d_range": [r[4][1] for r in rows],
        },
        schema_overrides={"label": pl.Int8},
    )


def test_window_columns_names_max_and_range():
    assert window_columns(ATTRS[:1], 30) == [
        "reallocated_sector_count_30d_max",
        "reallocated_sector_count_30d_range",
    ]


def test_signal_class_four_ways():
    frame = _frame(
        [
            ("nodata", 0.1, 1, (None, None), (None, None)),
            ("silent", 0.1, 1, (0.0, 0.0), (None, None)),
            ("static", 0.1, 1, (8.0, 0.0), (0.0, 0.0)),
            ("active", 0.1, 1, (0.0, 0.0), (3.0, 2.0)),
        ]
    )
    got = dict(signal_class(frame, ATTRS).select("drive_id", "signal_class").iter_rows())
    assert got == {"nodata": "no_data", "silent": "silent", "static": "static", "active": "active"}


def test_null_counter_is_not_a_silent_zero():
    # One counter null, the other non-zero and moving: active, not dragged
    # down by the missing one.
    frame = _frame([("d", 0.1, 1, (None, None), (5.0, 1.0))])
    assert signal_class(frame, ATTRS)["signal_class"].item() == "active"


def test_signal_class_refuses_a_frame_without_window_columns():
    with pytest.raises(ValueError):
        signal_class(pl.DataFrame({"drive_id": ["a"]}), ATTRS)


def test_failing_drive_takes_its_best_class_and_highest_positive_score():
    obs = signal_class(
        pl.concat(
            [
                _frame([("d", 0.2, 1, (0.0, 0.0), (0.0, 0.0))]),
                _frame([("d", 0.6, 1, (4.0, 4.0), (0.0, 0.0))]).with_columns(
                    pl.lit(T + dt.timedelta(days=7)).alias("date")
                ),
            ]
        ),
        ATTRS,
    )
    drives = failing_drives(obs, threshold=0.5)
    row = drives.row(0, named=True)
    assert row["signal_class"] == "active"
    assert row["max_p_fail"] == pytest.approx(0.6)
    assert row["caught"] is True
    assert row["positive_snapshots"] == 2


def test_a_flag_before_the_warning_window_does_not_catch():
    # High score on a label-0 snapshot of a drive that later fails: not a catch.
    obs = signal_class(
        pl.concat(
            [
                _frame([("d", 0.9, 0, (1.0, 1.0), (0.0, 0.0))]),
                _frame([("d", 0.1, 1, (1.0, 0.0), (0.0, 0.0))]).with_columns(
                    pl.lit(T + dt.timedelta(days=40)).alias("date")
                ),
            ]
        ),
        ATTRS,
    )
    drives = failing_drives(obs, threshold=0.5)
    assert drives.height == 1
    assert drives["caught"].item() is False
    assert drives["signal_class"].item() == "static"


def _drives():
    obs = signal_class(
        _frame(
            [
                ("a1", 0.9, 1, (3.0, 2.0), (0.0, 0.0)),  # active, caught
                ("a2", 0.3, 1, (3.0, 2.0), (0.0, 0.0)),  # active, missed
                ("s1", 0.2, 1, (5.0, 0.0), (0.0, 0.0)),  # static, missed
                ("q1", 0.1, 1, (0.0, 0.0), (0.0, 0.0)),  # silent, missed
                ("q2", 0.1, 1, (0.0, 0.0), (0.0, 0.0)),  # silent, missed
                ("h1", 0.05, 0, (0.0, 0.0), (0.0, 0.0)),
                ("h2", 0.25, 0, (5.0, 0.0), (0.0, 0.0)),
                ("h3", 0.95, 0, (0.0, 0.0), (0.0, 0.0)),
                ("h4", 0.01, 0, (0.0, 0.0), (0.0, 0.0)),
            ]
        ),
        ATTRS,
    )
    return obs, failing_drives(obs, threshold=0.5)


def test_recall_by_class_counts_by_hand():
    _, drives = _drives()
    table = {r["signal_class"]: r for r in recall_by_class(drives).iter_rows(named=True)}
    assert table["active"]["failing_drives"] == 2
    assert table["active"]["recall"] == pytest.approx(0.5)
    assert table["silent"]["share_of_failing"] == pytest.approx(0.4)
    assert table["silent"]["share_of_missed"] == pytest.approx(0.5)
    assert table["no_data"]["failing_drives"] == 0
    assert table["no_data"]["recall"] is None


def test_recall_ceiling_counts_by_hand():
    _, drives = _drives()
    c = recall_ceiling(drives)
    assert c["model_recall"] == pytest.approx(0.2)
    assert c["ceiling_with_static"] == pytest.approx(0.6)
    assert c["ceiling_active_only"] == pytest.approx(0.4)


def test_class_base_rates_counts_by_hand():
    obs, _ = _drives()
    table = {r["signal_class"]: r for r in class_base_rates(obs).iter_rows(named=True)}
    # 4 healthy snapshots: 3 silent, 1 static. 5 failing: 2 active, 1 static, 2 silent.
    assert table["silent"]["share_of_healthy"] == pytest.approx(0.75)
    assert table["static"]["failure_rate"] == pytest.approx(0.5)
    assert "active" not in table or table["active"]["healthy"] == 0


def test_false_alarm_rate_to_catch():
    obs, drives = _drives()
    healthy = obs.filter(pl.col("label") == 0)["p_fail"].to_numpy()
    with_far = dict(
        false_alarm_rate_to_catch(drives, healthy).select("drive_id", "far_to_catch").iter_rows()
    )
    # healthy scores 0.01, 0.05, 0.25, 0.95
    assert with_far["a1"] == pytest.approx(0.25)  # only 0.95 beats 0.9
    assert with_far["a2"] == pytest.approx(0.25)
    assert with_far["s1"] == pytest.approx(0.5)  # 0.25 and 0.95 at or above 0.2
    assert with_far["q1"] == pytest.approx(0.5)


def test_false_alarm_rate_to_catch_needs_healthy_scores():
    _, drives = _drives()
    with pytest.raises(ValueError):
        false_alarm_rate_to_catch(drives, np.array([]))


def test_far_quantiles_only_missed():
    obs, drives = _drives()
    healthy = obs.filter(pl.col("label") == 0)["p_fail"].to_numpy()
    q = far_quantiles(false_alarm_rate_to_catch(drives, healthy))
    by_class = {r["signal_class"]: r for r in q.iter_rows(named=True)}
    assert by_class["active"]["missed"] == 1  # a1 was caught
    assert by_class["silent"]["missed"] == 2


def test_class_by_family_folds_small_families():
    _, drives = _drives()
    drives = drives.with_columns(pl.Series("drive_model", ["BIG", "BIG", "BIG", "BIG", "TINY"]))
    table = {
        r["drive_model"]: r for r in class_by_family(drives, min_failing=3).iter_rows(named=True)
    }
    assert set(table) == {"BIG", "(other)"}
    assert table["BIG"]["failing_drives"] == 4
    assert table["BIG"]["recall"] == pytest.approx(0.25)
