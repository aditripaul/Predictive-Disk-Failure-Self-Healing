import polars as pl
import pytest

from src.features.velocity import add_defect_velocity


def _frame():
    # drive a: total defects 0,1,3,6 over four observations; drive b flat at 2.
    return pl.DataFrame(
        {
            "drive_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "date": [1, 2, 3, 4, 1, 2, 3, 4],
            "reallocated_sector_count": [0, 1, 2, 4, 2, 2, 2, 2],
            "current_pending_sector_count": [0, 0, 1, 2, 0, 0, 0, 0],
        }
    )


def test_active_defect_total_sums_the_defect_counters():
    out = add_defect_velocity(_frame(), windows_days=(1, 3))
    assert out["active_defect_total"].to_list() == [0, 1, 3, 6, 2, 2, 2, 2]


def test_velocity_is_change_per_observation_within_each_drive():
    out = add_defect_velocity(_frame(), windows_days=(1, 3))
    v1 = out["active_defect_velocity_1d"].to_list()
    assert v1[:4] == [0.0, 1.0, 2.0, 3.0]  # drive a: increments of 1 per observation
    assert v1[4:] == [0.0, 0.0, 0.0, 0.0]  # drive b: flat


def test_acceleration_compares_short_and_long_windows():
    out = add_defect_velocity(_frame(), windows_days=(1, 3))
    accel = out["active_defect_acceleration_1d_vs_3d"].to_list()
    assert accel[3] == pytest.approx(3.0 - 2.0)


def test_returns_the_frame_unchanged_without_defect_columns():
    frame = pl.DataFrame({"drive_id": ["a"], "date": [1], "temperature_celsius": [30.0]})
    assert add_defect_velocity(frame).equals(frame)


def test_acceleration_short_window_can_be_chosen_explicitly():
    out = add_defect_velocity(_frame(), windows_days=(1, 2, 3), acceleration_short_days=2)
    assert "active_defect_acceleration_2d_vs_3d" in out.columns
    with pytest.raises(ValueError, match="not one of"):
        add_defect_velocity(_frame(), windows_days=(1, 3), acceleration_short_days=2)


def test_temperature_spike_is_recent_max_minus_running_mean_per_drive():
    from src.features.velocity import add_temperature_spike

    frame = pl.DataFrame(
        {
            "drive_id": ["a"] * 4 + ["b"] * 2,
            "date": [1, 2, 3, 4, 1, 2],
            "temperature_celsius": [30.0, 30.0, 30.0, 42.0, 50.0, 50.0],
        }
    )
    spike = add_temperature_spike(frame, short_observations=2, long_observations=4)[
        "temperature_spike"
    ].to_list()
    assert spike[:3] == [0.0, 0.0, 0.0]
    assert spike[3] == pytest.approx(42.0 - (30 + 30 + 30 + 42) / 4)
    assert spike[4:] == [0.0, 0.0]  # drive b's window never sees drive a's rows
    assert add_temperature_spike(frame.drop("temperature_celsius")).columns == ["drive_id", "date"]


def test_days_since_last_increase_counts_from_the_last_rise_per_drive():
    import datetime as dt

    from src.features.velocity import NEVER_INCREASED_DAYS, add_days_since_last_increase

    days = [dt.date(2026, 1, d) for d in (1, 2, 3, 4, 5)]
    frame = pl.DataFrame(
        {
            "drive_id": ["a"] * 5 + ["b"] * 5,
            "date": days + days,
            # a: rises on day 2 and day 4. b: already 7 on its first day, never rises.
            "reallocated_sector_count": [0, 1, 1, 3, 3, 7, 7, 7, 7, 7],
        }
    )
    out = add_days_since_last_increase(frame, ["reallocated_sector_count", "not_present"])
    col = out["reallocated_sector_count_days_since_last_increase"].to_list()
    assert col[:5] == [NEVER_INCREASED_DAYS, 0, 1, 0, 1]
    assert col[5:] == [NEVER_INCREASED_DAYS] * 5
