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
