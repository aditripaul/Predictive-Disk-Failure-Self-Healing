import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.features.sequence_windows import (
    build_dense_telemetry,
    channel_count,
    fit_normalization,
    gather_windows,
    static_feature_count,
)

START = dt.date(2026, 1, 1)
END = dt.date(2026, 1, 20)
ATTRS = ["reallocated_sector_count", "temperature_celsius"]


def _write_gold(tmp_path):
    rows = []
    for day in range(20):
        date = START + dt.timedelta(days=day)
        # drive a: reports both attributes every day, counter grows 0,1,2,...
        rows.append(("a", date, float(day), 30.0 + day, 100 + day, "Seagate", 4000.0))
        # drive b: skips days 5-9, never reports temperature
        if not 5 <= day <= 9:
            rows.append(("b", date, 7.0, None, 10 + day, "HGST", 8000.0))
    frame = pl.DataFrame(
        rows,
        schema=[
            "drive_id",
            "date",
            ATTRS[0],
            ATTRS[1],
            "drive_age_days",
            "manufacturer",
            "capacity_gb",
        ],
        orient="row",
    )
    path = tmp_path / "part.parquet"
    frame.write_parquet(path, row_group_size=7)  # several row groups
    return path


def _build(tmp_path, **kwargs):
    return build_dense_telemetry(
        _write_gold(tmp_path), attributes=ATTRS, start_date=START, end_date=END, **kwargs
    )


def test_dense_telemetry_places_each_row_on_its_calendar_day(tmp_path):
    dense = _build(tmp_path)
    a, b = dense.drive_ids.index("a"), dense.drive_ids.index("b")
    n = len(ATTRS)
    assert dense.values.shape == (2, 20, channel_count(n))
    assert dense.values[a, 3, 0] == pytest.approx(np.log1p(3.0), abs=1e-2)
    assert dense.values[a, 3, 1] == pytest.approx(33.0)  # level attribute, no log
    # absent days stay all-zero, including the "present" flag
    assert dense.values[b, 7].sum() == 0
    assert dense.values[b, 4, 2 * n + 1] == 1.0
    # b never reports temperature: value 0, reported flag 0; reports reallocated
    assert dense.values[b, 4, n + 1] == 0 and dense.values[b, 4, n + 0] == 1


def test_window_ends_on_the_requested_day_and_pads_before_the_data(tmp_path):
    dense = _build(tmp_path)
    fit_normalization(dense, last_day=19)
    a = dense.drive_ids.index("a")
    x, static = gather_windows(dense, np.array([a, a]), np.array([2, 10]), window_days=5)
    assert x.shape == (2, 5, dense.n_channels)
    assert static.shape == (2, static_feature_count())
    n = len(ATTRS)
    # window ending day 2 covers days -2..2: the first two steps are padding
    assert x[0, :2].sum() == 0 and x[0, 2:, 2 * n + 1].tolist() == [1.0, 1.0, 1.0]
    # the counter is increasing across the window ending day 10
    counter = x[1, :, 0]
    assert np.all(np.diff(counter) > 0)


def test_unreported_attribute_is_zero_after_normalization(tmp_path):
    dense = _build(tmp_path)
    fit_normalization(dense, last_day=19)
    b = dense.drive_ids.index("b")
    x, _ = gather_windows(dense, np.array([b]), np.array([12]), window_days=3)
    assert np.all(x[0, :, 1] == 0.0)  # temperature never reported
    assert np.isfinite(x).all()


def test_normalization_uses_only_days_up_to_last_day(tmp_path):
    early = _build(tmp_path)
    fit_normalization(early, last_day=5)
    late = _build(tmp_path)
    fit_normalization(late, last_day=19)
    assert early.mean[0] != late.mean[0]
    # day 5's counter mean over drives a (0..5 -> log1p) and b (7)
    expected = np.mean([np.log1p(d) for d in range(6)] * 1 + [np.log1p(7.0)] * 5)
    assert early.mean[0] == pytest.approx(expected, abs=0.05)


def test_gather_requires_fitted_normalization(tmp_path):
    dense = _build(tmp_path)
    with pytest.raises(ValueError, match="fit_normalization"):
        gather_windows(dense, np.array([0]), np.array([3]), window_days=3)


def test_unknown_attribute_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not in the gold feature table"):
        build_dense_telemetry(
            _write_gold(tmp_path), attributes=["nope"], start_date=START, end_date=END
        )


def test_drive_fraction_keeps_a_deterministic_subset(tmp_path):
    full = _build(tmp_path)
    none = _build(tmp_path, drive_fraction=0.0001)
    assert len(full.drive_ids) == 2
    assert len(none.drive_ids) <= 1
