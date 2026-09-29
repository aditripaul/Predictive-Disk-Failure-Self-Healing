import datetime as dt

import polars as pl

from src.features.confidence import _recency_factor_scalar, add_feature_confidence
from src.features.cross_vendor import add_attribute_ratios, add_model_family_zscores
from src.features.derivatives import add_acceleration, add_deltas
from src.features.events import add_positive_day_counts, add_spike_counts, add_zero_to_nonzero_flags
from src.features.lifecycle import add_lifecycle_features
from src.features.pivot import pivot_badness_wide
from src.features.registry import build_registry
from src.features.windows import add_rolling_aggregates


def _wide_drive_frame() -> pl.DataFrame:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    return pl.DataFrame(
        {
            "drive_id": ["A"] * 10,
            "date": dates,
            "reallocated_sector_count": [0.0, 0.0, 1.0, 1.0, 2.0, 5.0, 5.0, 8.0, 8.0, 20.0],
        }
    )


def test_pivot_badness_wide():
    long_df = pl.DataFrame(
        {
            "drive_id": ["A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "smart_attribute_name": ["reallocated_sector_count", "current_pending_sector_count"],
            "smart_raw_value": [1.0, 2.0],
            "smart_badness_value": [1.0, 2.0],
        }
    )
    wide = pivot_badness_wide(long_df)
    assert wide.height == 1
    assert wide["reallocated_sector_count"][0] == 1.0
    assert wide["current_pending_sector_count"][0] == 2.0


def test_pivot_badness_wide_orders_each_drives_rows_by_date_even_when_input_is_scrambled():
    """The output is sorted by `date` alone, not `[drive_id, date]` (a
    multi-key sort keyed partly on the string drive_id is what crashed
    pipelines/build_silver.py's analogous sort against real fleet data) -
    downstream `.over("drive_id")` window features only need each drive's
    own subset of rows in non-decreasing date order, which a global
    single-key date sort still guarantees regardless of input order."""
    long_df = pl.DataFrame(
        {
            "drive_id": ["B", "A", "B", "A"],
            "date": [
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "smart_attribute_name": ["reallocated_sector_count"] * 4,
            "smart_raw_value": [1.0, 2.0, 3.0, 4.0],
            "smart_badness_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide = pivot_badness_wide(long_df)
    for drive_id in ("A", "B"):
        dates = wide.filter(pl.col("drive_id") == drive_id)["date"].to_list()
        assert dates == sorted(dates)


def test_add_rolling_aggregates():
    df = _wide_drive_frame()
    out = add_rolling_aggregates(df, ["reallocated_sector_count"], windows_days=(7,))
    last_row = out.tail(1)
    # 7-day window ending 2024-01-10 covers 2024-01-04..01-10: [1,2,5,5,8,8,20]
    assert last_row["reallocated_sector_count_7d_max"][0] == 20.0
    assert last_row["reallocated_sector_count_7d_min"][0] == 1.0
    expected_range = (
        last_row["reallocated_sector_count_7d_max"][0]
        - last_row["reallocated_sector_count_7d_min"][0]
    )
    assert last_row["reallocated_sector_count_7d_range"][0] == expected_range


def test_add_deltas_and_acceleration():
    df = _wide_drive_frame()
    out = add_deltas(df, ["reallocated_sector_count"], windows_days=(7,))
    last_row = out.tail(1)
    # value[t]=20, value[t-7]=1 -> delta=19, slope=19/7
    assert last_row["reallocated_sector_count_7d_delta"][0] == 19.0
    assert abs(last_row["reallocated_sector_count_7d_slope"][0] - 19.0 / 7) < 1e-9

    out = add_deltas(out, ["reallocated_sector_count"], windows_days=(7, 9))
    out = add_acceleration(
        out, ["reallocated_sector_count"], short_window_days=7, long_window_days=9
    )
    assert "reallocated_sector_count_acceleration_7d_vs_9d" in out.columns


def test_add_positive_day_counts_and_zero_to_nonzero():
    df = _wide_drive_frame()
    out = add_positive_day_counts(df, ["reallocated_sector_count"], windows_days=(7,))
    assert out.tail(1)["reallocated_sector_count_7d_positive_count"][0] > 0

    out = add_zero_to_nonzero_flags(df, ["reallocated_sector_count"])
    # row index 2 (value goes from 0.0 -> 1.0) should flip True
    assert out["reallocated_sector_count_zero_to_nonzero"][2] is True
    assert out["reallocated_sector_count_zero_to_nonzero"][0] is False


def test_add_spike_counts():
    df = _wide_drive_frame()
    out = add_spike_counts(df, {"reallocated_sector_count": 5.0}, windows_days=(30,))
    # day 9: 20 - 8 = 12 >= 5 -> counts as a spike
    assert out.tail(1)["reallocated_sector_count_30d_spike_count"][0] >= 1


def test_add_lifecycle_features():
    df = _wide_drive_frame()
    out = add_lifecycle_features(df)
    assert out["drive_age_days"].to_list() == list(range(10))
    assert out["drive_age_days_squared"].to_list() == [i**2 for i in range(10)]


def test_add_feature_confidence_matches_formula():
    df = pl.DataFrame(
        {
            "drive_id": ["A"],
            "date": [dt.date(2024, 1, 1)],
            "telemetry_coverage_30d": [0.5],
            "days_since_last_telemetry": [2],
            "reallocated_sector_count": [1.0],
        }
    )
    out = add_feature_confidence(df, ["reallocated_sector_count"], recency_tau_hours=48.0)
    expected_recency = _recency_factor_scalar(48.0, tau=48.0)
    assert abs(out["recency_factor"][0] - expected_recency) < 1e-9
    assert out["attribute_coverage_factor"][0] == 1.0
    assert abs(out["feature_confidence"][0] - 0.5 * expected_recency * 1.0) < 1e-9


def test_build_registry_covers_all_families():
    entries = build_registry(
        attributes=["reallocated_sector_count"],
        windows_days=(7, 30),
        spike_thresholds={"reallocated_sector_count": 5.0},
    )
    names = {e.feature_name for e in entries}
    assert "reallocated_sector_count_7d_mean" in names
    assert "reallocated_sector_count_7d_slope" in names
    assert "reallocated_sector_count_acceleration_7d_vs_30d" in names
    assert "reallocated_sector_count_7d_spike_count" in names
    assert "drive_age_days" in names
    assert "feature_confidence" in names
    assert "reallocated_per_capacity" in names
    assert "reallocated_sector_count_7d_model_zscore" in names


def test_add_attribute_ratios_computes_known_ratios():
    df = pl.DataFrame(
        {
            "drive_id": ["A"],
            "current_pending_sector_count": [4.0],
            "reallocated_sector_count": [1.0],
            "offline_uncorrectable": [9.0],
            "power_on_hours": [899.0],
            "capacity_gb": [4000.0],
        }
    )
    out = add_attribute_ratios(df)
    assert out["pending_to_reallocated_ratio"][0] == 4.0 / (1.0 + 1)
    assert out["uncorrectable_per_power_on_hour"][0] == 9.0 / (899.0 + 1)
    assert out["reallocated_per_capacity"][0] == 1.0 / 4000.0


def test_add_attribute_ratios_is_a_noop_when_columns_missing():
    df = pl.DataFrame({"drive_id": ["A"], "unrelated_column": [1.0]})
    out = add_attribute_ratios(df)
    assert out.columns == df.columns


def test_add_model_family_zscores_normalizes_relative_to_model_family():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "B", "C", "D"],
            "model_family": ["Seagate HDD", "Seagate HDD", "WD HDD", "WD HDD"],
            "reallocated_sector_count_7d_mean": [0.0, 10.0, 100.0, 300.0],
        }
    )
    out = add_model_family_zscores(df, ["reallocated_sector_count"], windows_days=(7,))
    col = "reallocated_sector_count_7d_model_zscore"
    assert col in out.columns
    by_id = dict(zip(out["drive_id"], out[col], strict=True))
    # each family's two members are equidistant from their own family mean,
    # so they get opposite-signed z-scores despite very different raw scales.
    assert by_id["A"] < 0 < by_id["B"]
    assert by_id["C"] < 0 < by_id["D"]


def test_add_model_family_zscores_is_a_noop_without_model_family_column():
    df = pl.DataFrame({"drive_id": ["A"], "reallocated_sector_count_7d_mean": [1.0]})
    out = add_model_family_zscores(df, ["reallocated_sector_count"], windows_days=(7,))
    assert out.columns == df.columns
