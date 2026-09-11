import datetime as dt

import polars as pl

from src.preprocess.feature_maturity import build_drive_metadata
from src.preprocess.identifiers import infer_model_family, normalize_drive_id, parse_capacity_gb
from src.preprocess.quality_checks import (
    check_capacity_positive,
    check_failure_date_after_date,
    check_no_duplicate_drive_day_attribute,
    check_no_null_dates,
)
from src.preprocess.smart_mapping import melt_smart_attributes
from src.preprocess.telemetry_gaps import compute_telemetry_gaps


def test_normalize_drive_id_trims_and_uppercases():
    df = pl.DataFrame({"drive_id": [" z1234 ", "z5678"]})
    out = normalize_drive_id(df.lazy()).collect()
    assert out["drive_id"].to_list() == ["Z1234", "Z5678"]


def test_parse_capacity_gb():
    df = pl.DataFrame({"capacity_bytes": [4_000_787_030_016]})
    out = parse_capacity_gb(df.lazy()).collect()
    assert out["capacity_gb"][0] == 4000.78703001600


def test_infer_model_family_matches_known_prefix():
    df = pl.DataFrame({"drive_model": ["ST4000DM000", "WUS721414AL5204", "UNKNOWN_MODEL_X"]})
    out = infer_model_family(df.lazy()).collect()
    assert out["manufacturer"].to_list() == ["Seagate", "Western Digital", "unknown"]
    assert out["model_family"][2] == "UNKNOWN_MODEL_X"


def test_melt_smart_attributes_produces_long_canonical_rows():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "smart_5_raw": [0, 10],
            "smart_197_raw": [0, 3],
        }
    )
    long_df = melt_smart_attributes(df)
    assert long_df.height == 4
    assert set(long_df["smart_attribute_name"].unique()) == {
        "reallocated_sector_count",
        "current_pending_sector_count",
    }
    row_b_reallocated = long_df.filter(
        (pl.col("drive_id") == "B") & (pl.col("smart_attribute_name") == "reallocated_sector_count")
    )
    assert row_b_reallocated["smart_badness_value"][0] == 10.0


def test_compute_telemetry_gaps_flags_stale_and_short_gaps():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 20)],
        }
    )
    out = compute_telemetry_gaps(df, short_gap_days=3, stale_gap_days=7)
    assert out["days_since_last_telemetry"].to_list() == [0, 1, 18]
    assert out["telemetry_gap_flag"].to_list() == [False, False, True]
    assert out["stale_telemetry_flag"].to_list() == [False, False, True]
    assert out["telemetry_coverage_30d"][2] == 3 / 30.0


def test_build_drive_metadata_marks_maturity_states():
    mature_dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(35)]
    as_of = mature_dates[-1]
    warmup_dates = [as_of - dt.timedelta(days=i) for i in range(5)]

    df = pl.DataFrame(
        {
            "drive_id": ["mature"] * 35 + ["warmup"] * 5,
            "date": mature_dates + warmup_dates,
            "model_family": ["Seagate HDD"] * 40,
            "capacity_gb": [4000.0] * 40,
            "stale_telemetry_flag": [False] * 40,
        }
    )
    metadata = build_drive_metadata(df, min_history_days=30, as_of_date=as_of)

    by_id = {row["drive_id"]: row for row in metadata.to_dicts()}
    assert by_id["mature"]["feature_maturity"] == "MATURE"
    assert by_id["mature"]["survivorship_valid"] is True
    assert by_id["warmup"]["feature_maturity"] == "WARMUP"
    assert by_id["warmup"]["survivorship_valid"] is False


def test_quality_checks_detect_violations():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), None],
            "smart_attribute_name": ["x", "x", "x"],
            "capacity_gb": [4000.0, -1.0, 4000.0],
            "failure_date": [dt.date(2023, 12, 1), None, None],
        }
    )
    assert check_no_null_dates(df)["violation_count"] == 1
    assert check_no_duplicate_drive_day_attribute(df)["violation_count"] == 1
    assert check_capacity_positive(df)["violation_count"] == 1
    assert check_failure_date_after_date(df)["violation_count"] == 1
