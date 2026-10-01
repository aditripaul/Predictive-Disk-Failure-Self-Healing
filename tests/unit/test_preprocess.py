import datetime as dt

import polars as pl

from src.preprocess.failure_events import derive_failure_date
from src.preprocess.feature_maturity import build_drive_metadata
from src.preprocess.identifiers import infer_model_family, normalize_drive_id, parse_capacity_gb
from src.preprocess.quality_checks import (
    check_capacity_positive,
    check_failure_date_after_date,
    check_no_duplicate_drive_day_attribute,
    check_no_null_dates,
    run_all_checks,
)
from src.preprocess.smart_mapping import build_canonical_attribute_map, melt_smart_attributes
from src.preprocess.telemetry_gaps import compute_telemetry_gaps


def test_derive_failure_date_broadcasts_the_failure_day_to_every_row():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "A", "A", "B", "B"],
            "date": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "failure": [0, 0, 1, 0, 0],
        }
    )
    out = derive_failure_date(df)
    by_drive = {row["drive_id"]: row["failure_date"] for row in out.to_dicts()}
    a_rows = out.filter(pl.col("drive_id") == "A")
    assert (a_rows["failure_date"] == dt.date(2024, 1, 3)).all()
    assert out.filter(pl.col("drive_id") == "B")["failure_date"].is_null().all()
    assert by_drive["A"] == dt.date(2024, 1, 3)


def test_derive_failure_date_is_a_noop_without_a_failure_column():
    df = pl.DataFrame({"drive_id": ["A"], "date": [dt.date(2024, 1, 1)]})
    out = derive_failure_date(df)
    assert "failure_date" not in out.columns


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


def test_melt_smart_attributes_with_explicit_id_columns_drops_everything_else():
    """pipelines/build_silver.py passes id_columns=["drive_id", "date"] to
    avoid duplicating every other drive-day column (drive_model,
    capacity_gb, telemetry-gap flags, ...) once per SMART attribute - a
    real, avoidable ~10x memory/disk cost at fleet scale that a caller
    needing those columns should instead join back afterward, at the
    drive-day grain, from wherever they're actually persisted."""
    df = pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "drive_model": ["ST4000DM000", "ST4000DM000"],
            "capacity_gb": [4000.0, 4000.0],
            "smart_5_raw": [0, 10],
            "smart_197_raw": [0, 3],
        }
    )
    long_df = melt_smart_attributes(df, id_columns=["drive_id", "date"])
    assert set(long_df.columns) == {
        "drive_id",
        "date",
        "smart_attribute_name",
        "smart_raw_value",
        "smart_badness_value",
    }
    assert long_df.height == 4


def test_build_canonical_attribute_map_differs_per_source():
    backblaze_map = build_canonical_attribute_map("backblaze")
    smartz_map = build_canonical_attribute_map("smartz")
    assert backblaze_map["smart_5_raw"] == "reallocated_sector_count"
    assert smartz_map["smart_5_normalized"] == "reallocated_sector_count"
    # same canonical attributes, different bronze column names
    assert set(backblaze_map.values()) == set(smartz_map.values())


def test_melt_smart_attributes_harmonizes_mixed_backblaze_and_smartz_sources():
    """A single silver-harmonization pass over a Bronze frame containing
    both sources must map each source's own column-naming convention to the
    same canonical attribute names (docs/dataset_strategy.md section 2
    cross-vendor generalization objective)."""
    df = pl.DataFrame(
        {
            "drive_id": ["A", "Z1"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "source_dataset": ["backblaze", "smartz"],
            "smart_5_raw": [10, None],
            "smart_5_normalized": [None, 99],
        }
    )
    long_df = melt_smart_attributes(df)

    assert set(long_df["smart_attribute_name"].unique()) == {"reallocated_sector_count"}
    by_drive = {row["drive_id"]: row["smart_badness_value"] for row in long_df.to_dicts()}
    assert by_drive["A"] == 10.0
    assert by_drive["Z1"] == 99.0


def test_melt_smart_attributes_with_a_single_source_present_still_harmonizes_correctly():
    """Real Bronze data ingested so far is Backblaze-only, but every row
    still carries a `source_dataset` column - this must take the same
    code path as the harmonized-mapping case above, not silently fall
    back to the no-source_column path (which would use the wrong map for
    a non-Backblaze single source)."""
    df = pl.DataFrame(
        {
            "drive_id": ["Z1", "Z2"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "source_dataset": ["smartz", "smartz"],
            "smart_5_normalized": [7, 99],
        }
    )
    long_df = melt_smart_attributes(df)
    assert set(long_df["smart_attribute_name"].unique()) == {"reallocated_sector_count"}
    by_drive = {row["drive_id"]: row["smart_badness_value"] for row in long_df.to_dicts()}
    assert by_drive["Z1"] == 7.0
    assert by_drive["Z2"] == 99.0


def test_melt_smart_attributes_skips_source_whose_template_columns_are_entirely_absent():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "Z1"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "source_dataset": ["backblaze", "smartz"],
            # only Backblaze's "_raw" template column exists at all; SMART-Z's
            # "_normalized" template has no matching column anywhere in this
            # frame, so its rows must be skipped rather than melted as nulls.
            "smart_5_raw": [10, None],
        }
    )
    long_df = melt_smart_attributes(df)
    assert set(long_df["drive_id"].unique()) == {"A"}


def test_melt_smart_attributes_handles_a_schema_valid_but_empty_frame():
    """pipelines/build_silver.py batches Bronze rows by hash(drive_id), so
    a batch can legitimately end up with zero rows while still carrying a
    `source_dataset` column in its schema. Before this was handled, that
    zero-row case fell through melt_smart_attributes' multi-source branch
    (df[source_column].unique() is empty, so the `parts` accumulator loop
    never runs) and raised, even though a correctly-shaped empty result -
    not an error - is what a batching caller needs to concat with its
    other, non-empty batches."""
    df = pl.DataFrame(
        {
            "drive_id": [],
            "date": [],
            "source_dataset": [],
            "smart_5_raw": [],
        },
        schema={
            "drive_id": pl.Utf8,
            "date": pl.Date,
            "source_dataset": pl.Utf8,
            "smart_5_raw": pl.Int64,
        },
    )
    long_df = melt_smart_attributes(df, id_columns=["drive_id", "date"])
    assert long_df.height == 0
    assert set(long_df.columns) == {
        "drive_id",
        "date",
        "smart_attribute_name",
        "smart_raw_value",
        "smart_badness_value",
    }

    # The actual code path build_silver.py exercises: a LazyFrame filtered
    # down to zero rows, not an eagerly-empty DataFrame from the start.
    lazy_long_df = melt_smart_attributes(
        df.lazy().filter(pl.col("drive_id") == "nonexistent"), id_columns=["drive_id", "date"]
    ).collect()
    assert lazy_long_df.height == 0
    assert set(lazy_long_df.columns) == set(long_df.columns)


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


def test_quality_checks_accept_a_lazyframe_not_just_a_dataframe():
    """build_silver sinks the canonical long table straight to disk and
    re-scans it lazily to run these checks against fleet-scale data
    without ever materializing it in memory - every check must therefore
    work identically whether given a DataFrame or a LazyFrame."""
    df = pl.DataFrame(
        {
            "drive_id": ["A", "A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1), None],
            "smart_attribute_name": ["x", "x", "x"],
            "capacity_gb": [4000.0, -1.0, 4000.0],
            "failure_date": [dt.date(2023, 12, 1), None, None],
        }
    )
    lf = df.lazy()
    assert check_no_null_dates(lf) == check_no_null_dates(df)
    assert check_no_duplicate_drive_day_attribute(lf) == check_no_duplicate_drive_day_attribute(df)
    assert check_capacity_positive(lf) == check_capacity_positive(df)
    assert check_failure_date_after_date(lf) == check_failure_date_after_date(df)
    assert run_all_checks(lf) == run_all_checks(df)
