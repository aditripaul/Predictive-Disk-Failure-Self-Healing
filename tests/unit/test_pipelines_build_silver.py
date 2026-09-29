"""build_silver() sinks drive_day and the canonical long telemetry table
straight to disk instead of returning either in memory
(docs/developer_guide.md 5.10) - these tests exercise that behavior
end-to-end against a small Bronze fixture, not just the individual
preprocess functions it composes."""

from pathlib import Path

import polars as pl
import pytest

from pipelines.build_silver import build_silver
from src.ingest.synthetic_stub import ingest_synthetic_stub

CONFIG = {
    "telemetry_gap": {"short_gap_days": 3, "stale_gap_days": 7},
    "feature_maturity": {"min_history_days": 30},
}


def test_build_silver_sinks_drive_day_and_canonical_long_and_returns_row_counts(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)
    drive_day_path = tmp_path / "silver" / "drive_day" / "part.parquet"
    canonical_path = tmp_path / "silver" / "canonical_telemetry" / "part.parquet"

    drive_day_row_count, canonical_row_count, drive_metadata, quality_reports = build_silver(
        bronze_root, CONFIG, drive_day_path=drive_day_path, canonical_path=canonical_path
    )

    assert drive_day_path.exists()
    assert canonical_path.exists()
    drive_day = pl.read_parquet(drive_day_path)
    canonical_long = pl.read_parquet(canonical_path)
    assert drive_day_row_count == drive_day.height
    assert canonical_row_count == canonical_long.height

    # canonical_long only carries the melt id columns (drive_id, date) plus
    # the attribute columns - everything else (drive_model, capacity_gb,
    # telemetry-gap flags, ...) lives only in drive_day now, to avoid
    # duplicating it once per SMART attribute.
    assert set(canonical_long.columns) == {
        "drive_id",
        "date",
        "smart_attribute_name",
        "smart_raw_value",
        "smart_badness_value",
    }
    n_attributes = canonical_long["smart_attribute_name"].n_unique()
    assert canonical_long.height % n_attributes == 0
    assert canonical_long.height // n_attributes == drive_day.height
    assert canonical_long.height // n_attributes >= drive_metadata.height
    assert {"drive_model", "capacity_gb", "telemetry_gap_flag"}.issubset(drive_day.columns)
    assert all(report["passed"] for report in quality_reports)


def test_build_silver_creates_missing_parent_directories(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)
    drive_day_path = tmp_path / "deeply" / "nested" / "silver" / "drive_day.parquet"
    canonical_path = tmp_path / "deeply" / "nested" / "silver" / "canonical.parquet"

    build_silver(bronze_root, CONFIG, drive_day_path=drive_day_path, canonical_path=canonical_path)

    assert drive_day_path.exists()
    assert canonical_path.exists()


def test_build_silver_raises_a_clear_error_with_no_bronze_files(tmp_path: Path):
    bronze_root = tmp_path / "empty_bronze"
    bronze_root.mkdir()
    drive_day_path = tmp_path / "silver" / "drive_day.parquet"
    canonical_path = tmp_path / "silver" / "part.parquet"

    with pytest.raises(FileNotFoundError, match="No Bronze Parquet files found"):
        build_silver(
            bronze_root, CONFIG, drive_day_path=drive_day_path, canonical_path=canonical_path
        )
