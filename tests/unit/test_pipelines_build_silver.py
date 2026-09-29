"""build_silver() sinks the canonical long telemetry table straight to
disk instead of returning it in memory (docs/developer_guide.md 5.10) -
these tests exercise that behavior end-to-end against a small Bronze
fixture, not just the individual preprocess functions it composes."""

from pathlib import Path

import polars as pl
import pytest

from pipelines.build_silver import build_silver
from src.ingest.synthetic_stub import ingest_synthetic_stub

CONFIG = {
    "telemetry_gap": {"short_gap_days": 3, "stale_gap_days": 7},
    "feature_maturity": {"min_history_days": 30},
}


def test_build_silver_sinks_canonical_long_and_returns_its_row_count(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)
    canonical_path = tmp_path / "silver" / "canonical_telemetry" / "part.parquet"

    row_count, drive_metadata, quality_reports = build_silver(
        bronze_root, CONFIG, canonical_path=canonical_path
    )

    assert canonical_path.exists()
    canonical_long = pl.read_parquet(canonical_path)
    assert row_count == canonical_long.height
    # One row per (drive_id, date, smart_attribute_name): the total row
    # count must be an exact multiple of the number of distinct attributes
    # melted, and there must be at least as many drives worth of rows as
    # build_drive_metadata found.
    n_attributes = canonical_long["smart_attribute_name"].n_unique()
    assert canonical_long.height % n_attributes == 0
    assert canonical_long.height // n_attributes >= drive_metadata.height
    assert {"drive_id", "date", "smart_attribute_name", "smart_badness_value"}.issubset(
        canonical_long.columns
    )
    assert all(report["passed"] for report in quality_reports)


def test_build_silver_creates_missing_parent_directories(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)
    canonical_path = tmp_path / "deeply" / "nested" / "silver" / "part.parquet"

    build_silver(bronze_root, CONFIG, canonical_path=canonical_path)

    assert canonical_path.exists()


def test_build_silver_raises_a_clear_error_with_no_bronze_files(tmp_path: Path):
    bronze_root = tmp_path / "empty_bronze"
    bronze_root.mkdir()
    canonical_path = tmp_path / "silver" / "part.parquet"

    with pytest.raises(FileNotFoundError, match="No Bronze Parquet files found"):
        build_silver(bronze_root, CONFIG, canonical_path=canonical_path)
