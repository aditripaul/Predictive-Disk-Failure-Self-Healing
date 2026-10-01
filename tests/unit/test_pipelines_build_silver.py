"""build_silver's stage functions sink drive_day and the canonical long
telemetry table straight to disk instead of returning either in memory
(docs/developer_guide.md 5.10), processing drive batches independently
(see the module docstring for why batching is by drive_id, not by
month/Bronze-file) - these tests exercise that behavior end-to-end
against a small Bronze fixture, calling the stage functions directly
rather than through main()'s subprocess orchestration (which reads its
config paths from the real configs/data.yaml, not a tmp_path fixture)."""

import json
from pathlib import Path

import polars as pl
import pytest

from pipelines.build_silver import _stage_batch, _stage_finalize, _stage_prepare
from src.ingest.synthetic_stub import ingest_synthetic_stub


def _run_build_silver(
    bronze_root: Path, tmp_path: Path, *, drive_day_path: Path, canonical_path: Path
):
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    _stage_prepare(bronze_root, tmp_dir)
    manifest = json.loads((tmp_dir / "prepare_manifest.json").read_text())
    n_batches = manifest["n_batches"]

    batch_paths = []
    for batch_index in range(n_batches):
        batch_path = tmp_dir / f"drive_day_batch_{batch_index}.parquet"
        _stage_batch(bronze_root, batch_index, n_batches, batch_path)
        batch_paths.append(batch_path)

    metadata_path = tmp_path / "silver" / "drive_metadata" / "part.parquet"
    quality_report_path = tmp_path / "audit" / "silver_quality_report.json"
    result_path = tmp_dir / "finalize_result.json"
    _stage_finalize(
        batch_paths,
        drive_day_path,
        canonical_path,
        metadata_path,
        quality_report_path,
        result_path,
    )
    result = json.loads(result_path.read_text())
    quality_reports = json.loads(quality_report_path.read_text())
    drive_metadata = pl.read_parquet(metadata_path)
    return result, quality_reports, drive_metadata


def test_build_silver_sinks_drive_day_and_canonical_long_and_returns_row_counts(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)
    drive_day_path = tmp_path / "silver" / "drive_day" / "part.parquet"
    canonical_path = tmp_path / "silver" / "canonical_telemetry" / "part.parquet"

    result, quality_reports, drive_metadata = _run_build_silver(
        bronze_root, tmp_path, drive_day_path=drive_day_path, canonical_path=canonical_path
    )

    assert drive_day_path.exists()
    assert canonical_path.exists()
    drive_day = pl.read_parquet(drive_day_path)
    canonical_long = pl.read_parquet(canonical_path)
    assert result["drive_day_row_count"] == drive_day.height
    assert result["canonical_row_count"] == canonical_long.height

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


def test_build_silver_batching_does_not_split_a_drive_across_batches(tmp_path: Path):
    """The whole point of hashing the *normalized* drive_id for batch
    assignment: every row of a given drive must land in exactly one
    batch, or compute_telemetry_gaps would see only part of that drive's
    history in each batch."""
    bronze_root = tmp_path / "bronze"
    ingest_synthetic_stub(bronze_root)

    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    _stage_prepare(bronze_root, tmp_dir)
    manifest = json.loads((tmp_dir / "prepare_manifest.json").read_text())
    n_batches = max(manifest["n_batches"], 4)  # force multiple batches even for a tiny fixture

    batches = []
    for batch_index in range(n_batches):
        batch_path = tmp_dir / f"drive_day_batch_{batch_index}.parquet"
        _stage_batch(bronze_root, batch_index, n_batches, batch_path)
        batches.append(pl.read_parquet(batch_path))

    drive_ids_per_batch = [set(b["drive_id"].unique().to_list()) for b in batches]
    for i in range(len(drive_ids_per_batch)):
        for j in range(i + 1, len(drive_ids_per_batch)):
            assert drive_ids_per_batch[i].isdisjoint(drive_ids_per_batch[j])

    total_rows = sum(b.height for b in batches)
    all_drive_ids = set().union(*drive_ids_per_batch) if drive_ids_per_batch else set()
    direct = pl.concat(
        [pl.scan_parquet(p) for p in sorted(bronze_root.glob("**/*.parquet"))],
        how="diagonal_relaxed",
    ).collect()
    assert total_rows == direct.height
    direct_drive_ids = direct["drive_id"].str.strip_chars().str.to_uppercase().unique().to_list()
    assert all_drive_ids == set(direct_drive_ids)


def test_build_silver_raises_a_clear_error_with_no_bronze_files(tmp_path: Path):
    bronze_root = tmp_path / "empty_bronze"
    bronze_root.mkdir()
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No Bronze Parquet files found"):
        _stage_prepare(bronze_root, tmp_dir)
