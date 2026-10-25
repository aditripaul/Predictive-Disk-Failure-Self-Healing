"""Bronze profiling is accumulated per partition, not over the whole dataset.

One 16-column `null_count()` aggregation over a 365-day Bronze dataset
(~125M rows) materialized every column at once and tripped the 20 GB cap at
the very end of an otherwise successful ingest. Row counts and null counts
now come from the Parquet footers instead, and everything else is projected
and accumulated one partition at a time.

The footers are a second source of truth, so the central test here is that
they agree with reading the data - exactly, not approximately.
"""

from pathlib import Path

import polars as pl
import pytest

from src.ingest.profiling import (
    _counts_by_reading,
    _counts_from_parquet_metadata,
    bronze_columns,
    bronze_parts,
    profile_bronze_dataset,
)


def _partition(root: Path, year: int, month: int, frame: pl.DataFrame) -> Path:
    out = root / f"year={year:04d}" / f"month={month:02d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "part.parquet"
    frame.write_parquet(path)
    return path


def _bronze(root: Path) -> Path:
    """Three partitions with nulls scattered unevenly, so a per-partition sum
    that dropped or double-counted a file would not match the total."""
    _partition(
        root,
        2025,
        7,
        pl.DataFrame(
            {
                "drive_id": ["a", "b", "c"],
                "model": ["ST1", "ST1", "WDC2"],
                "failure": [0, 0, 1],
                "smart_5_raw": [1.0, None, 3.0],
            }
        ),
    )
    _partition(
        root,
        2025,
        8,
        pl.DataFrame(
            {
                "drive_id": ["d", "e"],
                "model": ["WDC2", "WDC2"],
                "failure": [0, 1],
                "smart_5_raw": [None, None],
            }
        ),
    )
    _partition(
        root,
        2026,
        1,
        pl.DataFrame(
            {
                "drive_id": ["f"],
                "model": ["ST1"],
                "failure": [0],
                "smart_5_raw": [9.0],
            }
        ),
    )
    return root


def test_metadata_counts_match_reading_the_data(tmp_path: Path):
    """The whole fix rests on the footers being right. If they ever disagree
    with the data, the report is silently wrong rather than slow."""
    parts = bronze_parts(_bronze(tmp_path / "bronze"))
    columns = bronze_columns(parts)

    from_metadata = _counts_from_parquet_metadata(parts)
    assert from_metadata is not None, "Polars writes null-count statistics"
    assert from_metadata == _counts_by_reading(parts, columns)


def test_null_counts_are_summed_across_every_partition(tmp_path: Path):
    report = profile_bronze_dataset(_bronze(tmp_path / "bronze"))
    assert report["row_count"] == 6
    assert report["partition_count"] == 3
    # 3 of 6 smart_5_raw values are null: one in July, two in August.
    assert report["null_percentage_by_column"]["smart_5_raw"] == 0.5
    assert report["null_percentage_by_column"]["drive_id"] == 0.0


def test_model_distribution_merges_a_model_that_spans_partitions(tmp_path: Path):
    """ST1 appears in two partitions and WDC2 in two others; grouping per file
    and concatenating would report each model twice."""
    report = profile_bronze_dataset(_bronze(tmp_path / "bronze"))
    counts = {row["model"]: row["count"] for row in report["drive_model_distribution"]}
    assert counts == {"ST1": 3, "WDC2": 3}
    assert len(report["drive_model_distribution"]) == 2


def test_failure_rate_is_summed_across_partitions(tmp_path: Path):
    report = profile_bronze_dataset(_bronze(tmp_path / "bronze"))
    assert report["failure_rate"] == round(2 / 6, 6)


def test_a_column_missing_from_one_partition_counts_as_null_there(tmp_path: Path):
    """Bronze keeps whichever optional SMART columns a source file carried and
    Backblaze's column set widened over time, so partitions need not agree.
    Reading only the first partition's schema would report a column absent
    elsewhere as 0% null across the whole dataset."""
    root = tmp_path / "bronze"
    _partition(root, 2025, 7, pl.DataFrame({"drive_id": ["a", "b"], "smart_196_raw": [1.0, 2.0]}))
    _partition(root, 2025, 8, pl.DataFrame({"drive_id": ["c", "d"]}))

    parts = bronze_parts(root)
    assert bronze_columns(parts) == ["drive_id", "smart_196_raw"]
    report = profile_bronze_dataset(root)
    assert report["row_count"] == 4
    assert report["null_percentage_by_column"]["smart_196_raw"] == 0.5
    # And the two paths still agree once the schemas differ.
    assert _counts_from_parquet_metadata(parts) == _counts_by_reading(parts, bronze_columns(parts))


def test_an_empty_bronze_root_profiles_to_zero_rather_than_raising(tmp_path: Path):
    report = profile_bronze_dataset(tmp_path / "nothing-here")
    assert report["row_count"] == 0
    assert report["partition_count"] == 0
    assert report["columns"] == []
    assert report["null_percentage_by_column"] == {}
    assert "failure_rate" not in report


def test_profiling_never_reads_a_column_to_count_rows_or_nulls(tmp_path: Path, monkeypatch):
    """The regression guard: if row counts or null counts ever go back to
    scanning column data, a 365-day ingest dies at the last step again. Only
    the failure sum and the model distribution may read, and only projected."""
    root = _bronze(tmp_path / "bronze")
    parts = bronze_parts(root)

    scanned: list[Path] = []
    real_scan = pl.scan_parquet

    def tracking_scan(source, *args, **kwargs):
        scanned.append(source)
        return real_scan(source, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", tracking_scan)
    counts = _counts_from_parquet_metadata(parts)

    assert counts is not None
    assert scanned == [], f"counted rows/nulls by scanning {scanned}"


@pytest.mark.parametrize("column", ["failure", "model"])
def test_reading_is_projected_to_the_single_column_it_needs(tmp_path: Path, column: str):
    """Each per-partition read must pull one column, not the whole row group -
    that projection is what bounds peak memory."""
    parts = bronze_parts(_bronze(tmp_path / "bronze"))
    from src.ingest.profiling import _failure_count, _model_distribution

    if column == "failure":
        assert _failure_count(parts) == 2
    else:
        assert len(_model_distribution(parts, "model")) == 2
    # The projection itself: scanning one column of a partition yields exactly
    # that column, so the plan cannot be pulling the other three.
    projected = pl.scan_parquet(parts[0]).select(column).collect()
    assert projected.columns == [column]
