import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata


def _stamped_lf(rows: list[dict], *, source_file: str) -> pl.LazyFrame:
    lf = pl.DataFrame(rows).lazy()
    return with_ingestion_metadata(
        lf, source_dataset=SourceDataset.BACKBLAZE, source_file=source_file
    )


def test_sink_partitioned_by_month_accumulates_across_multiple_calls_into_the_same_partition(
    tmp_path: Path,
):
    """Reproduces the real ingestion pattern: pipelines/ingest_backblaze.py
    calls this once per daily CSV file, so a month's partition is written
    to by many separate calls, not one. Before this was fixed, each call
    overwrote the previous one's rows entirely - only the last day
    ingested into a given month survived."""
    bronze_root = tmp_path / "bronze"

    day1 = _stamped_lf(
        [{"date": dt.date(2026, 1, 1), "drive_id": "A", "failure": 0}],
        source_file="2026-01-01.csv",
    )
    day2 = _stamped_lf(
        [{"date": dt.date(2026, 1, 2), "drive_id": "A", "failure": 0}],
        source_file="2026-01-02.csv",
    )
    day3 = _stamped_lf(
        [{"date": dt.date(2026, 1, 3), "drive_id": "A", "failure": 0}],
        source_file="2026-01-03.csv",
    )

    for day in (day1, day2, day3):
        sink_partitioned_by_month(day, bronze_root=bronze_root)

    result = pl.read_parquet(bronze_root / "year=2026" / "month=01" / "part.parquet")
    assert sorted(result["date"].to_list()) == [
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 3),
    ]


def test_sink_partitioned_by_month_reingesting_the_same_file_is_idempotent(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    lf = _stamped_lf(
        [{"date": dt.date(2026, 1, 1), "drive_id": "A", "failure": 0}],
        source_file="2026-01-01.csv",
    )

    sink_partitioned_by_month(lf, bronze_root=bronze_root)
    sink_partitioned_by_month(lf, bronze_root=bronze_root)

    result = pl.read_parquet(bronze_root / "year=2026" / "month=01" / "part.parquet")
    assert result.height == 1


def test_sink_partitioned_by_month_keeps_different_months_in_separate_files(tmp_path: Path):
    bronze_root = tmp_path / "bronze"
    january = _stamped_lf(
        [{"date": dt.date(2026, 1, 15), "drive_id": "A", "failure": 0}],
        source_file="2026-01-15.csv",
    )
    february = _stamped_lf(
        [{"date": dt.date(2026, 2, 15), "drive_id": "A", "failure": 0}],
        source_file="2026-02-15.csv",
    )

    sink_partitioned_by_month(january, bronze_root=bronze_root)
    sink_partitioned_by_month(february, bronze_root=bronze_root)

    jan_result = pl.read_parquet(bronze_root / "year=2026" / "month=01" / "part.parquet")
    feb_result = pl.read_parquet(bronze_root / "year=2026" / "month=02" / "part.parquet")
    assert jan_result["date"].to_list() == [dt.date(2026, 1, 15)]
    assert feb_result["date"].to_list() == [dt.date(2026, 2, 15)]
