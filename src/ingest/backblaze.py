"""Ingestion of Backblaze Hard Drive Stats CSV files into the Bronze layer.

Backblaze publishes one CSV per day (bundled quarterly), with columns:
`date, serial_number, model, capacity_bytes, failure, smart_<id>_raw, smart_<id>_normalized, ...`

This module only lands the raw columns we need with minimal transformation
(select, null-date filter, provenance stamping). Canonical schema mapping,
SMART harmonization, and badness orientation happen in Phase 2 (silver layer).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata

REQUIRED_COLUMNS = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]


def filter_files_by_date_range(
    files: list[Path],
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> list[Path]:
    """Restricts `files` to those whose filename stem is a `YYYY-MM-DD`
    date within `[start_date, end_date]` (inclusive; either bound may be
    omitted) - Backblaze publishes one CSV per day named exactly that way
    (docs/dataset_strategy.md section 3.1), so this lets a much smaller
    date range be ingested (and thus fed to `make build-silver`) without
    touching what's already been downloaded into `data/raw/backblaze/`.
    A file whose stem isn't a plain date (unexpected, but not impossible)
    is kept rather than silently dropped, since it can't be judged against
    the range at all."""
    if start_date is None and end_date is None:
        return files

    kept = []
    for path in files:
        try:
            file_date = dt.date.fromisoformat(path.stem)
        except ValueError:
            kept.append(path)
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        kept.append(path)
    return kept


def ingest_backblaze_file(csv_path: Path, bronze_root: Path) -> list[Path]:
    """Lazily ingest a single Backblaze daily/quarterly CSV file into Bronze
    Parquet, partitioned by year/month. Never materializes the full file."""
    lf = pl.scan_csv(csv_path, infer_schema_length=10_000, try_parse_dates=True)

    available = [c for c in REQUIRED_COLUMNS if c in lf.collect_schema().names()]
    missing = set(REQUIRED_COLUMNS) - set(available)
    if missing:
        raise ValueError(f"{csv_path}: missing required Backblaze columns {sorted(missing)}")

    lf = (
        lf.select(available)
        .filter(pl.col("date").is_not_null())
        .rename({"serial_number": "drive_id", "model": "drive_model"})
    )
    lf = with_ingestion_metadata(
        lf, source_dataset=SourceDataset.BACKBLAZE, source_file=str(csv_path)
    )

    return sink_partitioned_by_month(lf, bronze_root=bronze_root)
