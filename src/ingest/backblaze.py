"""Ingestion of Backblaze Hard Drive Stats CSV files into the Bronze layer.

Backblaze publishes one CSV per day (bundled quarterly), with columns:
`date, serial_number, model, capacity_bytes, failure, smart_<id>_raw, smart_<id>_normalized, ...`

This module only lands the raw columns we need with minimal transformation
(select, null-date filter, provenance stamping). Canonical schema mapping,
SMART harmonization, and badness orientation happen in Phase 2 (silver layer).
"""

from __future__ import annotations

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
