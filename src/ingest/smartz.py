"""Ingestion of the SMART-Z cross-vendor validation dataset into the Bronze layer.

SMART-Z ships 65 standardized SMART attributes per disk. Unlike Backblaze we
do not hardcode a fixed attribute allowlist here: the source-to-canonical
attribute mapping is a Phase 2 (silver harmonization) concern, implemented in
`src/preprocess/smart_mapping.py::SOURCE_COLUMN_TEMPLATES["smartz"]`. This
stage only requires a `drive_id` and `date` column to exist and passes the
remaining columns through unchanged, stamped with provenance metadata.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata

REQUIRED_COLUMNS = ["date", "drive_id"]


def ingest_smartz_file(csv_path: Path, bronze_root: Path) -> list[Path]:
    """Lazily ingest a single SMART-Z CSV file into Bronze Parquet, partitioned
    by year/month. Never materializes the full file."""
    lf = pl.scan_csv(csv_path, infer_schema_length=10_000, try_parse_dates=True)

    schema_names = set(lf.collect_schema().names())
    missing = set(REQUIRED_COLUMNS) - schema_names
    if missing:
        raise ValueError(f"{csv_path}: missing required SMART-Z columns {sorted(missing)}")

    lf = lf.filter(pl.col("date").is_not_null())
    lf = with_ingestion_metadata(lf, source_dataset=SourceDataset.SMARTZ, source_file=str(csv_path))

    return sink_partitioned_by_month(lf, bronze_root=bronze_root)
