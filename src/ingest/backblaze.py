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


#: Extra SMART attributes ingested when present: 7 seek error rate, 9 power-on
#: hours, 194 temperature, 199 UDMA CRC errors, 196 reallocation event count.
#:
#: 196 was added from the attribute screen (docs/adr/0002): it out-separates
#: four of the five priority attributes, and at ~66% coverage against
#: smart_187/188's ~32% it reaches the non-Seagate drives where those two are
#: null - the part of the fleet with no defect-EVENT counter at all until now.
OPTIONAL_SMART_COLUMNS = [
    "smart_7_raw",
    "smart_9_raw",
    "smart_194_raw",
    "smart_199_raw",
    "smart_196_raw",
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
    # Every SMART column is read as Float64, never inferred. A column that is
    # empty in the first `infer_schema_length` rows of one day's file is
    # otherwise inferred as String; appended to its month, that turns the whole
    # month's column into text while other months stay numeric (seen on real
    # Q2 data for smart_187_raw, a core column).
    header = pl.scan_csv(csv_path, n_rows=0).collect_schema().names()
    lf = pl.scan_csv(
        csv_path,
        infer_schema_length=10_000,
        try_parse_dates=True,
        schema_overrides={c: pl.Float64 for c in header if c.startswith("smart_")},
    )

    names = lf.collect_schema().names()
    # Optional attributes are kept when the file has them. Backblaze has not
    # always published every attribute for every model, so requiring them
    # would reject whole days of data.
    available = [c for c in REQUIRED_COLUMNS if c in names] + [
        c for c in OPTIONAL_SMART_COLUMNS if c in names
    ]
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

    # One row per drive per day in Backblaze's files, so a re-ingested day
    # replaces what is stored (see sink_partitioned_by_month's dedup_keys).
    return sink_partitioned_by_month(lf, bronze_root=bronze_root, dedup_keys=["drive_id", "date"])
