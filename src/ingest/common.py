"""Shared, RAM-safe ingestion helpers for landing raw source files into the
Bronze layer.

Rules (see docs/dataset_strategy.md section 5):
  - never call `.collect()` on a full source file;
  - process one file at a time;
  - sink lazily to partitioned Parquet with ZSTD compression;
  - stamp every row with ingestion metadata for auditability.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset

SCHEMA_VERSION = 1


def with_ingestion_metadata(
    lf: pl.LazyFrame,
    *,
    source_dataset: SourceDataset,
    source_file: str,
) -> pl.LazyFrame:
    """Stamp a lazy frame with provenance columns required for every Bronze row."""
    ingested_at = dt.datetime.now(dt.UTC).isoformat()
    return lf.with_columns(
        pl.lit(source_dataset.value).alias("source_dataset"),
        pl.lit(source_file).alias("source_file"),
        pl.lit(ingested_at).alias("ingested_at"),
        pl.lit(SCHEMA_VERSION).alias("schema_version"),
    )


def sink_partitioned_by_month(
    lf: pl.LazyFrame,
    *,
    bronze_root: Path,
    date_column: str = "date",
    dedup_keys: list[str] | None = None,
) -> list[Path]:
    """Sink a lazy frame to `bronze_root/year=YYYY/month=MM/part.parquet`.

    Requires collecting once to know which year/month partitions are present
    (Polars streaming sinks do not yet support Hive-style partitioned writes
    for lazy frames), but does so with `streaming=True` so it still spills to
    disk rather than materializing everything in RAM at once.

    This is called once *per source file* (`pipelines/ingest_backblaze.py`/
    `ingest_smartz.py` loop over one file per call), and a real Backblaze
    quarterly archive ships one CSV per day - so a given month's partition
    is written to by ~30 separate calls, one per day. If an existing
    partition file already exists, its rows are read back and concatenated
    with the new ones (deduplicated on every ingestion column except
    `ingested_at`, so re-running ingestion on an already-ingested file is a
    no-op rather than adding duplicates) - **not** overwritten, which
    would otherwise silently discard every previously-ingested day in that
    month but the last one written.

    Pass `dedup_keys` (e.g. `["drive_id", "date"]`) when the source has one
    row per key: the newest row for a key then REPLACES the stored one.
    Comparing whole rows is not enough once the ingested column set changes -
    a row stored earlier without some columns no longer equals the same row
    re-ingested with them, so every such drive-day would be kept twice."""
    df = lf.with_columns(
        pl.col(date_column).dt.year().alias("_year"),
        pl.col(date_column).dt.month().alias("_month"),
    ).collect(engine="streaming")

    written: list[Path] = []
    for (year, month), part in df.partition_by(["_year", "_month"], as_dict=True).items():
        part = part.drop(["_year", "_month"])
        out_dir = bronze_root / f"year={year:04d}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part.parquet"
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            part = pl.concat([existing, part], how="diagonal_relaxed")
            dedup_subset = dedup_keys or [c for c in part.columns if c != "ingested_at"]
            part = part.unique(subset=dedup_subset, keep="last", maintain_order=True)
        part.write_parquet(out_path, compression="zstd")
        written.append(out_path)
    return written


def profile_bronze_output(paths: list[Path]) -> dict:
    """Compute a lightweight per-file profiling summary for the ingestion report."""
    total_rows = 0
    per_file: list[dict] = []
    for path in paths:
        n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        total_rows += n_rows
        per_file.append({"path": str(path), "row_count": n_rows})
    return {"total_rows": total_rows, "files": per_file}
