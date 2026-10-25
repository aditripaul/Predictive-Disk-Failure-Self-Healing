"""Data profiling for freshly ingested Bronze partitions.

Produces the row-count / missingness / failure-rate report required by
Phase 1's exit criteria, written to data/audit/data_quality_reports/.

Every figure here is accumulated one Bronze partition file at a time, with
only the columns a figure needs projected out of it - the same
batch-by-whole-files discipline the rest of the pipeline uses. Profiling is
a report about the data; it must never be the stage that sets the pipeline's
memory ceiling. It was: a single 16-column `null_count()` aggregation over a
365-day Bronze dataset (~125M rows) materialized every column at once and
tripped the 20 GB cap at the very end of an otherwise successful ingest.

Row counts and null counts do not need the data read at all. Parquet stores
both in each row group's footer statistics, so they come from metadata -
exact, and essentially free.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def bronze_parts(bronze_root: Path) -> list[Path]:
    """The Bronze partition files, in a deterministic order."""
    return sorted(bronze_root.glob("**/*.parquet"))


def bronze_columns(parts: list[Path]) -> list[str]:
    """Every column present across the partitions, in order of first
    appearance - read from the footers, so it costs nothing.

    The union, not the first partition's schema: Bronze keeps whichever
    optional SMART columns each source file carried
    (`src/ingest/backblaze.py::OPTIONAL_SMART_COLUMNS`), and Backblaze's
    column set has widened over time, so partitions from different months
    need not agree. Reading one partition's schema would silently report a
    column absent there as 0% null across the whole dataset."""
    names: dict[str, None] = {}
    for part in parts:
        for name in pq.ParquetFile(part).schema_arrow.names:
            names.setdefault(name, None)
    return list(names)


def _counts_from_parquet_metadata(parts: list[Path]) -> tuple[int, dict[str, int]] | None:
    """`(row_count, {column: null_count})` read from the Parquet footers.

    Reads no column data, so its cost is independent of how many rows the
    dataset holds. Returns None if any row group is missing the statistic, so
    the caller falls back to computing it rather than reporting a wrong
    number from a partial sum.

    A column absent from a partition is charged that partition's rows as
    nulls, which is what reading the data would report once the partitions
    are scanned together."""
    total_rows = 0
    nulls: dict[str, int] = {}
    all_columns = bronze_columns(parts)
    for part in parts:
        metadata = pq.ParquetFile(part).metadata
        total_rows += metadata.num_rows
        present: set[str] = set()
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.statistics is None or column.statistics.null_count is None:
                    return None
                name = column.path_in_schema
                present.add(name)
                nulls[name] = nulls.get(name, 0) + column.statistics.null_count
        for name in all_columns:
            if name not in present:
                nulls[name] = nulls.get(name, 0) + metadata.num_rows
    return total_rows, nulls


def _counts_by_reading(parts: list[Path], schema_names: list[str]) -> tuple[int, dict[str, int]]:
    """The fallback for Parquet without null-count statistics: one file at a
    time, so peak memory is set by the largest partition rather than by the
    dataset."""
    total_rows = 0
    nulls = dict.fromkeys(schema_names, 0)
    for part in parts:
        lf = pl.scan_parquet(part)
        present = [c for c in schema_names if c in lf.collect_schema().names()]
        counted = lf.select(
            pl.len().alias("_rows"),
            *[pl.col(c).null_count().alias(c) for c in present],
        ).collect()
        rows = counted["_rows"][0]
        total_rows += rows
        for name in schema_names:
            # Absent from this partition means null for all of its rows, the
            # same as scanning the partitions together would report.
            nulls[name] += counted[name][0] if name in present else rows
    return total_rows, nulls


def _failure_count(parts: list[Path]) -> int:
    """Projected to the one column, so each file costs a single Int8 array."""
    total = 0
    for part in parts:
        lf = pl.scan_parquet(part)
        if "failure" not in lf.collect_schema().names():
            continue
        total += lf.select(pl.col("failure").sum()).collect().item() or 0
    return total


def _model_distribution(parts: list[Path], model_column: str) -> list[dict]:
    """Per-model row counts, merged across partitions. A drive model spans
    months, so the per-file counts are summed rather than concatenated."""
    counts: dict[str, int] = {}
    for part in parts:
        lf = pl.scan_parquet(part)
        if model_column not in lf.collect_schema().names():
            continue
        grouped = (
            lf.select(model_column).group_by(model_column).agg(pl.len().alias("count")).collect()
        )
        for row in grouped.iter_rows(named=True):
            counts[row[model_column]] = counts.get(row[model_column], 0) + row["count"]
    return [
        {model_column: model, "count": count}
        for model, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    ]


def profile_bronze_dataset(bronze_root: Path) -> dict:
    parts = bronze_parts(bronze_root)
    schema_names = bronze_columns(parts)

    from_metadata = _counts_from_parquet_metadata(parts) if parts else (0, {})
    if from_metadata is None:
        n_rows, null_counts = _counts_by_reading(parts, schema_names)
    else:
        n_rows, null_counts = from_metadata

    null_pct: dict[str, float] = {}
    if n_rows > 0:
        null_pct = {c: round(null_counts.get(c, 0) / n_rows, 4) for c in schema_names}

    report: dict = {
        "bronze_root": str(bronze_root),
        "profiled_at": dt.datetime.now(dt.UTC).isoformat(),
        "row_count": n_rows,
        "columns": schema_names,
        "null_percentage_by_column": null_pct,
        "partition_count": len(parts),
    }

    if "model" in schema_names or "drive_model" in schema_names:
        model_col = "drive_model" if "drive_model" in schema_names else "model"
        report["drive_model_distribution"] = _model_distribution(parts, model_col)

    if "failure" in schema_names and n_rows > 0:
        report["failure_rate"] = round(_failure_count(parts) / n_rows, 6)

    return report


def write_profiling_report(report: dict, audit_dir: Path, name: str) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / f"{name}_profile.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return out_path
