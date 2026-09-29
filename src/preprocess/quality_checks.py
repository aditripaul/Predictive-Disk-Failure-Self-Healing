"""Data quality checks for the silver canonical telemetry table
(docs/dataset_strategy.md section 18).

Each check returns a small result dict rather than raising, so a single
pipeline run can report every violation instead of stopping at the first one.
Callers (e.g. pipelines/build_silver.py) decide whether any failing check
should abort the run.
"""

from __future__ import annotations

import polars as pl


def _lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Every check below only ever needs a handful of small aggregate
    results (a count, a null count, a filtered row count) - never the full
    table - so each one runs as its own narrow lazy query. This lets
    `run_all_checks` be pointed at a `LazyFrame` scanning an
    already-on-disk Silver Parquet file (see pipelines/build_silver.py,
    which sinks the canonical long table straight to disk rather than
    materializing its ~10x-expanded row count in memory) without ever
    pulling that whole table into memory just to check it, while still
    accepting a plain in-memory `DataFrame` directly (e.g. in tests)."""
    return df.lazy() if isinstance(df, pl.DataFrame) else df


def check_no_null_dates(df: pl.DataFrame | pl.LazyFrame) -> dict:
    n_null = _lazy(df).select(pl.col("date").null_count()).collect().item()
    return {"check": "no_null_dates", "passed": n_null == 0, "violation_count": n_null}


def check_no_duplicate_drive_day_attribute(df: pl.DataFrame | pl.LazyFrame) -> dict:
    lf = _lazy(df)
    key_columns = ["drive_id", "date", "smart_attribute_name"]
    if not set(key_columns).issubset(lf.collect_schema().names()):
        key_columns = ["drive_id", "date"]
    n_rows = lf.select(pl.len()).collect().item()
    n_unique = lf.select(key_columns).unique().select(pl.len()).collect().item()
    n_duplicates = n_rows - n_unique
    return {
        "check": "no_duplicate_drive_day_attribute",
        "passed": n_duplicates == 0,
        "violation_count": n_duplicates,
    }


def check_capacity_positive(df: pl.DataFrame | pl.LazyFrame) -> dict:
    lf = _lazy(df)
    if "capacity_gb" not in lf.collect_schema().names():
        return {"check": "capacity_positive", "passed": True, "violation_count": 0}
    n_bad = lf.filter(pl.col("capacity_gb") <= 0).select(pl.len()).collect().item()
    return {"check": "capacity_positive", "passed": n_bad == 0, "violation_count": n_bad}


def check_failure_date_after_date(df: pl.DataFrame | pl.LazyFrame) -> dict:
    lf = _lazy(df)
    if "failure_date" not in lf.collect_schema().names():
        return {"check": "failure_date_after_date", "passed": True, "violation_count": 0}
    n_bad = (
        lf.filter(pl.col("failure_date").is_not_null() & (pl.col("failure_date") < pl.col("date")))
        .select(pl.len())
        .collect()
        .item()
    )
    return {
        "check": "failure_date_after_date",
        "passed": n_bad == 0,
        "violation_count": n_bad,
    }


def run_all_checks(df: pl.DataFrame | pl.LazyFrame) -> list[dict]:
    lf = _lazy(df)
    return [
        check_no_null_dates(lf),
        check_no_duplicate_drive_day_attribute(lf),
        check_capacity_positive(lf),
        check_failure_date_after_date(lf),
    ]
