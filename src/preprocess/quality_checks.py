"""Data quality checks for the silver canonical telemetry table
(docs/dataset_strategy.md section 18).

Each check returns a small result dict rather than raising, so a single
pipeline run can report every violation instead of stopping at the first one.
Callers (e.g. pipelines/build_silver.py) decide whether any failing check
should abort the run.
"""

from __future__ import annotations

import polars as pl


def check_no_null_dates(df: pl.DataFrame) -> dict:
    n_null = df["date"].null_count()
    return {"check": "no_null_dates", "passed": n_null == 0, "violation_count": n_null}


def check_no_duplicate_drive_day_attribute(df: pl.DataFrame) -> dict:
    key_columns = ["drive_id", "date", "smart_attribute_name"]
    if not set(key_columns).issubset(df.columns):
        key_columns = ["drive_id", "date"]
    n_rows = df.height
    n_unique = df.select(key_columns).unique().height
    n_duplicates = n_rows - n_unique
    return {
        "check": "no_duplicate_drive_day_attribute",
        "passed": n_duplicates == 0,
        "violation_count": n_duplicates,
    }


def check_capacity_positive(df: pl.DataFrame) -> dict:
    if "capacity_gb" not in df.columns:
        return {"check": "capacity_positive", "passed": True, "violation_count": 0}
    n_bad = df.filter(pl.col("capacity_gb") <= 0).height
    return {"check": "capacity_positive", "passed": n_bad == 0, "violation_count": n_bad}


def check_failure_date_after_date(df: pl.DataFrame) -> dict:
    if "failure_date" not in df.columns:
        return {"check": "failure_date_after_date", "passed": True, "violation_count": 0}
    n_bad = df.filter(
        pl.col("failure_date").is_not_null() & (pl.col("failure_date") < pl.col("date"))
    ).height
    return {
        "check": "failure_date_after_date",
        "passed": n_bad == 0,
        "violation_count": n_bad,
    }


def run_all_checks(df: pl.DataFrame) -> list[dict]:
    return [
        check_no_null_dates(df),
        check_no_duplicate_drive_day_attribute(df),
        check_capacity_positive(df),
        check_failure_date_after_date(df),
    ]
