"""Event-type classification for drive removals (docs/dataset_strategy.md
section 8.3). Not all drive removals are the same, and ambiguous events must
not silently become negative labels.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

CONFIRMED_FAILURE = "confirmed_failure"
FAILURE_FOLLOWED_BY_REPLACEMENT = "failure_followed_by_replacement"
PREVENTIVE_REPLACEMENT = "preventive_replacement"
REMOVED_WITHOUT_FAILURE = "removed_without_failure"
STILL_ACTIVE = "still_active"

#: event types that represent a genuine failure and should drive a positive label
FAILURE_EVENT_TYPES = {CONFIRMED_FAILURE, FAILURE_FOLLOWED_BY_REPLACEMENT}


def classify_event_types(drive_metadata: pl.DataFrame, *, as_of_date: dt.date) -> pl.DataFrame:
    """Adds `event_date` and `event_type` to a drive_metadata frame. The
    optional `failure_date` / `removal_date` columns (nullable dates) are
    treated as entirely absent if the source ingestion never populated them;
    `last_seen_date` is required."""
    has_failure_col = "failure_date" in drive_metadata.columns
    has_removal_col = "removal_date" in drive_metadata.columns

    has_failure = pl.col("failure_date").is_not_null() if has_failure_col else pl.lit(False)
    has_removal = pl.col("removal_date").is_not_null() if has_removal_col else pl.lit(False)
    removal_follows_failure = (
        has_failure & has_removal & (pl.col("removal_date") >= pl.col("failure_date"))
        if (has_removal_col and has_failure_col)
        else pl.lit(False)
    )
    is_decommissioned = pl.col("last_seen_date") < as_of_date

    event_type = (
        pl.when(removal_follows_failure)
        .then(pl.lit(FAILURE_FOLLOWED_BY_REPLACEMENT))
        .when(has_failure)
        .then(pl.lit(CONFIRMED_FAILURE))
        .when(has_removal)
        .then(pl.lit(PREVENTIVE_REPLACEMENT))
        .when(is_decommissioned)
        .then(pl.lit(REMOVED_WITHOUT_FAILURE))
        .otherwise(pl.lit(STILL_ACTIVE))
    )

    failure_date_expr = (
        pl.col("failure_date") if has_failure_col else pl.lit(None, dtype=pl.Date)
    )
    removal_date_expr = (
        pl.col("removal_date") if has_removal_col else pl.lit(None, dtype=pl.Date)
    )
    event_date = pl.when(has_failure).then(failure_date_expr).otherwise(removal_date_expr)

    return drive_metadata.with_columns(
        event_type.alias("event_type"),
        event_date.alias("event_date"),
    )
