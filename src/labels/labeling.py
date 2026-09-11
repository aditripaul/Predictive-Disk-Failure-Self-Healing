"""Failure-horizon labeling (docs/dataset_strategy.md section 8).

label = 1 if the drive has a confirmed failure within the next N days
label = 0 if the drive is confirmed healthy for the next N days
label = null/censored if the outcome cannot be safely determined

The label must only use information that would have been available at
prediction time (leakage rule 3: label[t] depends only on events in
(t, t + N]).
"""

from __future__ import annotations

import polars as pl

from src.labels.event_types import FAILURE_EVENT_TYPES


def compute_labels_for_horizon(
    drive_days: pl.DataFrame,
    drive_metadata: pl.DataFrame,
    *,
    horizon_days: int,
) -> pl.DataFrame:
    """`drive_days` needs `drive_id`, `date`. `drive_metadata` needs
    `drive_id`, `event_date`, `event_type`, `last_seen_date` (see
    src/labels/event_types.py). Returns one row per drive-day with the
    label-table schema from docs/dataset_strategy.md section 8.5."""
    joined = drive_days.select(["drive_id", "date"]).join(
        drive_metadata.select(["drive_id", "event_date", "event_type", "last_seen_date"]),
        on="drive_id",
        how="left",
    )

    horizon_end = pl.col("date") + pl.duration(days=horizon_days)

    event_within_horizon = (
        pl.col("event_date").is_not_null()
        & (pl.col("event_date") > pl.col("date"))
        & (pl.col("event_date") <= horizon_end)
    )
    is_failure_event = pl.col("event_type").is_in(list(FAILURE_EVENT_TYPES))
    observable_until_horizon = pl.col("last_seen_date") >= horizon_end

    label = (
        pl.when(event_within_horizon & is_failure_event)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(event_within_horizon & ~is_failure_event)
        .then(pl.lit(None, dtype=pl.Int8))  # ambiguous event: censor, not a default negative
        .when(observable_until_horizon)
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
    )

    days_to_event = (pl.col("event_date") - pl.col("date")).dt.total_days()

    out = joined.with_columns(
        pl.lit(horizon_days).alias("horizon_days"),
        label.alias("label"),
        (observable_until_horizon | event_within_horizon).alias("observable_until_horizon"),
        days_to_event.alias("days_to_event"),
    ).select(
        [
            "drive_id",
            "date",
            "horizon_days",
            "label",
            "event_date",
            "event_type",
            "observable_until_horizon",
            "days_to_event",
        ]
    )
    return out


def compute_labels(
    drive_days: pl.DataFrame,
    drive_metadata: pl.DataFrame,
    *,
    horizons_days: list[int],
) -> pl.DataFrame:
    return pl.concat(
        [
            compute_labels_for_horizon(drive_days, drive_metadata, horizon_days=h)
            for h in horizons_days
        ]
    )
