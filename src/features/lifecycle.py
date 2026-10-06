"""Feature Family F — drive metadata and lifecycle context
(docs/dataset_strategy.md section 10.6).
"""

from __future__ import annotations

import polars as pl


def add_lifecycle_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adds `drive_age_days` and `drive_age_days_squared`, computed from each
    drive's first observed date, and `power_on_days` when `power_on_hours` is
    present. Requires the frame to be sorted by
    drive_id, date."""
    first_seen = pl.col("date").min().over("drive_id")
    age_days = (pl.col("date") - first_seen).dt.total_days()

    df = df.with_columns(
        age_days.alias("drive_age_days"),
        (age_days**2).alias("drive_age_days_squared"),
    )
    # `drive_age_days` counts from the first day in the LOADED data, so it is the
    # same for every drive that predates the window. Power-on hours is the
    # drive's own lifetime counter - the real age - where the model reports it.
    if "power_on_hours" in df.columns:
        df = df.with_columns(
            (pl.col("power_on_hours") / 24.0).cast(pl.Float32).alias("power_on_days")
        )
    return df
