"""Feature Family F — drive metadata and lifecycle context
(docs/dataset_strategy.md section 10.6).
"""

from __future__ import annotations

import polars as pl


def add_lifecycle_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adds `drive_age_days` and `drive_age_days_squared`, computed from each
    drive's first observed date. Requires the frame to be sorted by
    drive_id, date."""
    first_seen = pl.col("date").min().over("drive_id")
    age_days = (pl.col("date") - first_seen).dt.total_days()

    return df.with_columns(
        age_days.alias("drive_age_days"),
        (age_days**2).alias("drive_age_days_squared"),
    )
