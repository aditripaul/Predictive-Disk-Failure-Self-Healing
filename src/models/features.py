"""Assembles a model-ready feature matrix by joining gold features with
labels for a single horizon, per docs/dataset_strategy.md section 16.
"""

from __future__ import annotations

import polars as pl

NON_FEATURE_COLUMNS = {
    "drive_id",
    "date",
    "horizon_days",
    "label",
    "event_date",
    "event_type",
    "observable_until_horizon",
    "days_to_event",
    "split",
    "split_version",
    "split_strategy",
}


def assemble_training_frame(
    gold_features: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    horizon_days: int,
) -> pl.DataFrame:
    """Joins gold features to the label table for one horizon, keeping only
    rows with an observed (non-censored) label."""
    horizon_labels = labels.filter(
        (pl.col("horizon_days") == horizon_days) & pl.col("label").is_not_null()
    )
    return gold_features.join(horizon_labels, on=["drive_id", "date"], how="inner")


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric feature columns only: excludes ids, dates, label/split
    metadata, and non-numeric columns (e.g. model_family strings)."""
    numeric_dtypes = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
    )
    return [
        c
        for c, dtype in zip(df.columns, df.dtypes, strict=True)
        if c not in NON_FEATURE_COLUMNS and dtype in numeric_dtypes
    ]
