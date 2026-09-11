"""Temporal and drive-level splitting (docs/dataset_strategy.md section 9).

Random splitting is forbidden because it leaks future information; splits
must be chronological, with an optional drive-level holdout to test whether
the model is memorizing specific drives rather than generalizing.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

SPLIT_VERSION = "v1.0"


def add_chronological_split(
    df: pl.DataFrame,
    *,
    train_end: str | dt.date,
    validation_end: str | dt.date,
    date_column: str = "date",
) -> pl.DataFrame:
    """Adds a `split` column: "train" up to `train_end`, "validation" up to
    `validation_end`, "test" afterwards."""
    train_end = _to_date(train_end)
    validation_end = _to_date(validation_end)

    split_expr = (
        pl.when(pl.col(date_column) <= train_end)
        .then(pl.lit("train"))
        .when(pl.col(date_column) <= validation_end)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
    )

    return df.with_columns(
        split_expr.alias("split"),
        pl.lit(SPLIT_VERSION).alias("split_version"),
        pl.lit("chronological").alias("split_strategy"),
    )


def apply_drive_level_holdout(
    df: pl.DataFrame,
    *,
    holdout_fraction: float = 0.1,
    seed: int = 42,
) -> pl.DataFrame:
    """Reassigns a random subset of drives so they appear only in
    validation/test, never in train — testing generalization to unseen
    physical units. Assumes `add_chronological_split` has already run."""
    drive_ids = df.select("drive_id").unique().sort("drive_id")
    n_holdout = max(1, int(drive_ids.height * holdout_fraction))
    holdout_ids = drive_ids.sample(n=n_holdout, seed=seed)["drive_id"].to_list()

    is_holdout = pl.col("drive_id").is_in(holdout_ids)
    df = df.with_columns(
        pl.when(is_holdout & (pl.col("split") == "train"))
        .then(pl.lit("validation"))
        .otherwise(pl.col("split"))
        .alias("split"),
        pl.when(is_holdout)
        .then(pl.lit("drive_holdout"))
        .otherwise(pl.col("split_strategy"))
        .alias("split_strategy"),
    )
    return df


def _to_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)
