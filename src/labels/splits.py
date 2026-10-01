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


def sample_holdout_drive_ids(
    drive_ids: pl.Series | list[str], *, holdout_fraction: float = 0.1, seed: int = 42
) -> list[str]:
    """The random ~`holdout_fraction` subset of `drive_ids` that
    `apply_drive_level_holdout` reassigns to validation. Extracted so a
    caller processing a dataset in batches (see
    `pipelines/build_labels.py`) can sample this set ONCE, from the full
    fleet-wide drive list, and pass the SAME set to every batch's
    `apply_drive_level_holdout` call - resampling independently per batch
    would give each batch its own random ~10%, silently breaking the
    "these drives, consistently held out everywhere" guarantee the whole
    point of this holdout depends on."""
    unique_ids = pl.Series("drive_id", drive_ids).unique().sort()
    n_holdout = max(1, int(unique_ids.len() * holdout_fraction))
    return unique_ids.sample(n=n_holdout, seed=seed).to_list()


def apply_drive_level_holdout(
    df: pl.DataFrame,
    *,
    holdout_fraction: float = 0.1,
    seed: int = 42,
    holdout_drive_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Reassigns a random subset of drives so they appear only in
    validation/test, never in train — testing generalization to unseen
    physical units. Assumes `add_chronological_split` has already run.

    Pass a precomputed `holdout_drive_ids` (from `sample_holdout_drive_ids`,
    run once over the full fleet) when calling this once per batch of a
    larger, batched dataset - otherwise each batch samples its own
    independent ~`holdout_fraction`, which is wrong (see
    `sample_holdout_drive_ids`'s docstring). Left as `None` (the default),
    this samples from `df`'s own drive_id column exactly as before, which
    is correct only when `df` already contains the whole fleet."""
    if holdout_drive_ids is None:
        drive_ids = df.select("drive_id").unique().sort("drive_id")
        n_holdout = max(1, int(drive_ids.height * holdout_fraction))
        holdout_drive_ids = drive_ids.sample(n=n_holdout, seed=seed)["drive_id"].to_list()

    is_holdout = pl.col("drive_id").is_in(holdout_drive_ids)
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


def apply_vendor_holdout(
    df: pl.DataFrame,
    *,
    source_column: str = "source_dataset",
    external_source: str = "smartz",
) -> pl.DataFrame:
    """External split (docs/dataset_strategy.md section 9.3/9.4): every row
    from `external_source` is reassigned `split="external_smartz"` and
    `split_strategy="vendor_holdout"`, overriding whatever the chronological
    (+ drive-holdout) split assigned. SMART-Z exists to test cross-vendor
    generalization and identify Backblaze-specific overfitting, so it must
    never leak into train/validation - it is evaluation-only. A no-op if the
    frame carries no `source_column` (e.g. a Backblaze-only build)."""
    if source_column not in df.columns:
        return df

    is_external = pl.col(source_column) == external_source
    return df.with_columns(
        pl.when(is_external).then(pl.lit("external_smartz")).otherwise(pl.col("split")).alias(
            "split"
        ),
        pl.when(is_external)
        .then(pl.lit("vendor_holdout"))
        .otherwise(pl.col("split_strategy"))
        .alias("split_strategy"),
    )


def _to_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)
