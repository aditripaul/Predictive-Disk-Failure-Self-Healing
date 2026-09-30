"""Feature Family E — cross-vendor harmonization and model-relative
features (docs/dataset_strategy.md section 10.5): attribute ratios and
model-family z-scores that compare a drive to similar drives (same model
family) rather than the whole fleet, reducing false alarms caused by
model-specific baselines and improving SMART-Z cross-vendor generalization.
"""

from __future__ import annotations

import polars as pl

EPSILON = 1e-6

#: (numerator, denominator, output name) - the `+1` on the denominator
#: avoids division by zero, per docs/dataset_strategy.md section 10.5.
ATTRIBUTE_RATIOS: tuple[tuple[str, str, str], ...] = (
    ("current_pending_sector_count", "reallocated_sector_count", "pending_to_reallocated_ratio"),
    ("offline_uncorrectable", "power_on_hours", "uncorrectable_per_power_on_hour"),
)


def add_attribute_ratios(df: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        (pl.col(numerator) / (pl.col(denominator) + 1)).alias(name)
        for numerator, denominator, name in ATTRIBUTE_RATIOS
        if numerator in df.columns and denominator in df.columns
    ]
    if {"reallocated_sector_count", "capacity_gb"} <= set(df.columns):
        exprs.append(
            (pl.col("reallocated_sector_count") / pl.col("capacity_gb")).alias(
                "reallocated_per_capacity"
            )
        )
    return df.with_columns(exprs) if exprs else df


def model_family_zscore_plan(
    columns: list[str], attributes: list[str], *, windows_days: tuple[int, ...]
) -> list[tuple[str, str]]:
    """`(rolling-mean column, z-score output column)` for every
    attribute/window whose rolling mean is actually present. Empty when the
    frame has no `model_family` column - there is then nothing to be
    relative to."""
    if "model_family" not in columns:
        return []
    plan = []
    for attr in attributes:
        for window in windows_days:
            col = f"{attr}_{window}d_mean"
            if col in columns:
                plan.append((col, f"{attr}_{window}d_model_zscore"))
    return plan


def _family_mean_column(column: str) -> str:
    return f"__{column}_family_mean"


def _family_std_column(column: str) -> str:
    return f"__{column}_family_std"


def add_model_family_zscores(
    df: pl.DataFrame, attributes: list[str], *, windows_days: tuple[int, ...]
) -> pl.DataFrame:
    """For each attribute's `{window}d_mean` rolling feature, adds
    `{attr}_{window}d_model_zscore = (value - model_family_mean) /
    (model_family_std + epsilon)`, with mean/std computed per `model_family`
    across the whole dataset - "some drive models naturally report higher
    error counts" (docs/dataset_strategy.md section 10.5), so this is a
    baseline relative to similar drives, not the fleet as a whole.

    Requires the whole fleet in one frame, because the per-family mean/std
    span every drive. When the feature pipeline runs in per-drive batches
    (see pipelines/build_gold_features.py), use
    `model_family_zscore_stats` + `add_model_family_zscores_from_stats`
    instead, which split that into a fleet-wide aggregate and a
    per-batch application."""
    plan = model_family_zscore_plan(df.columns, attributes, windows_days=windows_days)
    if not plan:
        return df

    exprs = [
        (
            (pl.col(col) - pl.col(col).mean().over("model_family"))
            / (pl.col(col).std().over("model_family").fill_null(0.0) + EPSILON)
        ).alias(name)
        for col, name in plan
    ]
    return df.with_columns(exprs)


def model_family_zscore_stats(
    source: pl.LazyFrame, plan: list[tuple[str, str]]
) -> pl.DataFrame:
    """Fleet-wide per-`model_family` mean and std of each rolling-mean
    column in `plan` - the only cross-drive statistic the gold feature
    pipeline needs. Narrow by construction (one row per model family), so
    this can be aggregated straight off the per-batch Parquet files
    without materializing any of them."""
    mean_columns = [col for col, _ in plan]
    return (
        source.select(["model_family", *mean_columns])
        .group_by("model_family")
        .agg(
            [pl.col(col).mean().alias(_family_mean_column(col)) for col in mean_columns]
            + [pl.col(col).std().alias(_family_std_column(col)) for col in mean_columns]
        )
        .collect()
    )


def add_model_family_zscores_from_stats(
    df: pl.DataFrame, stats: pl.DataFrame, plan: list[tuple[str, str]]
) -> pl.DataFrame:
    """Applies the z-scores `add_model_family_zscores` would have produced,
    taking the per-family mean/std from `model_family_zscore_stats` instead
    of computing them with `.over("model_family")` over the whole fleet.

    `nulls_equal=True` matches `.over`'s grouping semantics, which treat a
    null `model_family` as its own group rather than as unmatched."""
    if not plan:
        return df

    joined = df.join(stats, on="model_family", how="left", nulls_equal=True)
    exprs = [
        (
            (pl.col(col) - pl.col(_family_mean_column(col)))
            / (pl.col(_family_std_column(col)).fill_null(0.0) + EPSILON)
        ).alias(name)
        for col, name in plan
    ]
    helper_columns = [f(col) for col, _ in plan for f in (_family_mean_column, _family_std_column)]
    return joined.with_columns(exprs).drop(helper_columns)
