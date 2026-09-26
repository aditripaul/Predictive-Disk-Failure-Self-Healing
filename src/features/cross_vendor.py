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


def add_model_family_zscores(
    df: pl.DataFrame, attributes: list[str], *, windows_days: tuple[int, ...]
) -> pl.DataFrame:
    """For each attribute's `{window}d_mean` rolling feature, adds
    `{attr}_{window}d_model_zscore = (value - model_family_mean) /
    (model_family_std + epsilon)`, with mean/std computed per `model_family`
    across the whole dataset - "some drive models naturally report higher
    error counts" (docs/dataset_strategy.md section 10.5), so this is a
    baseline relative to similar drives, not the fleet as a whole."""
    if "model_family" not in df.columns:
        return df

    exprs = []
    for attr in attributes:
        for window in windows_days:
            col = f"{attr}_{window}d_mean"
            if col not in df.columns:
                continue
            family_mean = pl.col(col).mean().over("model_family")
            family_std = pl.col(col).std().over("model_family").fill_null(0.0)
            exprs.append(
                ((pl.col(col) - family_mean) / (family_std + EPSILON)).alias(
                    f"{attr}_{window}d_model_zscore"
                )
            )
    return df.with_columns(exprs) if exprs else df
