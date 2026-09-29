"""Drive-level metadata and feature-maturity classification
(docs/dataset_strategy.md section 7.5).

Rolling features need at least `min_history_days` of observed history before
they can be trusted, so drives are tagged WARMUP / MATURE / STALE /
DECOMMISSIONED. Immature or stale drives may still be monitored, but should
not trigger high-confidence destructive autonomous actions.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

DEFAULT_MIN_HISTORY_DAYS = 30


def build_drive_metadata(
    telemetry_df: pl.DataFrame | pl.LazyFrame,
    *,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    as_of_date: dt.date | None = None,
) -> pl.DataFrame:
    """`telemetry_df` must be at the drive-day grain (already deduplicated
    across SMART attributes) and must carry `stale_telemetry_flag`,
    `failure_date`, `removal_date`, `model_family`, `capacity_gb`, and
    `drive_type` columns where available.

    Accepts either a `DataFrame` or a `LazyFrame` - the output here is
    always tiny (one row per drive), so a caller with a much larger
    fleet-scale `telemetry_df` (e.g. scanned lazily from an on-disk
    Parquet file rather than held in memory - see
    pipelines/build_silver.py) never needs to materialize it just to call
    this."""
    lf = telemetry_df.lazy() if isinstance(telemetry_df, pl.DataFrame) else telemetry_df
    columns = lf.collect_schema().names()
    if as_of_date is not None:
        as_of = as_of_date
    else:
        as_of = lf.select(pl.col("date").max()).collect().item()

    agg_exprs = [
        pl.col("date").min().alias("first_seen_date"),
        pl.col("date").max().alias("last_seen_date"),
        pl.col("date").n_unique().alias("total_observed_days"),
        pl.col("model_family").first().alias("model_family"),
        pl.col("capacity_gb").first().alias("capacity_gb"),
    ]
    if "failure_date" in columns:
        agg_exprs.append(pl.col("failure_date").max().alias("failure_date"))
    if "removal_date" in columns:
        agg_exprs.append(pl.col("removal_date").max().alias("removal_date"))
    has_stale_flag = "stale_telemetry_flag" in columns
    if has_stale_flag:
        agg_exprs.append(pl.col("stale_telemetry_flag").last().alias("_is_currently_stale"))

    metadata = lf.group_by("drive_id").agg(agg_exprs)

    metadata = metadata.with_columns(
        (pl.col("last_seen_date") - pl.col("first_seen_date"))
        .dt.total_days()
        .add(1)
        .alias("active_history_days"),
        (pl.col("total_observed_days") >= min_history_days).alias("survivorship_valid"),
    )

    is_decommissioned = pl.col("last_seen_date") < as_of
    is_currently_stale = pl.col("_is_currently_stale") if has_stale_flag else pl.lit(False)
    is_mature = pl.col("total_observed_days") >= min_history_days

    metadata = metadata.with_columns(
        pl.when(is_decommissioned)
        .then(pl.lit("DECOMMISSIONED"))
        .when(is_currently_stale)
        .then(pl.lit("STALE"))
        .when(is_mature)
        .then(pl.lit("MATURE"))
        .otherwise(pl.lit("WARMUP"))
        .alias("feature_maturity")
    )

    if has_stale_flag:
        metadata = metadata.drop("_is_currently_stale")

    return metadata.collect()
