"""SMART attribute harmonization: maps vendor-specific raw columns to
canonical attribute names and applies badness-orientation normalization so
that, for every attribute, a higher `smart_badness_value` always means worse
drive health (docs/dataset_strategy.md section 7.2-7.3).
"""

from __future__ import annotations

import polars as pl

# Bronze column -> canonical SMART attribute name.
CANONICAL_ATTRIBUTE_NAMES: dict[str, str] = {
    "smart_5_raw": "reallocated_sector_count",
    "smart_187_raw": "reported_uncorrectable_errors",
    "smart_188_raw": "command_timeout",
    "smart_197_raw": "current_pending_sector_count",
    "smart_198_raw": "offline_uncorrectable",
}

# All attributes ingested so far are raw SMART counters where a higher raw
# value always means worse health, so no inversion is needed. Attributes
# using vendor-normalized scales (where lower can mean worse) should be added
# here as they are onboarded, e.g. {"smart_9_normalized": 253}.
NORMALIZED_SCALE_MAX: dict[str, int] = {}


def melt_smart_attributes(df: pl.DataFrame) -> pl.DataFrame:
    """Reshape wide Bronze rows (one column per SMART attribute) into long
    canonical rows: one row per (drive_id, date, smart_attribute_name)."""
    bronze_columns = [c for c in CANONICAL_ATTRIBUTE_NAMES if c in df.columns]
    if not bronze_columns:
        raise ValueError("No known SMART attribute columns found to melt.")

    id_columns = [c for c in df.columns if c not in bronze_columns]

    long_df = df.unpivot(
        index=id_columns,
        on=bronze_columns,
        variable_name="_bronze_column",
        value_name="smart_raw_value",
    )

    long_df = long_df.with_columns(
        pl.col("_bronze_column")
        .replace_strict(CANONICAL_ATTRIBUTE_NAMES, default=None)
        .alias("smart_attribute_name"),
        pl.col("smart_raw_value").cast(pl.Float32),
    ).drop("_bronze_column")

    # Direction normalization: currently identity for all onboarded raw
    # counters (higher raw value == worse health).
    long_df = long_df.with_columns(pl.col("smart_raw_value").alias("smart_badness_value"))

    return long_df
