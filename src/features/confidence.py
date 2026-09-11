"""Feature Family D — feature confidence for destructive-action gating
(docs/dataset_strategy.md section 10.4 / 14).

feature_confidence = telemetry_coverage_30d x recency_factor x attribute_coverage_factor
recency_factor      = exp(-hours_since_last_telemetry / tau)
"""

from __future__ import annotations

import math

import polars as pl

DEFAULT_RECENCY_TAU_HOURS = 48.0


def add_feature_confidence(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    recency_tau_hours: float = DEFAULT_RECENCY_TAU_HOURS,
) -> pl.DataFrame:
    """Requires `telemetry_coverage_30d` and `days_since_last_telemetry`
    columns (see src/preprocess/telemetry_gaps.py)."""
    hours_since_last = pl.col("days_since_last_telemetry") * 24.0
    recency_factor = (-hours_since_last / recency_tau_hours).exp()

    n_attrs = len(attributes)
    attribute_coverage_factor = (
        pl.sum_horizontal([pl.col(attr).is_not_null() for attr in attributes]) / n_attrs
        if n_attrs
        else pl.lit(1.0)
    )

    df = df.with_columns(
        hours_since_last.alias("hours_since_last_telemetry"),
        recency_factor.alias("recency_factor"),
        attribute_coverage_factor.alias("attribute_coverage_factor"),
    )
    df = df.with_columns(
        (
            pl.col("telemetry_coverage_30d")
            * pl.col("recency_factor")
            * pl.col("attribute_coverage_factor")
        ).alias("feature_confidence")
    )
    return df


def _recency_factor_scalar(
    hours_since_last: float, tau: float = DEFAULT_RECENCY_TAU_HOURS
) -> float:
    """Non-Polars reference implementation used by unit tests to check the
    expression above matches the documented formula exactly."""
    return math.exp(-hours_since_last / tau)
