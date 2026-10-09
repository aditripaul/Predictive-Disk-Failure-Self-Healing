"""Feature Family D — feature confidence for destructive-action gating
(docs/dataset_strategy.md section 10.4 / 14).

feature_confidence = telemetry_coverage_30d x recency_factor x attribute_coverage_factor
recency_factor      = exp(-hours_since_last_telemetry / tau)

`attribute_coverage_factor` is scored against the attributes the drive's MODEL
reports, not against every priority attribute in the fleet. Vendors publish
different SMART attributes, so a global denominator marks a drive down for
lacking another vendor's attribute - which pinned about two thirds of the
fleet below the destructive-action confidence floor regardless of how good
their telemetry was. See docs/adr/0002.
"""

from __future__ import annotations

import math

import polars as pl

from src.features.cross_vendor import expected_attribute_column

DEFAULT_RECENCY_TAU_HOURS = 48.0


def add_feature_confidence(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    recency_tau_hours: float = DEFAULT_RECENCY_TAU_HOURS,
    expected_by_model: pl.DataFrame | None = None,
    model_column: str = "drive_model",
) -> pl.DataFrame:
    """Requires `telemetry_coverage_30d` and `days_since_last_telemetry`
    columns (see src/preprocess/telemetry_gaps.py).

    `expected_by_model` is `src/features/cross_vendor.py::
    model_expected_attributes`' table: one row per model, one boolean per
    attribute saying whether that model reports it. Given it,
    `attribute_coverage_factor` counts only the attributes that model is
    expected to report, so a drive is never marked down for an attribute its
    vendor does not publish. Without it, every attribute counts for every
    drive - the original behaviour, kept for callers that have no model
    table (and because a fleet of one vendor needs no correction)."""
    hours_since_last = pl.col("days_since_last_telemetry") * 24.0
    recency_factor = (-hours_since_last / recency_tau_hours).exp()

    n_attrs = len(attributes)
    expected_columns: list[str] = []
    if not n_attrs:
        attribute_coverage_factor = pl.lit(1.0)
    elif expected_by_model is None:
        attribute_coverage_factor = (
            pl.sum_horizontal([pl.col(attr).is_not_null() for attr in attributes]) / n_attrs
        )
    else:
        expected_columns = [expected_attribute_column(a) for a in attributes]
        missing = [c for c in expected_columns if c not in expected_by_model.columns]
        if missing:
            raise ValueError(f"expected_by_model is missing {missing}")
        df = df.join(
            expected_by_model.select([model_column, *expected_columns]),
            on=model_column,
            how="left",
        )
        # A model absent from the table (unseen drive_model) is treated as
        # reporting everything: the conservative direction, since it can only
        # lower confidence, never raise it above what the drive has earned.
        df = df.with_columns([pl.col(c).fill_null(True) for c in expected_columns])
        expected_count = pl.sum_horizontal([pl.col(c) for c in expected_columns])
        present_expected = pl.sum_horizontal(
            [pl.col(a).is_not_null() & pl.col(expected_attribute_column(a)) for a in attributes]
        )
        # No attribute expected at all would be a divide-by-zero; such a drive
        # has nothing to be judged on, so its attribute factor is 1.0 and the
        # telemetry and recency terms alone decide its confidence.
        attribute_coverage_factor = (
            pl.when(expected_count > 0)
            .then(present_expected / expected_count)
            .otherwise(pl.lit(1.0))
            .cast(pl.Float64)
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
    # The joined `_expected_*` flags are scaffolding, and pl.Boolean counts as
    # a feature dtype (src/models/features.py::NUMERIC_DTYPES), so leaving them
    # on the frame would silently turn them into model inputs.
    return df.drop(expected_columns) if expected_columns else df


def _recency_factor_scalar(
    hours_since_last: float, tau: float = DEFAULT_RECENCY_TAU_HOURS
) -> float:
    """Non-Polars reference implementation used by unit tests to check the
    expression above matches the documented formula exactly."""
    return math.exp(-hours_since_last / tau)
