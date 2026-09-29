"""Telemetry-gap and staleness features (docs/dataset_strategy.md section 7.4).

Missing SMART telemetry is treated as potentially informative rather than
harmless silence: a failing drive may stop reporting because its agent
crashed or the drive became unresponsive.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

DEFAULT_SHORT_GAP_DAYS = 3
DEFAULT_STALE_GAP_DAYS = 7

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def compute_telemetry_gaps(
    df: FrameT,
    *,
    short_gap_days: int = DEFAULT_SHORT_GAP_DAYS,
    stale_gap_days: int = DEFAULT_STALE_GAP_DAYS,
) -> FrameT:
    """Adds `days_since_last_telemetry`, `telemetry_gap_flag`,
    `stale_telemetry_flag`, and `telemetry_coverage_30d` to a per-drive-day
    table (one row per drive_id + date; call after de-duplicating attributes,
    e.g. on the drive-day grain, not the long attribute grain).

    Must be partitioned by drive_id and ordered by date — a single drive's
    history must never leak into another drive's window (leakage rule 2).

    Accepts either a `DataFrame` or a `LazyFrame`, so a caller can fold this
    into a larger lazy pipeline it collects only once.
    """
    # Sort by `date` alone rather than `[drive_id, date]`. At fleet scale
    # (tens of millions of rows), a multi-column sort keyed partly on the
    # string drive_id forces Polars to row-encode that string into the sort
    # key, which is what actually exhausted memory here (confirmed by a
    # real crash inside arg_sort_multiple::<BinaryType>) - not the row
    # count itself. A single-key sort on `date` (a plain numeric/date
    # comparison, no row-encoding) is dramatically cheaper, and a stable
    # global sort by `date` alone still guarantees every drive's own subset
    # of rows lands in non-decreasing date order - the only ordering
    # `.rolling(group_by=...)` below actually requires (verified: it does
    # NOT require the groups themselves to be contiguous or sorted, only
    # that each group's index column is ascending within itself).
    df = df.sort("date")

    df = df.with_columns(
        (
            pl.col("date") - pl.col("date").shift(1).over("drive_id", order_by="date")
        ).dt.total_days().fill_null(0).alias("days_since_last_telemetry")
    )

    df = df.with_columns(
        (pl.col("days_since_last_telemetry") > short_gap_days).alias("telemetry_gap_flag"),
        (pl.col("days_since_last_telemetry") > stale_gap_days).alias("stale_telemetry_flag"),
    )

    coverage = (
        df.rolling(index_column="date", period="30d", group_by="drive_id")
        .agg(pl.len().alias("_coverage_count_30d"))
        .with_columns((pl.col("_coverage_count_30d") / 30.0).clip(upper_bound=1.0).alias(
            "telemetry_coverage_30d"
        ))
        .drop("_coverage_count_30d")
    )

    df = df.join(coverage, on=["drive_id", "date"], how="left")

    return df
