"""Feature Family C — threshold crossings and event counts
(docs/dataset_strategy.md section 10.3).
"""

from __future__ import annotations

import polars as pl

DEFAULT_WINDOWS_DAYS = (7, 14, 30)
DEFAULT_SPIKE_THRESHOLD = 5.0


def add_positive_day_counts(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    windows_days: tuple[int, ...] = DEFAULT_WINDOWS_DAYS,
) -> pl.DataFrame:
    """Adds `{attr}_{window}d_positive_count`: days in the window where the
    attribute's badness value was greater than zero.

    Rolls from a narrow `base`, collecting each window's result, then
    combines them with a single horizontal concat (not a join) before
    joining the wide `df` in only once at the end - see
    src/features/windows.py for why neither `df` nor a `combined`
    accumulator can be joined inside this loop without the same cost
    reappearing one level down."""
    base = df.select(["drive_id", "date", *attributes])
    rolled_frames = []
    for window in windows_days:
        period = f"{window}d"
        agg_exprs = [
            (pl.col(attr) > 0).sum().alias(f"{attr}_{window}d_positive_count")
            for attr in attributes
        ]
        rolled_frames.append(
            base.rolling(index_column="date", period=period, group_by="drive_id").agg(agg_exprs)
        )
    combined = pl.concat(
        [
            rolled_frames[0].select(["drive_id", "date"]),
            *[r.drop(["drive_id", "date"]) for r in rolled_frames],
        ],
        how="horizontal_extend",
    )
    return df.join(combined, on=["drive_id", "date"], how="left")


def add_spike_counts(
    df: pl.DataFrame,
    attribute_thresholds: dict[str, float],
    *,
    windows_days: tuple[int, ...] = DEFAULT_WINDOWS_DAYS,
) -> pl.DataFrame:
    """Adds `{attr}_{window}d_spike_count`: days in the window where the
    day-over-day increase exceeded the attribute's configured threshold.

    Same narrow-base/horizontal-concat shape as `add_positive_day_counts`
    (see there, and src/features/windows.py, for why)."""
    base = df.select(["drive_id", "date", *attribute_thresholds])
    daily_delta_exprs = [
        (pl.col(attr) - pl.col(attr).shift(1).over("drive_id")).alias(f"_{attr}_daily_delta")
        for attr in attribute_thresholds
    ]
    base = base.with_columns(daily_delta_exprs)

    rolled_frames = []
    for window in windows_days:
        period = f"{window}d"
        agg_exprs = [
            (pl.col(f"_{attr}_daily_delta") >= threshold)
            .fill_null(False)
            .sum()
            .alias(f"{attr}_{window}d_spike_count")
            for attr, threshold in attribute_thresholds.items()
        ]
        rolled_frames.append(
            base.rolling(index_column="date", period=period, group_by="drive_id").agg(agg_exprs)
        )

    combined = pl.concat(
        [
            rolled_frames[0].select(["drive_id", "date"]),
            *[r.drop(["drive_id", "date"]) for r in rolled_frames],
        ],
        how="horizontal_extend",
    )
    return df.join(combined, on=["drive_id", "date"], how="left")


def add_zero_to_nonzero_flags(df: pl.DataFrame, attributes: list[str]) -> pl.DataFrame:
    """Adds `{attr}_zero_to_nonzero`: 1 if yesterday's value was zero and
    today's is positive."""
    exprs = [
        (
            (pl.col(attr).shift(1).over("drive_id") == 0) & (pl.col(attr) > 0)
        )
        .fill_null(False)
        .alias(f"{attr}_zero_to_nonzero")
        for attr in attributes
    ]
    return df.with_columns(exprs)
