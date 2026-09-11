"""Feature Family A — time-window rolling aggregates
(docs/dataset_strategy.md section 10.1).

Operates on a wide per-drive-day frame (see src/features/pivot.py), one
attribute per column. Windows are time-based (`Nd`) and grouped by drive_id
so one drive's history never leaks into another drive's window (leakage
rule 2 in docs/dataset_strategy.md section 17).
"""

from __future__ import annotations

import polars as pl

DEFAULT_WINDOWS_DAYS = (7, 14, 30)


def add_rolling_aggregates(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    windows_days: tuple[int, ...] = DEFAULT_WINDOWS_DAYS,
) -> pl.DataFrame:
    """Adds `{attr}_{window}d_{mean,median,min,max,std,range}` for every
    attribute and window. `df` must be sorted by drive_id, date."""
    for window in windows_days:
        agg_exprs = []
        for attr in attributes:
            period = f"{window}d"
            agg_exprs.extend(
                [
                    pl.col(attr).mean().alias(f"{attr}_{window}d_mean"),
                    pl.col(attr).median().alias(f"{attr}_{window}d_median"),
                    pl.col(attr).min().alias(f"{attr}_{window}d_min"),
                    pl.col(attr).max().alias(f"{attr}_{window}d_max"),
                    pl.col(attr).std().fill_null(0.0).alias(f"{attr}_{window}d_std"),
                ]
            )
        rolled = df.rolling(index_column="date", period=period, group_by="drive_id").agg(
            agg_exprs
        )
        for attr in attributes:
            rolled = rolled.with_columns(
                (pl.col(f"{attr}_{window}d_max") - pl.col(f"{attr}_{window}d_min")).alias(
                    f"{attr}_{window}d_range"
                )
            )
        df = df.join(rolled, on=["drive_id", "date"], how="left")

    return df
