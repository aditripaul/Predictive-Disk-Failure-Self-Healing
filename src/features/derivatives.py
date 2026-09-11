"""Feature Family B — derivatives and rate of change
(docs/dataset_strategy.md section 10.2).

`slope_W` is approximated as the secant rate of change over the window
(`delta_W / W`) rather than a full least-squares fit. This is a deliberate
MVP simplification: it is monotonically related to the least-squares slope
for roughly-linear trajectories and is far cheaper to compute in Polars.
Revisit with a proper rolling regression if validation shows it matters.
"""

from __future__ import annotations

import polars as pl

DEFAULT_DELTA_WINDOWS_DAYS = (7, 14, 30)


def add_deltas(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    windows_days: tuple[int, ...] = DEFAULT_DELTA_WINDOWS_DAYS,
) -> pl.DataFrame:
    """Adds `{attr}_{window}d_delta` and `{attr}_{window}d_slope`."""
    exprs = []
    for attr in attributes:
        for window in windows_days:
            lagged = pl.col(attr).shift(window).over("drive_id")
            delta = (pl.col(attr) - lagged).alias(f"{attr}_{window}d_delta")
            exprs.append(delta)
    df = df.with_columns(exprs)

    slope_exprs = [
        (pl.col(f"{attr}_{window}d_delta") / window).alias(f"{attr}_{window}d_slope")
        for attr in attributes
        for window in windows_days
    ]
    df = df.with_columns(slope_exprs)
    return df


def add_acceleration(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    short_window_days: int = 7,
    long_window_days: int = 30,
) -> pl.DataFrame:
    """Adds `{attr}_acceleration_{short}d_vs_{long}d = slope_short - slope_long`."""
    exprs = [
        (
            pl.col(f"{attr}_{short_window_days}d_slope")
            - pl.col(f"{attr}_{long_window_days}d_slope")
        ).alias(f"{attr}_acceleration_{short_window_days}d_vs_{long_window_days}d")
        for attr in attributes
    ]
    return df.with_columns(exprs)
