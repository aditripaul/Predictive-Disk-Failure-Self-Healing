"""Active defect velocity: how fast a drive's defect counts are growing.

Defect counts (reallocated, pending, offline-uncorrectable and reported
uncorrectable errors) are summed per drive-day into one "active defect"
total. Velocity is that total's change over a window divided by the window
length, and acceleration compares a short window's velocity with a long one.
A single counter can move for benign reasons; several defect types rising
together is a stronger signal than any one of them.

Like the existing deltas, windows count observations (rows), not calendar
days, so the rows must already be sorted by date within each drive.
"""

from __future__ import annotations

import polars as pl

DEFECT_ATTRIBUTES = (
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
)
DEFAULT_VELOCITY_WINDOWS_DAYS = (7, 14, 30)
_TOTAL = "_active_defect_total"


def add_defect_velocity(
    df: pl.DataFrame, *, windows_days: tuple[int, ...] = DEFAULT_VELOCITY_WINDOWS_DAYS
) -> pl.DataFrame:
    """Adds `active_defect_total`, `active_defect_velocity_{w}d` (change per
    observation over the last w observations) and
    `active_defect_acceleration_{short}d_vs_{long}d`. Returns `df` unchanged
    if none of the defect attributes are present."""
    attributes = [a for a in DEFECT_ATTRIBUTES if a in df.columns]
    if not attributes or len(windows_days) == 0:
        return df

    total = pl.sum_horizontal([pl.col(a).fill_null(0) for a in attributes])
    df = df.with_columns(total.cast(pl.Float32).alias(_TOTAL))
    velocity_exprs = []
    for window in windows_days:
        lagged = pl.col(_TOTAL).shift(window).over("drive_id")
        velocity_exprs.append(
            ((pl.col(_TOTAL) - lagged) / window)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(f"active_defect_velocity_{window}d")
        )
    df = df.with_columns(velocity_exprs)

    short, long = windows_days[0], windows_days[-1]
    df = df.with_columns(
        (
            pl.col(f"active_defect_velocity_{short}d") - pl.col(f"active_defect_velocity_{long}d")
        ).alias(f"active_defect_acceleration_{short}d_vs_{long}d")
    )
    return df.rename({_TOTAL: "active_defect_total"})
