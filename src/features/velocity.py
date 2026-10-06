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
    df: pl.DataFrame,
    *,
    windows_days: tuple[int, ...] = DEFAULT_VELOCITY_WINDOWS_DAYS,
    acceleration_short_days: int | None = None,
) -> pl.DataFrame:
    """Adds `active_defect_total`, `active_defect_velocity_{w}d` (change per
    observation over the last w observations) and
    `active_defect_acceleration_{short}d_vs_{long}d` (`short` is
    `acceleration_short_days`, default the first window; `long` is the last
    window). Returns `df` unchanged if none of the defect attributes are
    present."""
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

    short = acceleration_short_days if acceleration_short_days is not None else windows_days[0]
    long = windows_days[-1]
    if short not in windows_days:
        raise ValueError(f"acceleration_short_days={short} is not one of {windows_days}")
    df = df.with_columns(
        (
            pl.col(f"active_defect_velocity_{short}d") - pl.col(f"active_defect_velocity_{long}d")
        ).alias(f"active_defect_acceleration_{short}d_vs_{long}d")
    )
    return df.rename({_TOTAL: "active_defect_total"})


def add_temperature_spike(
    df: pl.DataFrame,
    *,
    attribute: str = "temperature_celsius",
    short_observations: int = 3,
    long_observations: int = 30,
) -> pl.DataFrame:
    """Adds `temperature_spike`: the highest temperature over the drive's last
    `short_observations` rows minus its mean over the last `long_observations`
    rows (whatever history exists so far). Returns `df` unchanged without the
    temperature column."""
    if attribute not in df.columns:
        return df
    recent_max = pl.col(attribute).rolling_max(window_size=short_observations, min_samples=1)
    baseline = pl.col(attribute).rolling_mean(window_size=long_observations, min_samples=1)
    return df.with_columns(
        (recent_max.over("drive_id") - baseline.over("drive_id"))
        .cast(pl.Float32)
        .alias("temperature_spike")
    )


#: `days_since_last_increase` for a counter that never increased in the loaded
#: history. A real value can never reach it within a few quarters of data.
NEVER_INCREASED_DAYS = 999


def add_days_since_last_increase(df: pl.DataFrame, attributes: list[str]) -> pl.DataFrame:
    """Adds `{attr}_days_since_last_increase`: days since the counter last went
    up for that drive, `NEVER_INCREASED_DAYS` if it has not gone up in the loaded
    history. "A drive is 39 times more likely to fail within 60 days of its
    first scan error" (Pinheiro et al., 2007) - how recently a counter moved
    matters, not only its level. A counter already above zero on the drive's
    first loaded day counts as never increased: the increase happened before
    the data starts. Needs `date` as a Date column, rows in date order per
    drive."""
    present = [a for a in attributes if a in df.columns]
    if not present:
        return df
    exprs = []
    for attr in present:
        increased = (pl.col(attr) - pl.col(attr).shift(1).over("drive_id")) > 0
        last_increase = (
            pl.when(increased).then(pl.col("date")).otherwise(None).forward_fill().over("drive_id")
        )
        exprs.append(
            (pl.col("date") - last_increase)
            .dt.total_days()
            .fill_null(NEVER_INCREASED_DAYS)
            .clip(upper_bound=NEVER_INCREASED_DAYS)
            .cast(pl.Int16)
            .alias(f"{attr}_days_since_last_increase")
        )
    return df.with_columns(exprs)
