"""Derives `failure_date` from the raw per-day `failure` flag Backblaze
(and this project's synthetic stub) reports: `1` on the day a drive was
marked failed, `0` otherwise (docs/dataset_strategy.md section 3.1).

Every downstream consumer of failure information
(`src/preprocess/feature_maturity.py::build_drive_metadata`,
`src/labels/event_types.py::classify_event_types`) expects a single
`failure_date` per drive, not a per-day flag - without this step, a raw
`failure` column is silently ignored and no drive is ever classified as
`CONFIRMED_FAILURE`, so no label is ever positive even with real data
that has genuine failures in it.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def derive_failure_date(df: FrameT, *, failure_column: str = "failure") -> FrameT:
    """Adds a `failure_date` column: the date `failure_column == 1` for a
    drive, broadcast to every row of that drive (null for drives that
    never failed). A no-op if `failure_column` isn't present (e.g. a
    source that doesn't report a per-day failure flag). Accepts either a
    `DataFrame` or a `LazyFrame` (the window expression below works
    identically in both) so a caller can fold this into a larger lazy
    pipeline instead of forcing an eager materialization here."""
    columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    if failure_column not in columns:
        return df

    failure_date = (
        pl.when(pl.col(failure_column) == 1)
        .then(pl.col("date"))
        .otherwise(None)
        .max()
        .over("drive_id")
    )
    return df.with_columns(failure_date.alias("failure_date"))
