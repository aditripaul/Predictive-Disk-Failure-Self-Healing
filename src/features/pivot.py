"""Pivot the silver canonical long telemetry table (one row per
drive_id/date/smart_attribute_name) into a wide per-drive-day table
(one row per drive_id/date, one column per attribute's badness value).

Feature-window computations (rolling aggregates, deltas, slopes, event
counts) are far more natural in this wide shape.
"""

from __future__ import annotations

import polars as pl

from src.preprocess.smart_mapping import ALL_BRONZE_SMART_COLUMNS


def pivot_badness_wide(
    canonical_long: pl.DataFrame, *, drive_day: pl.DataFrame | None = None
) -> pl.DataFrame:
    """`canonical_long` may only carry `drive_id`/`date` alongside the
    melted attribute columns (see
    `src/preprocess/smart_mapping.py::melt_smart_attributes`'s
    `id_columns` - `pipelines/build_silver.py` passes a narrowed frame
    like this to avoid ~10x-duplicating every other drive-day column at
    fleet scale). Pass the corresponding `drive_day` table (one row per
    drive-day, e.g. read from `data/silver/drive_day/`) to left-join its
    other columns (drive_model, capacity_gb, telemetry-gap flags, ...)
    back in here instead, at this grain - the same row count as
    `canonical_long` had before the ~10x melt, so this join is far
    cheaper than carrying those columns through the melt itself would
    have been. Omit it if `canonical_long` already carries everything it
    needs (e.g. a test fixture, or any other caller not using the
    narrowed `id_columns` path)."""
    id_columns = [
        c
        for c in canonical_long.columns
        if c
        not in {
            "smart_attribute_name",
            "smart_raw_value",
            "smart_badness_value",
        }
    ]

    wide = canonical_long.pivot(
        on="smart_attribute_name",
        index=id_columns,
        values="smart_badness_value",
        aggregate_function="first",
    )

    if drive_day is not None:
        extra_columns = [
            c
            for c in drive_day.columns
            if c not in wide.columns and c not in ALL_BRONZE_SMART_COLUMNS
        ]
        wide = wide.join(
            drive_day.select(["drive_id", "date", *extra_columns]),
            on=["drive_id", "date"],
            how="left",
        )

    # Sort by `date` alone, not `[drive_id, date]`: at fleet scale, a
    # multi-column sort keyed partly on the string drive_id forces Polars
    # to row-encode that string into the sort key, which is materially
    # more expensive than a plain numeric/date comparison (this is what
    # crashed pipelines/build_silver.py's analogous sort against real
    # data - see src/preprocess/telemetry_gaps.py). A stable global sort
    # by `date` alone still guarantees every drive's own rows land in
    # non-decreasing date order, which is all downstream `.over("drive_id")`
    # window features need.
    return wide.sort("date")
