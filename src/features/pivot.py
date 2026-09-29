"""Pivot the silver canonical long telemetry table (one row per
drive_id/date/smart_attribute_name) into a wide per-drive-day table
(one row per drive_id/date, one column per attribute's badness value).

Feature-window computations (rolling aggregates, deltas, slopes, event
counts) are far more natural in this wide shape.
"""

from __future__ import annotations

import polars as pl


def pivot_badness_wide(canonical_long: pl.DataFrame) -> pl.DataFrame:
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
