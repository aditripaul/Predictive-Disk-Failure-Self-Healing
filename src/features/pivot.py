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
    return wide.sort(["drive_id", "date"])
