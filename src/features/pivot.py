"""Pivot the silver canonical long telemetry table (one row per
drive_id/date/smart_attribute_name) into a wide per-drive-day table
(one row per drive_id/date, one column per attribute's badness value).

Feature-window computations (rolling aggregates, deltas, slopes, event
counts) are far more natural in this wide shape.
"""

from __future__ import annotations

import time

import polars as pl

from src.logging_config import get_logger
from src.preprocess.smart_mapping import ALL_BRONZE_SMART_COLUMNS

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"pivot_badness_wide_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def pivot_badness_wide(
    canonical_long: pl.DataFrame,
    *,
    drive_day: pl.DataFrame | None = None,
    attribute_names: list[str] | None = None,
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
    narrowed `id_columns` path).

    Pass `attribute_names` explicitly when `canonical_long` is one batch
    of a larger, batched pivot (see `pipelines/build_gold_features.py`):
    deriving the attribute list from each batch's own data independently
    risks a batch that happens to be missing some attribute entirely
    producing a different (narrower) schema than the others, which would
    break concatenating the batches back together. Every batch should be
    given the SAME globally-determined list instead, so a batch missing
    an attribute just gets an all-null column for it rather than no
    column at all."""
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

    # Equivalent to `canonical_long.pivot(on="smart_attribute_name",
    # index=id_columns, values="smart_badness_value",
    # aggregate_function="first")`, but via an explicit hash-based
    # group_by/agg rather than `.pivot()` - at fleet scale (millions of
    # distinct (drive_id, date) groups), `.pivot()` internally sorts by
    # the index columns, hitting the exact same expensive string-key
    # row-encoding cost (`arg_sort_multiple` on the string `drive_id`)
    # that crashed the analogous sort in
    # src/preprocess/telemetry_gaps.py. `group_by` uses hash aggregation
    # instead, never sorting by drive_id at all. Requires no duplicate
    # (drive_id, date, smart_attribute_name) rows to be equivalent to
    # `aggregate_function="first"` - guaranteed by
    # `run_all_checks`/`check_no_duplicate_drive_day_attribute` in
    # pipelines/build_silver.py.
    t0 = time.perf_counter()
    if attribute_names is None:
        attribute_names = canonical_long["smart_attribute_name"].unique().to_list()
    wide = canonical_long.group_by(id_columns).agg(
        [
            pl.col("smart_badness_value")
            .filter(pl.col("smart_attribute_name") == attribute)
            .first()
            .alias(attribute)
            for attribute in attribute_names
        ]
    )
    _log_stage("grouped", t0, row_count=wide.height, attribute_count=len(attribute_names))

    if drive_day is not None:
        t0 = time.perf_counter()
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
        _log_stage("drive_day_joined", t0, row_count=wide.height)

    # Sort by `date` alone, not `[drive_id, date]`: at fleet scale, a
    # multi-column sort keyed partly on the string drive_id forces Polars
    # to row-encode that string into the sort key, which is materially
    # more expensive than a plain numeric/date comparison (this is what
    # crashed pipelines/build_silver.py's analogous sort against real
    # data - see src/preprocess/telemetry_gaps.py). A stable global sort
    # by `date` alone still guarantees every drive's own rows land in
    # non-decreasing date order, which is all downstream `.over("drive_id")`
    # window features need.
    t0 = time.perf_counter()
    wide = wide.sort("date")
    _log_stage("sorted", t0, row_count=wide.height)
    return wide
