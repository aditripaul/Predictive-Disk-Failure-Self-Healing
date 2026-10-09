"""SMART attribute harmonization: maps vendor-specific raw columns to
canonical attribute names and applies badness-orientation normalization so
that, for every attribute, a higher `smart_badness_value` always means worse
drive health (docs/dataset_strategy.md section 7.2-7.3).

SMART attribute IDs themselves are standardized by the ATA/SMART
specification (ID 5 is always "Reallocated Sector Count" regardless of
vendor) - what differs between sources is the *column-naming convention* a
given export uses for "the raw value of SMART attribute 5". This module
maps each source's naming convention to the same canonical attribute names,
which is what actually makes cross-vendor generalization
(docs/dataset_strategy.md section 2, "Strategic Objectives") possible.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def _columns_of(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    return df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns

#: Standard SMART attribute IDs -> canonical name (docs/dataset_strategy.md
#: section 7.2's mapping table). These 10 are the priority attributes
#: (configs/features.yaml's `priority_smart_attributes`); the two currently
#: onboarded (5, 187, 188, 197, 198) are Backblaze's initial set.
STANDARD_SMART_ID_TO_CANONICAL: dict[int, str] = {
    5: "reallocated_sector_count",
    187: "reported_uncorrectable_errors",
    188: "command_timeout",
    197: "current_pending_sector_count",
    198: "offline_uncorrectable",
    1: "raw_read_error_rate",
    7: "seek_error_rate",
    9: "power_on_hours",
    10: "spin_retry_count",
    199: "udma_crc_error_count",
    194: "temperature_celsius",
    196: "reallocation_event_count",
}

#: Per-source column-naming template for "the raw value of SMART attribute
#: {id}". SMART-Z's real column names haven't been verified against the
#: actual dataset yet (the project-wide "real data gap" -
#: docs/developer_guide.md section 5/13) - this is the best-effort mapping
#: per its published standardized-attribute schema
#: (docs/dataset_strategy.md section 3.2); adjust the template once real
#: SMART-Z files are ingested if the actual column names differ.
SOURCE_COLUMN_TEMPLATES: dict[str, str] = {
    "backblaze": "smart_{id}_raw",
    "smartz": "smart_{id}_normalized",
}

DEFAULT_SOURCE = "backblaze"

#: Every bronze column name any source's template could produce for any
#: standard attribute, regardless of which are actually onboarded/present
#: in a given dataset - i.e. everything `melt_smart_attributes` ever
#: consumes as a `bronze_column` to unpivot. Used by
#: `src/features/pivot.py::pivot_badness_wide` to exclude these from its
#: drive_day join-back, since they're superseded there by the harmonized
#: `smart_badness_value` columns the pivot itself produces - joining them
#: back in too would reintroduce raw, un-harmonized duplicates of data
#: the melt already captured.
ALL_BRONZE_SMART_COLUMNS: frozenset[str] = frozenset(
    template.format(id=smart_id)
    for template in SOURCE_COLUMN_TEMPLATES.values()
    for smart_id in STANDARD_SMART_ID_TO_CANONICAL
)

# All attributes ingested so far are raw SMART counters where a higher raw
# value always means worse health, so no inversion is needed. Attributes
# using vendor-normalized scales (where lower can mean worse) should be added
# here as they are onboarded, e.g. {"smart_9_normalized": 253}.
NORMALIZED_SCALE_MAX: dict[str, int] = {}


def build_canonical_attribute_map(source_dataset: str) -> dict[str, str]:
    """The bronze-column -> canonical-name mapping for one source, e.g.
    `{"smart_5_raw": "reallocated_sector_count", ...}` for Backblaze or
    `{"smart_5_normalized": "reallocated_sector_count", ...}` for SMART-Z."""
    template = SOURCE_COLUMN_TEMPLATES.get(source_dataset, SOURCE_COLUMN_TEMPLATES[DEFAULT_SOURCE])
    return {
        template.format(id=smart_id): name
        for smart_id, name in STANDARD_SMART_ID_TO_CANONICAL.items()
    }


#: Kept for backward compatibility with callers that only ever handled
#: Backblaze (its bronze-column names were always the module-level default).
CANONICAL_ATTRIBUTE_NAMES: dict[str, str] = build_canonical_attribute_map(DEFAULT_SOURCE)


def melt_smart_attributes(
    df: FrameT, *, source_column: str = "source_dataset", id_columns: list[str] | None = None
) -> FrameT:
    """Reshape wide Bronze rows (one column per SMART attribute) into long
    canonical rows: one row per (drive_id, date, smart_attribute_name).

    If `df` carries a `source_column` (as real Bronze output does - see
    `src/ingest/common.py::with_ingestion_metadata`), each source's rows are
    melted using that source's own column-naming template before being
    concatenated, so a mixed Backblaze+SMART-Z frame harmonizes correctly
    even though the two use different bronze column names for the same
    canonical attribute. Callers without that column (e.g. a single-source
    frame in a test) fall back to the Backblaze mapping.

    By default (`id_columns=None`), every non-SMART column is carried
    through unchanged, alongside the new `smart_attribute_name`/
    `smart_raw_value`/`smart_badness_value` ones - simple, but means every
    one of those other columns gets duplicated once per attribute (~10x at
    fleet scale), which is real, avoidable bloat for anything that doesn't
    vary by attribute (drive_model, capacity_gb, telemetry-gap flags, ...).
    Pass an explicit `id_columns` (e.g. `["drive_id", "date"]`) to keep
    only those - see `pipelines/build_silver.py`, which does this and
    persists the rest separately (as `data/silver/drive_day/`) for
    `src/features/pivot.py::pivot_badness_wide` to join back afterward, at
    the drive-day grain rather than the ~10x-larger melted grain.

    Accepts either a `DataFrame` or a `LazyFrame` (every operation below
    works identically in both) so a caller can fold this row expansion
    into a larger lazy pipeline it sinks straight to disk
    (`LazyFrame.sink_parquet`) instead of ever materializing the long-form
    result in memory."""
    columns = _columns_of(df)
    if source_column not in columns:
        return _melt_with_map(df, CANONICAL_ATTRIBUTE_NAMES, id_columns=id_columns)

    if isinstance(df, pl.LazyFrame):
        source_values = df.select(pl.col(source_column).unique()).collect()[source_column].to_list()
    else:
        source_values = df[source_column].unique().to_list()

    if not source_values:
        # A zero-row frame that still carries a source_column in its
        # schema (e.g. one drive-batch of a batched pipeline that happened
        # to get no rows - see pipelines/build_silver.py) has no actual
        # source value to look up a mapping for, but still needs a
        # correctly-shaped (if empty) result: which mapping is used can't
        # matter, since there's no data for it to act on, only a schema
        # for the caller's concat with other (non-empty) batches to match.
        return _melt_with_map(df, CANONICAL_ATTRIBUTE_NAMES, id_columns=id_columns)

    if len(source_values) == 1:
        # The common case (only one source has ever been onboarded into a
        # given Bronze root so far): melt the frame directly rather than
        # `df.filter(...)`-ing out a full redundant copy of it first, which
        # would otherwise sit in memory alongside `df` itself right before
        # the row expansion below.
        attribute_map = build_canonical_attribute_map(source_values[0])
        if not any(c in columns for c in attribute_map):
            raise ValueError("No known SMART attribute columns found to melt for any source.")
        return _melt_with_map(df, attribute_map, id_columns=id_columns)

    parts = []
    for source_value in source_values:
        attribute_map = build_canonical_attribute_map(source_value)
        subset = df.filter(pl.col(source_column) == source_value)
        if any(c in columns for c in attribute_map):
            parts.append(_melt_with_map(subset, attribute_map, id_columns=id_columns))
    if not parts:
        raise ValueError("No known SMART attribute columns found to melt for any source.")
    return pl.concat(parts, how="diagonal_relaxed")


def _melt_with_map(
    df: FrameT, attribute_map: dict[str, str], *, id_columns: list[str] | None = None
) -> FrameT:
    columns = _columns_of(df)
    bronze_columns = [c for c in attribute_map if c in columns]
    if not bronze_columns:
        raise ValueError("No known SMART attribute columns found to melt.")

    resolved_id_columns = (
        id_columns if id_columns is not None else [c for c in columns if c not in bronze_columns]
    )

    long_df = df.unpivot(
        index=resolved_id_columns,
        on=bronze_columns,
        variable_name="_bronze_column",
        value_name="smart_raw_value",
    )

    long_df = long_df.with_columns(
        pl.col("_bronze_column")
        .replace_strict(attribute_map, default=None)
        .alias("smart_attribute_name"),
        pl.col("smart_raw_value").cast(pl.Float32),
    ).drop("_bronze_column")

    # Direction normalization: currently identity for all onboarded raw
    # counters (higher raw value == worse health).
    long_df = long_df.with_columns(pl.col("smart_raw_value").alias("smart_badness_value"))

    return long_df
