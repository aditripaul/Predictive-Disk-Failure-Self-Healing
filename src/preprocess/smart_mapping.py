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

import polars as pl

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
    df: pl.DataFrame, *, source_column: str = "source_dataset"
) -> pl.DataFrame:
    """Reshape wide Bronze rows (one column per SMART attribute) into long
    canonical rows: one row per (drive_id, date, smart_attribute_name).

    If `df` carries a `source_column` (as real Bronze output does - see
    `src/ingest/common.py::with_ingestion_metadata`), each source's rows are
    melted using that source's own column-naming template before being
    concatenated, so a mixed Backblaze+SMART-Z frame harmonizes correctly
    even though the two use different bronze column names for the same
    canonical attribute. Callers without that column (e.g. a single-source
    frame in a test) fall back to the Backblaze mapping."""
    if source_column not in df.columns:
        return _melt_with_map(df, CANONICAL_ATTRIBUTE_NAMES)

    parts = []
    for source_value in df[source_column].unique().to_list():
        attribute_map = build_canonical_attribute_map(source_value)
        subset = df.filter(pl.col(source_column) == source_value)
        if any(c in subset.columns for c in attribute_map):
            parts.append(_melt_with_map(subset, attribute_map))
    if not parts:
        raise ValueError("No known SMART attribute columns found to melt for any source.")
    return pl.concat(parts, how="diagonal_relaxed")


def _melt_with_map(df: pl.DataFrame, attribute_map: dict[str, str]) -> pl.DataFrame:
    bronze_columns = [c for c in attribute_map if c in df.columns]
    if not bronze_columns:
        raise ValueError("No known SMART attribute columns found to melt.")

    id_columns = [c for c in df.columns if c not in bronze_columns]

    long_df = df.unpivot(
        index=id_columns,
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
