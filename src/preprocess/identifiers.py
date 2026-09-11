"""Identifier normalization: drive_id cleanup, capacity parsing, and
model-family / manufacturer inference.

The model-family table is intentionally small and pattern-based (prefix
match on the raw model string). It is meant to be extended as new drive
models are observed in real Backblaze/SMART-Z data; unmatched models fall
back to a normalized version of the raw model string with an unknown
manufacturer.
"""

from __future__ import annotations

import polars as pl

# (model prefix, manufacturer, model_family)
MODEL_FAMILY_TABLE: list[tuple[str, str, str]] = [
    ("ST", "Seagate", "Seagate HDD"),
    ("WDC", "Western Digital", "Western Digital HDD"),
    ("WUS", "Western Digital", "Western Digital Enterprise HDD"),
    ("HGST", "HGST", "HGST HDD"),
    ("HUH", "HGST", "HGST Enterprise HDD"),
    ("TOSHIBA", "Toshiba", "Toshiba HDD"),
    ("HDS", "HGST", "HGST HDD"),
    ("MTFD", "Micron", "Micron SSD"),
    ("DELLBOSS", "Dell", "Dell BOSS SSD"),
]


def normalize_drive_id(lf: pl.LazyFrame, column: str = "drive_id") -> pl.LazyFrame:
    return lf.with_columns(pl.col(column).str.strip_chars().str.to_uppercase())


def parse_capacity_gb(
    lf: pl.LazyFrame,
    *,
    bytes_column: str = "capacity_bytes",
    out_column: str = "capacity_gb",
) -> pl.LazyFrame:
    return lf.with_columns(
        (pl.col(bytes_column).cast(pl.Float64) / 1_000_000_000).alias(out_column)
    )


def infer_model_family(lf: pl.LazyFrame, *, model_column: str = "drive_model") -> pl.LazyFrame:
    """Adds `model_family` and `manufacturer` columns derived from a prefix
    match against MODEL_FAMILY_TABLE. Unmatched models get `model_family`
    equal to the raw model string and `manufacturer` of "unknown"."""
    model_family_expr = pl.col(model_column)
    manufacturer_expr = pl.lit("unknown")

    for prefix, manufacturer, family in MODEL_FAMILY_TABLE:
        is_match = pl.col(model_column).str.to_uppercase().str.starts_with(prefix)
        model_family_expr = pl.when(is_match).then(pl.lit(family)).otherwise(model_family_expr)
        manufacturer_expr = (
            pl.when(is_match).then(pl.lit(manufacturer)).otherwise(manufacturer_expr)
        )

    return lf.with_columns(
        model_family_expr.alias("model_family"),
        manufacturer_expr.alias("manufacturer"),
    )


def normalize_identifiers(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Convenience wrapper applying all identifier-normalization steps."""
    lf = normalize_drive_id(lf)
    lf = parse_capacity_gb(lf)
    lf = infer_model_family(lf)
    return lf
