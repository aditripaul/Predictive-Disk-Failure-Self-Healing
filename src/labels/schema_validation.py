"""Pandera schema validation for the gold label table
(docs/project_plan.md Phase 1 Key Task "Data validation: Pandera with
Polars backend, or native Polars schema checks + pytest").
`src/preprocess/quality_checks.py` already covers the "native Polars
checks" alternative for bronze/silver leakage-adjacent invariants (no
null dates, no duplicate drive-day-attribute rows, positive capacity,
failure-after-removal ordering); this module is the Pandera half of that
either/or, applied at the gold/labels boundary right before the label
table is written and consumed by training - a schema regression here
(wrong dtype, an out-of-range split value, an impossible horizon) is
exactly the kind of bug `quality_checks.py`'s bronze/silver-focused checks
would never see.
"""

from __future__ import annotations

import polars as pl
from pandera import Check
from pandera.polars import Column, DataFrameSchema

KNOWN_SPLITS = ("train", "validation", "test", "external_smartz")
KNOWN_SPLIT_STRATEGIES = ("chronological", "drive_holdout", "vendor_holdout")


def build_gold_labels_schema(horizons_days: list[int]) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "drive_id": Column(str, nullable=False),
            "date": Column(pl.Date, nullable=False),
            "horizon_days": Column(pl.Int64, checks=Check.isin(horizons_days)),
            "label": Column(pl.Int8, nullable=True, checks=Check.isin([0, 1])),
            "event_type": Column(str, nullable=True),
            "observable_until_horizon": Column(pl.Boolean, nullable=False),
            "split": Column(str, checks=Check.isin(list(KNOWN_SPLITS))),
            "split_version": Column(str, nullable=False),
            "split_strategy": Column(str, checks=Check.isin(list(KNOWN_SPLIT_STRATEGIES))),
        },
        strict=False,  # extra columns (e.g. source_dataset) are expected and fine
    )


def validate_gold_labels(labels: pl.DataFrame, *, horizons_days: list[int]) -> pl.DataFrame:
    """Validates `labels` against `build_gold_labels_schema`, raising
    `pandera.errors.SchemaErrors` (with every failing row/check, not just
    the first) if it doesn't conform. Returns `labels` unchanged on
    success, for a fluent `labels = validate_gold_labels(labels, ...)`
    call site."""
    schema = build_gold_labels_schema(horizons_days)
    schema.validate(labels, lazy=True)
    return labels
