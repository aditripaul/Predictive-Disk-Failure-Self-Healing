import datetime as dt

import polars as pl
import pytest
from pandera.errors import SchemaErrors

from src.labels.schema_validation import validate_gold_labels


def _valid_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "horizon_days": [14, 14],
            "label": pl.Series([0, 1], dtype=pl.Int8),
            "event_type": [None, "confirmed_failure"],
            "observable_until_horizon": [True, True],
            "split": ["train", "test"],
            "split_version": ["v1.0", "v1.0"],
            "split_strategy": ["chronological", "chronological"],
        }
    )


def test_validate_gold_labels_passes_a_conforming_frame():
    labels = _valid_labels()
    out = validate_gold_labels(labels, horizons_days=[14])
    assert out.height == 2


def test_validate_gold_labels_allows_extra_columns():
    labels = _valid_labels().with_columns(pl.lit("backblaze").alias("source_dataset"))
    out = validate_gold_labels(labels, horizons_days=[14])
    assert "source_dataset" in out.columns


def test_validate_gold_labels_rejects_unknown_split_value():
    labels = _valid_labels().with_columns(pl.Series("split", ["train", "not_a_real_split"]))
    with pytest.raises(SchemaErrors):
        validate_gold_labels(labels, horizons_days=[14])


def test_validate_gold_labels_rejects_out_of_range_label():
    labels = _valid_labels().with_columns(pl.Series("label", [0, 2], dtype=pl.Int8))
    with pytest.raises(SchemaErrors):
        validate_gold_labels(labels, horizons_days=[14])


def test_validate_gold_labels_rejects_null_drive_id():
    labels = _valid_labels().with_columns(pl.Series("drive_id", ["A", None]))
    with pytest.raises(SchemaErrors):
        validate_gold_labels(labels, horizons_days=[14])


def test_validate_gold_labels_rejects_unconfigured_horizon():
    labels = _valid_labels()
    with pytest.raises(SchemaErrors):
        validate_gold_labels(labels, horizons_days=[7, 30])  # 14 not in the configured set
