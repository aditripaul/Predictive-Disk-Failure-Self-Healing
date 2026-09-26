import datetime as dt

import polars as pl

from src.labels.dataset_version import (
    build_dataset_version_record,
    latest_dataset_version,
    write_dataset_version,
)


def _labels_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
            "split": ["train", "train", "validation"],
        }
    )


def test_build_dataset_version_record_captures_counts_and_hash(tmp_path):
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    features_path.write_bytes(b"features-bytes")
    labels_path.write_bytes(b"labels-bytes")

    record = build_dataset_version_record(
        features_path=features_path,
        labels=_labels_df(),
        labels_path=labels_path,
        feature_registry_version=1,
        model_config_version=1,
    )

    assert record["gold_labels"]["row_count"] == 3
    assert record["gold_labels"]["date_range"] == ["2024-01-01", "2024-01-02"]
    assert record["gold_features"]["content_hash"] is not None
    assert record["gold_labels"]["content_hash"] is not None
    split_counts = {row["split"]: row["len"] for row in record["gold_labels"]["split_counts"]}
    assert split_counts == {"train": 2, "validation": 1}


def test_build_dataset_version_record_handles_missing_files(tmp_path):
    record = build_dataset_version_record(
        features_path=tmp_path / "missing_features.parquet",
        labels=_labels_df(),
        labels_path=tmp_path / "missing_labels.parquet",
        feature_registry_version=1,
        model_config_version=1,
    )
    assert record["gold_features"]["content_hash"] is None
    assert record["gold_labels"]["content_hash"] is None


def test_write_and_read_latest_dataset_version(tmp_path):
    audit_dir = tmp_path / "audit"
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    features_path.write_bytes(b"f")
    labels_path.write_bytes(b"l")

    record = build_dataset_version_record(
        features_path=features_path,
        labels=_labels_df(),
        labels_path=labels_path,
        feature_registry_version=2,
        model_config_version=3,
    )
    written_path = write_dataset_version(record, audit_dir)
    assert written_path.exists()

    latest = latest_dataset_version(audit_dir)
    assert latest is not None
    assert latest["version_id"] == record["version_id"]
    assert latest["feature_registry_version"] == 2


def test_latest_dataset_version_is_none_when_no_versions_exist(tmp_path):
    assert latest_dataset_version(tmp_path / "audit") is None
