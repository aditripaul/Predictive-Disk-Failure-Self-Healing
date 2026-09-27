import datetime as dt

import polars as pl

from src.reliability.analytics import failure_rate_by_model_family


def test_failure_rate_by_model_family_computes_per_family_rate(tmp_path):
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"

    pl.DataFrame(
        {
            "drive_id": ["A", "B", "C", "D"],
            "date": [dt.date(2024, 1, 1)] * 4,
            "model_family": ["Seagate HDD", "Seagate HDD", "WD HDD", "WD HDD"],
        }
    ).write_parquet(features_path)

    pl.DataFrame(
        {
            "drive_id": ["A", "B", "C", "D"],
            "date": [dt.date(2024, 1, 1)] * 4,
            "horizon_days": [14, 14, 14, 14],
            "label": [1, 0, 0, 0],
        }
    ).write_parquet(labels_path)

    rows = failure_rate_by_model_family(features_path, labels_path, horizon_days=14)
    by_family = {row["model_family"]: row for row in rows}

    assert by_family["Seagate HDD"]["failure_rate"] == 0.5
    assert by_family["Seagate HDD"]["drive_day_count"] == 2
    assert by_family["WD HDD"]["failure_rate"] == 0.0


def test_failure_rate_by_model_family_excludes_censored_and_wrong_horizon_rows(tmp_path):
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"

    pl.DataFrame(
        {
            "drive_id": ["A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "model_family": ["Seagate HDD", "Seagate HDD"],
        }
    ).write_parquet(features_path)

    pl.DataFrame(
        {
            "drive_id": ["A", "A"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "horizon_days": [14, 30],
            "label": [None, 1],
        }
    ).write_parquet(labels_path)

    rows = failure_rate_by_model_family(features_path, labels_path, horizon_days=14)
    assert rows == []  # the only horizon=14 row is censored (label is null)


def test_failure_rate_by_model_family_returns_empty_when_files_missing(tmp_path):
    rows = failure_rate_by_model_family(
        tmp_path / "missing_features.parquet", tmp_path / "missing_labels.parquet", horizon_days=14
    )
    assert rows == []
