import datetime as dt

import polars as pl

from src.labels.event_types import classify_event_types
from src.labels.imbalance import class_distribution_report, compute_scale_pos_weight
from src.labels.labeling import compute_labels_for_horizon
from src.labels.splits import (
    add_chronological_split,
    apply_drive_level_holdout,
    apply_vendor_holdout,
)


def _metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": ["failing", "healthy", "removed", "active"],
            "last_seen_date": [
                dt.date(2024, 1, 20),
                dt.date(2024, 3, 1),
                dt.date(2024, 1, 10),
                dt.date(2024, 3, 1),
            ],
            "failure_date": [dt.date(2024, 1, 20), None, None, None],
            "removal_date": [None, None, dt.date(2024, 1, 10), None],
        }
    )


def test_classify_event_types():
    metadata = classify_event_types(_metadata(), as_of_date=dt.date(2024, 3, 1))
    by_id = {row["drive_id"]: row for row in metadata.to_dicts()}
    assert by_id["failing"]["event_type"] == "confirmed_failure"
    # removal_date present, no failure_date recorded -> preventive replacement
    assert by_id["removed"]["event_type"] == "preventive_replacement"
    assert by_id["active"]["event_type"] == "still_active"
    assert by_id["healthy"]["event_type"] == "still_active"


def test_compute_labels_for_horizon_positive_negative_and_censored():
    metadata = classify_event_types(_metadata(), as_of_date=dt.date(2024, 3, 1))
    drive_days = pl.DataFrame(
        {
            "drive_id": ["failing", "healthy", "removed"],
            "date": [dt.date(2024, 1, 10), dt.date(2024, 1, 10), dt.date(2024, 1, 5)],
        }
    )
    labels = compute_labels_for_horizon(drive_days, metadata, horizon_days=14)
    by_id = {row["drive_id"]: row for row in labels.to_dicts()}

    # failure_date 2024-01-20 is within (2024-01-10, 2024-01-24] -> positive
    assert by_id["failing"]["label"] == 1
    # healthy drive observed through 2024-03-01, well past 2024-01-24 -> negative
    assert by_id["healthy"]["label"] == 0
    # removed_without_failure within horizon (2024-01-10 in (2024-01-05, 2024-01-19]) -> censored
    assert by_id["removed"]["label"] is None


def test_add_chronological_split_and_drive_holdout():
    df = pl.DataFrame(
        {
            "drive_id": [f"d{i}" for i in range(20)],
            "date": [dt.date(2022, 1, 1)] * 20,
        }
    )
    df = add_chronological_split(df, train_end="2022-12-31", validation_end="2023-12-31")
    assert set(df["split"].unique()) == {"train"}

    holdout = apply_drive_level_holdout(df, holdout_fraction=0.2, seed=1)
    assert "validation" in holdout["split"].unique()
    assert (holdout["split"] == "validation").sum() == 4


def test_apply_vendor_holdout_reassigns_smartz_rows_regardless_of_chronological_split():
    df = pl.DataFrame(
        {
            "drive_id": ["bb1", "bb2", "sz1", "sz2"],
            "date": [dt.date(2022, 1, 1)] * 4,
            "source_dataset": ["backblaze", "backblaze", "smartz", "smartz"],
        }
    )
    df = add_chronological_split(df, train_end="2022-12-31", validation_end="2023-12-31")
    assert set(df["split"].unique()) == {"train"}

    out = apply_vendor_holdout(df)
    by_id = {row["drive_id"]: row for row in out.to_dicts()}
    assert by_id["bb1"]["split"] == "train"
    assert by_id["bb1"]["split_strategy"] == "chronological"
    assert by_id["sz1"]["split"] == "external_smartz"
    assert by_id["sz1"]["split_strategy"] == "vendor_holdout"
    assert by_id["sz2"]["split"] == "external_smartz"


def test_apply_vendor_holdout_is_a_noop_without_source_column():
    df = pl.DataFrame({"drive_id": ["a"], "split": ["train"], "split_strategy": ["chronological"]})
    out = apply_vendor_holdout(df)
    assert out["split"].to_list() == ["train"]


def test_imbalance_reporting():
    labels = pl.DataFrame(
        {
            "horizon_days": [14, 14, 14, 14],
            "split": ["train", "train", "train", "train"],
            "label": [1, 0, 0, None],
        }
    )
    weight = compute_scale_pos_weight(labels)
    assert weight == 2 / 1

    report = class_distribution_report(labels)
    assert report[0]["positive_count"] == 1
    assert report[0]["negative_count"] == 2
    assert report[0]["censored_count"] == 1
