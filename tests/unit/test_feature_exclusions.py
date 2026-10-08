import polars as pl

from pipelines.train_model import _rate
from src.models.features import select_feature_columns


def test_raw_failure_flag_is_never_a_feature():
    frame = pl.DataFrame(
        {
            "drive_id": ["a", "b"],
            "failure": [0, 1],
            "smart_5_raw": [1.0, 2.0],
            "label": [0, 1],
        }
    )
    columns = select_feature_columns(frame)
    assert "failure" not in columns
    assert "label" not in columns
    assert columns == ["smart_5_raw"]


def test_evaluated_rate_uses_recorded_counts():
    assert _rate({"row_count": 200, "positive_count": 5}) == 0.025


def test_evaluated_rate_is_none_for_old_work_dirs():
    assert _rate({"row_count": 200}) is None
    assert _rate({"row_count": 0, "positive_count": 0}) is None


def test_lift_targets_convert_to_precision_at_the_tuning_rate():
    from src.models.threshold import precision_targets_from_lift

    targets = precision_targets_from_lift({"warn": 56.6, "drain": 226.4}, 0.0026498633930167304)
    assert abs(targets["warn"] - 0.15) < 1e-3
    assert abs(targets["drain"] - 0.60) < 1e-3
    assert precision_targets_from_lift({"x": 1e6}, 0.5) == {"x": 1.0}
