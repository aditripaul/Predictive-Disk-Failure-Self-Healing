import datetime as dt

import numpy as np
import polars as pl

from src.features.sequences import (
    apply_time_decay_weights,
    build_sequence_tensors,
    compute_time_decay_weights,
    load_sequence_tensors,
    write_sequence_tensors,
)


def _wide_features_two_drives() -> pl.DataFrame:
    dates_a = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    dates_b = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    return pl.DataFrame(
        {
            "drive_id": ["A"] * 40 + ["B"] * 5,
            "date": dates_a + dates_b,
            "reallocated_sector_count": list(range(40)) + list(range(5)),
            "power_on_hours": [x * 24 for x in range(40)] + [x * 24 for x in range(5)],
        }
    )


def test_build_sequence_tensors_shape_and_attribute_filtering():
    df = _wide_features_two_drives()
    tensor, attrs_used, drive_ids, as_of_dates = build_sequence_tensors(
        df,
        attributes=("reallocated_sector_count", "power_on_hours", "does_not_exist"),
        time_steps=30,
    )
    assert attrs_used == ["reallocated_sector_count", "power_on_hours"]
    assert tensor.shape == (2, 30, 2)
    assert set(drive_ids) == {"A", "B"}
    assert len(as_of_dates) == 2


def test_build_sequence_tensors_uses_last_time_steps_days_for_a_long_history_drive():
    df = _wide_features_two_drives()
    tensor, _, drive_ids, as_of_dates = build_sequence_tensors(
        df, attributes=("reallocated_sector_count",), time_steps=30
    )
    a_idx = drive_ids.index("A")
    # drive A has 40 days (values 0..39); the last 30 days are values 10..39
    assert tensor[a_idx, 0, 0] == 10.0
    assert tensor[a_idx, -1, 0] == 39.0
    assert as_of_dates[a_idx] == dt.date(2024, 1, 1) + dt.timedelta(days=39)


def test_build_sequence_tensors_left_pads_short_history_drives_with_zeros():
    df = _wide_features_two_drives()
    tensor, _, drive_ids, _ = build_sequence_tensors(
        df, attributes=("reallocated_sector_count",), time_steps=30
    )
    b_idx = drive_ids.index("B")
    # drive B only has 5 real days (values 0..4); the other 25 are zero-padding
    assert np.all(tensor[b_idx, :25, 0] == 0.0)
    assert tensor[b_idx, 25:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_build_sequence_tensors_raises_when_no_attributes_are_present():
    df = _wide_features_two_drives()
    try:
        build_sequence_tensors(df, attributes=("nonexistent",), time_steps=30)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "nonexistent" in str(exc)


def test_compute_time_decay_weights_is_increasing_toward_the_present():
    weights = compute_time_decay_weights(10, lambda_=0.1)
    assert weights.shape == (10,)
    assert weights[-1] == 1.0  # most recent day: age 0
    assert np.all(np.diff(weights) > 0)  # strictly increasing toward the present


def test_apply_time_decay_weights_scales_each_time_step():
    tensor = np.ones((2, 3, 2))
    weights = np.array([0.1, 0.5, 1.0])
    weighted = apply_time_decay_weights(tensor, weights)
    assert np.allclose(weighted[:, 0, :], 0.1)
    assert np.allclose(weighted[:, 1, :], 0.5)
    assert np.allclose(weighted[:, 2, :], 1.0)


def test_apply_time_decay_weights_rejects_mismatched_length():
    tensor = np.ones((2, 3, 2))
    weights = np.array([0.1, 0.5])
    try:
        apply_time_decay_weights(tensor, weights)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "time_steps" in str(exc)


def test_write_and_load_sequence_tensors_round_trip(tmp_path):
    df = _wide_features_two_drives()
    tensor, attrs_used, drive_ids, as_of_dates = build_sequence_tensors(
        df, attributes=("reallocated_sector_count", "power_on_hours"), time_steps=30
    )
    write_sequence_tensors(
        tensor,
        attributes_used=attrs_used,
        drive_ids=drive_ids,
        as_of_dates=as_of_dates,
        out_dir=tmp_path,
    )

    loaded_tensor, metadata, loaded_attrs = load_sequence_tensors(tmp_path)
    assert np.array_equal(np.asarray(loaded_tensor), tensor)
    assert loaded_attrs == attrs_used
    assert set(metadata["drive_id"]) == set(drive_ids)
