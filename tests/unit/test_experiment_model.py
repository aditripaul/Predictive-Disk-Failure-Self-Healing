import numpy as np
import polars as pl
import pytest

from pipelines.experiment_model import (
    _excluded_indices,
    _sample_weights,
    _Zeroed,
    threshold_for_precision,
)
from pipelines.train_model import _build_feature_arrays, _write_id_columns


def test_zeroed_zeroes_columns_inside_the_block_and_restores_them_after():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    original = x.copy()
    with _Zeroed([x], [1, 3]):
        assert (x[:, [1, 3]] == 0).all()
        assert (x[:, [0, 2]] == original[:, [0, 2]]).all()
    assert (x == original).all()


def test_zeroed_restores_even_when_the_block_raises():
    x = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError), _Zeroed([x], [0]):
        raise RuntimeError("boom")
    assert (x == 1).all()


def test_excluded_indices_modes():
    cols = ["drive_age_days", "reallocated_sector_count", "reallocated_sector_count_7d_mean"]
    assert _excluded_indices(cols, "none") == []
    assert _excluded_indices(cols, "identity") == [0]
    assert _excluded_indices(cols, "non_windowed") == [0, 1]


def test_threshold_for_precision_picks_highest_recall_meeting_the_target():
    labels = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    assert threshold_for_precision(labels, scores, 1.0) == pytest.approx(0.8)
    assert threshold_for_precision(labels, scores, 0.75) == pytest.approx(0.6)
    assert threshold_for_precision(np.array([0, 0]), np.array([0.5, 0.4]), 0.9) is None


def test_drive_weights_make_each_failing_drive_count_equally():
    drives = np.array(["A"] * 4 + ["B"] * 1 + ["C"] * 5)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    w = _sample_weights(y, drives, "drive")
    assert w is not None
    assert w[:4].sum() == pytest.approx(w[4])  # A's 4 rows together weigh as B's 1
    assert (w[y == 0] == 1).all()
    assert _sample_weights(y, drives, "is_unbalance") is None


def test_id_columns_are_row_aligned_with_the_feature_arrays(tmp_path):
    parts = []
    for i, ids in enumerate([["a", "b"], ["c"], ["d", "e", "f"]]):
        path = tmp_path / f"part_{i}.parquet"
        pl.DataFrame(
            {"f1": [float(j) for j in range(len(ids))], "label": [0] * len(ids), "drive_id": ids}
        ).write_parquet(path)
        parts.append(path)
    x, y = _build_feature_arrays(parts, ["f1"])
    _write_id_columns(parts, ["drive_id"], tmp_path / "ids.parquet")
    ids = pl.read_parquet(tmp_path / "ids.parquet")["drive_id"].to_list()
    assert ids == ["a", "b", "c", "d", "e", "f"]
    assert len(ids) == x.shape[0] == len(y)
