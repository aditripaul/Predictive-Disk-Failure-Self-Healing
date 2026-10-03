import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from pipelines.spike_duckdb_extract import (
    _sample_predicate,
    compare_with_polars,
    extract_train_duckdb,
)


@pytest.fixture
def frame(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    f1 = [None if rng.random() < 0.1 else float(v) for v in rng.normal(size=n)]  # real nulls
    df = pl.DataFrame(
        {
            "drive_id": [f"d{i % 25}" for i in range(n)],
            "date": [dt.date(2026, 1, 1) + dt.timedelta(days=i // 25) for i in range(n)],
            "split": ["train" if i < 300 else "validation" for i in range(n)],
            "label": [1 if i % 37 == 0 else 0 for i in range(n)],
            "f1": f1,
            "f2": rng.integers(0, 5, size=n).astype(float),
        }
    )
    path = tmp_path / "frame.parquet"
    df.write_parquet(path)
    (tmp_path / "frame_meta.json").write_text(json.dumps({"feature_columns": ["f1", "f2"]}))
    return tmp_path


def test_duckdb_extract_matches_polars_exactly_on_the_uncapped_train_split(frame):
    out_dir = frame
    extract_train_duckdb(
        frame / "frame.parquet",
        ["f1", "f2"],
        out_dir,
        memory_limit="256MB",
        temp_dir=frame / "spill",
        max_train_rows=None,
    )
    report = compare_with_polars(frame / "frame.parquet", ["f1", "f2"], out_dir)
    assert report["same_shape"]
    assert report["features_identical"]
    assert report["labels_identical"]
    assert report["reference_shape"] == [300, 2]


def test_capped_extract_keeps_all_positives_and_caps_negatives(frame):
    stats = extract_train_duckdb(
        frame / "frame.parquet",
        ["f1", "f2"],
        frame,
        memory_limit="256MB",
        temp_dir=frame / "spill",
        max_train_rows=120,
    )
    x = np.load(frame / "x_train.duckdb.npy")
    y = np.load(frame / "y_train.duckdb.npy")
    assert stats["label_counts"] == {"0": 291, "1": 9}
    assert int(y.sum()) == 9  # every positive kept
    assert x.shape[0] == len(y) < 300  # negatives sampled down


def test_sample_predicate_is_none_when_under_the_cap():
    assert _sample_predicate({0: 10, 1: 2}, max_rows=100) is None
    assert _sample_predicate({0: 10, 1: 2}, max_rows=None) is None
