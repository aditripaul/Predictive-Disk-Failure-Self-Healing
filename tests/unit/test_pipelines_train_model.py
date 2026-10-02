"""_subsampled_train_lazy caps the primary model's training row count at
fleet scale (see pipelines/train_model.py's module docstring and
DEFAULT_MAX_TRAIN_ROWS) - these tests pin its three invariants directly
against a LazyFrame, without running the rest of the training pipeline:
every positive (failure) row survives, the total row count lands at
(approximately) the cap, and `max_rows=None` is a no-op."""

import datetime as dt

import numpy as np
import polars as pl

from pipelines.train_model import _subsampled_train_lazy


def _labeled_frame(n_rows: int = 20_000, positive_fraction: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_rows) < positive_fraction).astype(np.int64)
    return pl.DataFrame(
        {
            "drive_id": [f"D{i}" for i in range(n_rows)],
            "date": [dt.date(2024, 1, 1) + dt.timedelta(days=i % 90) for i in range(n_rows)],
            "label": labels,
        }
    )


def test_subsampled_train_lazy_is_a_no_op_when_max_rows_is_none():
    frame = _labeled_frame()
    out = _subsampled_train_lazy(frame.lazy(), max_rows=None).collect()
    assert out.equals(frame)


def test_subsampled_train_lazy_is_a_no_op_when_already_under_the_cap():
    frame = _labeled_frame(n_rows=1_000)
    out = _subsampled_train_lazy(frame.lazy(), max_rows=10_000).collect()
    assert out.equals(frame)


def test_subsampled_train_lazy_keeps_every_positive_row():
    frame = _labeled_frame(n_rows=20_000, positive_fraction=0.02)
    n_positive = frame["label"].sum()
    out = _subsampled_train_lazy(frame.lazy(), max_rows=5_000).collect()
    assert out["label"].sum() == n_positive


def test_subsampled_train_lazy_caps_row_count_near_the_target():
    frame = _labeled_frame(n_rows=100_000, positive_fraction=0.01)
    max_rows = 10_000
    out = _subsampled_train_lazy(frame.lazy(), max_rows=max_rows).collect()
    # Row-level hash sampling is probabilistic, not an exact count - it
    # should land close to the cap, never wildly over it.
    assert max_rows * 0.8 <= out.height <= max_rows * 1.2


def test_subsampled_train_lazy_is_deterministic_across_calls():
    frame = _labeled_frame(n_rows=20_000, positive_fraction=0.01)
    first = _subsampled_train_lazy(frame.lazy(), max_rows=5_000).collect()
    second = _subsampled_train_lazy(frame.lazy(), max_rows=5_000).collect()
    assert first.equals(second)


def test_subsampled_train_lazy_never_drops_rows_when_cap_exceeds_total():
    frame = _labeled_frame(n_rows=500, positive_fraction=0.05)
    out = _subsampled_train_lazy(frame.lazy(), max_rows=1_000_000).collect()
    assert out.height == frame.height
