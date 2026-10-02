"""_subsampled_train_lazy caps the primary model's training row count at
fleet scale (see pipelines/train_model.py's module docstring and
DEFAULT_MAX_TRAIN_ROWS) - these tests pin its three invariants directly
against a LazyFrame, without running the rest of the training pipeline:
every positive (failure) row survives, the total row count lands at
(approximately) the cap, and `max_rows=None` is a no-op.

_collect_split_rows and _train_sample_predicate cover the row-group-
native split extraction _stage_extract_split uses instead of a plain
scan_parquet(...).filter(...).collect() - see _collect_split_rows'
docstring for why a fully isolated, otherwise-idle subprocess still
wasn't enough on real data."""

import datetime as dt

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from pipelines.train_model import (
    _collect_split_rows,
    _subsampled_train_lazy,
    _train_sample_predicate,
)
from src.models.features import drive_batched_inner_join


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


def _split_frame(n_drives: int = 12, days: int = 10, seed: int = 0):
    """A frame.parquet-shaped fixture: drive_id/date/split/label plus one
    numeric feature column, sorted by date (like the real assembled
    frame's join output) rather than grouped by drive - so a forced
    row-group split doesn't align with split-name boundaries either."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "drive_id": f"D{d}",
            "date": dt.date(2024, 1, 1) + dt.timedelta(days=day),
            "split": ("train", "validation", "test")[d % 3],
            "label": int(rng.random() < 0.1),
            "f0": float(rng.random()),
        }
        for d in range(n_drives)
        for day in range(days)
    ]
    return pl.DataFrame(rows).sort("date")


def test_collect_split_rows_matches_a_direct_filter_across_row_group_counts(tmp_path):
    frame = _split_frame(n_drives=15, days=12, seed=1)
    direct = frame.filter(pl.col("split") == "validation").select(["f0", "label"])

    for row_group_size in (frame.height, 25, 7):
        path = tmp_path / f"frame_{row_group_size}.parquet"
        writer = pq.ParquetWriter(path, frame.to_arrow().schema)
        writer.write_table(frame.to_arrow(), row_group_size=row_group_size)
        writer.close()

        out = _collect_split_rows(
            path,
            split_name="validation",
            select_columns=["f0", "label"],
            read_columns=["f0", "label", "split"],
        )
        key = ["f0"]
        assert out.sort(key).equals(direct.sort(key)), row_group_size


def test_collect_split_rows_applies_extra_predicate(tmp_path):
    frame = _split_frame(n_drives=15, days=12, seed=2)
    path = tmp_path / "frame.parquet"
    writer = pq.ParquetWriter(path, frame.to_arrow().schema)
    writer.write_table(frame.to_arrow(), row_group_size=25)
    writer.close()

    direct = frame.filter((pl.col("split") == "train") & (pl.col("label") == 1)).select(
        ["f0", "label"]
    )
    out = _collect_split_rows(
        path,
        split_name="train",
        select_columns=["f0", "label"],
        read_columns=["f0", "label", "split"],
        extra_predicate=pl.col("label") == 1,
    )
    key = ["f0"]
    assert out.sort(key).equals(direct.sort(key))


def test_collect_split_rows_empty_result_keeps_the_selected_schema(tmp_path):
    frame = _split_frame(n_drives=6, days=5, seed=3)
    path = tmp_path / "frame.parquet"
    frame.write_parquet(path)

    out = _collect_split_rows(
        path,
        split_name="train",
        select_columns=["f0", "label"],
        read_columns=["f0", "label", "split"],
        extra_predicate=pl.lit(False),
    )
    assert out.shape == (0, 2)
    assert out.columns == ["f0", "label"]


def test_train_sample_predicate_is_none_when_max_rows_is_none_or_under_cap():
    assert _train_sample_predicate({1: 10, 0: 990}, max_rows=None, seed=0) is None
    assert _train_sample_predicate({1: 10, 0: 90}, max_rows=1_000, seed=0) is None


def test_train_sample_predicate_matches_subsampled_train_lazy():
    """_stage_extract_split applies this predicate eagerly, per row group;
    _subsampled_train_lazy applies the equivalent lazily. Same counts and
    seed must pick exactly the same rows either way, since both build the
    filter from the same label_counts."""
    frame = _labeled_frame(n_rows=20_000, positive_fraction=0.01)
    max_rows = 5_000
    lazy_result = _subsampled_train_lazy(frame.lazy(), max_rows=max_rows).collect()

    label_counts = dict(frame["label"].value_counts().iter_rows())
    predicate = _train_sample_predicate(label_counts, max_rows=max_rows, seed=0)
    assert predicate is not None
    eager_result = frame.filter(predicate)

    key = ["drive_id", "date"]
    assert lazy_result.sort(key).equals(eager_result.sort(key))


def test_finalize_batched_join_out_path_writes_one_row_group_per_batch(tmp_path):
    """Regression test for the row_group_size gap in
    src/models/features.py::_finalize_batched_join's out_path writer:
    pq.ParquetWriter.write_table's default caps a row group at 1,048,576
    rows, silently splitting any batch bigger than that into two-plus row
    groups instead of the one-row-group-per-batch this writer is meant to
    produce - the same issue already fixed once in
    pipelines/build_gold_features.py's _stage_finalize, missed here until
    pipelines/train_model.py's row-group-native split extraction relied
    on the assumption. Needs a >1,048,576-row batch to actually exercise
    pyarrow's default cap, unlike this file's other out_path tests."""
    n = 1_048_576 + 1000
    idx = pl.arange(0, n, eager=True)
    wide = pl.DataFrame({"idx": idx}).with_columns(
        ("D" + pl.col("idx").cast(pl.Utf8)).alias("drive_id"),
        pl.lit(dt.date(2024, 1, 1)).alias("date"),
        pl.col("idx").cast(pl.Float32).alias("f0"),
    )
    narrow = wide.select("drive_id", "date").with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))

    out_path = tmp_path / "joined.parquet"
    result = drive_batched_inner_join(
        wide.lazy(), narrow.lazy(), on=["drive_id", "date"], n_batches=1, out_path=out_path
    )
    assert result is None
    assert pq.ParquetFile(out_path).num_row_groups == 1
