"""Assembles a model-ready feature matrix by joining gold features with
labels for a single horizon, per docs/dataset_strategy.md section 16.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

#: Dtypes a gold feature column may have to be usable as a model feature.
NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
    pl.Boolean,
)

NON_FEATURE_COLUMNS = {
    "drive_id",
    "date",
    "horizon_days",
    "label",
    "event_date",
    "event_type",
    "observable_until_horizon",
    "days_to_event",
    "split",
    "split_version",
    "split_strategy",
}


def chunked_inner_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    on: list[str],
    chunk_rows: int,
) -> pl.DataFrame:
    """Inner-joins `left` to `right` in row slices of `left` rather than
    all at once, concatenating the results.

    For a wide `scan_parquet` left side there is often nothing for the
    query engine to prune - every column is wanted, and the rows to keep
    are defined by the join rather than by a filter - so an unsliced join
    materializes the entire left table alongside its own output and the
    join's hash table. Slicing bounds that by the slice size instead.

    Only valid when the join is row-wise: each left row must match at most
    one right row, and nothing may depend on a left row's neighbours.
    That is the caller's responsibility.

    Row order matches the unsliced join - slices are taken and
    concatenated in order - which matters for callers that later select
    rows by index.

    Each slice's joined result is spilled to its own temp Parquet file
    and freed before the next slice is read, rather than accumulated in a
    Python list - keeping every slice's result alive in a list until one
    final `pl.concat` just moves the "hold everything at once" problem
    this function exists to avoid down one level (the same growing-
    accumulator shape fixed in src/features/windows.py and
    src/features/events.py earlier), and a `del`/`gc.collect()` inside the
    loop cannot free a chunk that a list still holds a reference to. The
    final concat below instead reads the spilled chunks back lazily, so
    at most one chunk's worth of data needs to be resident at a time."""
    left_lazy = left.lazy()
    # Collected once: left as a LazyFrame would otherwise re-run the
    # right-hand plan (often a scan plus filter) for every slice.
    right_eager = right.lazy().collect().lazy()

    total_rows = left_lazy.select(pl.len()).collect().item()
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_paths = []
        for offset in range(0, total_rows, chunk_rows):
            part = (
                left_lazy.slice(offset, chunk_rows)
                .join(right_eager, on=on, how="inner")
                .collect()
            )
            if part.height:
                path = Path(tmp_dir) / f"chunk_{offset}.parquet"
                part.write_parquet(path, compression="zstd")
                chunk_paths.append(path)
            del part
            gc.collect()

        if not chunk_paths:
            # Preserve the joined schema rather than returning something a
            # caller's `.height == 0`/column check can't introspect.
            return left_lazy.slice(0, 0).join(right_eager, on=on, how="inner").collect()
        return pl.concat([pl.scan_parquet(p) for p in chunk_paths], how="vertical").collect()


def assemble_training_frame(
    gold_features: pl.DataFrame | pl.LazyFrame,
    labels: pl.DataFrame | pl.LazyFrame,
    *,
    horizon_days: int,
    label_columns: list[str] | None = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame:
    """Joins gold features to the label table for one horizon, keeping only
    rows with an observed (non-censored) label.

    `gold_features` may be passed as a `pl.scan_parquet(...)` LazyFrame
    (`pipelines/train_model.py` does this), and `chunk_rows` then joins it
    in row-slices of that size instead of all at once.

    Scanning lazily is not by itself enough to keep the gold feature table
    out of memory: every one of its ~196 columns is needed as a model
    feature and there is no predicate to push into the scan, so the
    default engine has nothing to prune and materializes the whole ~10GB
    table (one month of real data) to join it - confirmed against real
    data, where this join hit the memory cap with the full table plus the
    join output plus the join's hash table all resident.

    Slicing by row offset is safe here because nothing in this step is
    per-drive: the join matches each gold row to at most one label row on
    (drive_id, date), and the split assignment comes from the label table
    rather than from any computation over a drive's history. So a row
    slice can be joined independently and the results concatenated, and
    each slice's share of the gold table is freed before the next is read.

    Deliberately plain `.collect()`, not `.collect(engine="streaming")`:
    against real data, the streaming engine turned this exact join (a
    wide `scan_parquet` LazyFrame joined to a small in-memory-then-
    `.lazy()`'d one) into a request to allocate ~38TiB - a nonsensical
    size for row counts that were themselves entirely sane (labels was
    31,389,810 rows, exactly 3x drive-days as designed; no fanout). That
    failure mode never appeared anywhere else tonight, where every other
    lazy scan+join+collect in this codebase (src/features/pivot.py,
    pipelines/build_gold_features.py's finalize stage) uses plain
    `.collect()` and has run correctly against the same real data. That
    strongly points at a streaming-engine limitation for this join shape
    rather than a genuine memory need, so this stays on the default
    engine unless a real, scale-appropriate allocation failure shows up
    here instead.

    `labels` may likewise be a LazyFrame, and `label_columns` narrows it
    to just the columns the caller actually needs before the join. The
    label table is one row per drive-day *per horizon*, so it is ~3x the
    drive-day count (31.4M rows, ~5.6GB, for one month of real data)
    while only one horizon and a handful of its 12 columns are ever used
    downstream. Defaults to `None` (keep every label column) so existing
    callers are unaffected."""
    horizon_labels = labels.lazy().filter(
        (pl.col("horizon_days") == horizon_days) & pl.col("label").is_not_null()
    )
    if label_columns is not None:
        horizon_labels = horizon_labels.select(label_columns)

    if chunk_rows is None:
        return gold_features.lazy().join(
            horizon_labels, on=["drive_id", "date"], how="inner"
        ).collect()

    return chunked_inner_join(
        gold_features, horizon_labels, on=["drive_id", "date"], chunk_rows=chunk_rows
    )


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric feature columns only: excludes ids, dates, label/split
    metadata, and non-numeric columns (e.g. model_family strings)."""
    return [
        c
        for c, dtype in zip(df.columns, df.dtypes, strict=True)
        if c not in NON_FEATURE_COLUMNS and dtype in NUMERIC_DTYPES
    ]


def feature_matrix(df: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """The model input matrix for `df`, as float32 rather than the float64
    a plain `.to_numpy()` would produce.

    `.to_numpy()` on a frame mixing Float32/UInt32/Int64/Boolean columns
    upcasts everything to float64, which doubles the matrix for no benefit:
    146 of the ~196 gold feature columns are already Float32 on disk, the
    UInt32 ones are window day-counts (<= 30, exactly representable), and
    float32's ~7 significant digits are far more precision than the
    remaining ratio/count features carry. At fleet scale that difference
    is ~8.9GB vs ~4.5GB for one month of real data's training split.
    The tree models this feeds also bin every feature into at most
    `max_bin` (255 by default) histogram buckets internally, so the
    discarded mantissa bits cannot change a split point."""
    return (
        df.select([pl.col(c).cast(pl.Float32) for c in feature_columns])
        .fill_null(0.0)
        .to_numpy()
    )
