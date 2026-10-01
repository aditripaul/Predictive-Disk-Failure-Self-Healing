"""Assembles a model-ready feature matrix by joining gold features with
labels for a single horizon, per docs/dataset_strategy.md section 16.
"""

from __future__ import annotations

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


def assemble_training_frame(
    gold_features: pl.DataFrame | pl.LazyFrame,
    labels: pl.DataFrame | pl.LazyFrame,
    *,
    horizon_days: int,
    label_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Joins gold features to the label table for one horizon, keeping only
    rows with an observed (non-censored) label.

    `gold_features` may be passed as a `pl.scan_parquet(...)` LazyFrame
    (`pipelines/train_model.py` does this) - the gold feature table is
    ~196 columns and ~10GB for one month of real data, but only one
    horizon's observed-label rows ever survive this join, often a small
    fraction of the full table. Reading it eagerly first (`pl.read_parquet`)
    would materialize all ~10GB before the join ever gets to discard most
    of it; scanning it lazily lets Polars push the join down instead.

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
    joined = gold_features.lazy().join(horizon_labels, on=["drive_id", "date"], how="inner")
    return joined.collect()


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
