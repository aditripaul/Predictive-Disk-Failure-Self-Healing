"""Assembles a model-ready feature matrix by joining gold features with
labels for a single horizon, per docs/dataset_strategy.md section 16.
"""

from __future__ import annotations

import gc
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.logging_config import get_logger

logger = get_logger(__name__)

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
    # Raw Backblaze failure flag: marks the failure day itself, so it is a
    # label in disguise. All zeros in the current frame; excluded so it can
    # never become a model input.
    "failure",
}


def _finalize_batched_join(
    part_paths: list[Path],
    empty_result: pl.DataFrame,
    *,
    out_path: Path | None,
) -> pl.DataFrame | None:
    """Combines a batched join's already-spilled part files into the
    final result.

    When `out_path` is given, writes it there incrementally via a
    Parquet row-group writer, one part per row group, and returns
    `None` - the joined result is NEVER held as a single eager
    DataFrame. This is required once that result can be tens of
    millions of rows: `pipelines/train_model.py`'s training-frame join
    got every earlier fix right (bounding the wide table's own read,
    bounding each batch's join) and still crashed, because this final
    step used to do `pl.concat([...]).collect()` - reassembling every
    batch back into one eager ~26-30M-row DataFrame right after the
    batching had successfully avoided ever materializing anything
    that large. Since the caller (`_stage_assemble`) was just going to
    `.write_parquet()` that DataFrame immediately anyway, writing each
    batch straight to the destination - the same incremental-write
    pattern `pipelines/build_gold_features.py`'s finalize stage already
    uses - reaches the same end state without the pointless full
    materialization in between.

    When `out_path` is `None` (the default), concatenates the parts
    into one eager DataFrame and returns it instead - correct and
    proportionate for a genuinely small result, e.g.
    `src/models/serving.py::latest_row_per_drive`'s one-row-per-drive
    join, which a caller may need to keep working with in memory rather
    than just writing to disk.

    `empty_result` (an eager, correctly-shaped but zero-row DataFrame)
    covers the case where `part_paths` is empty - either returned
    directly, or written as a schema-correct empty file, so a caller's
    `.height == 0` check or the written file's schema is well-formed
    either way."""
    if out_path is not None:
        writer = None
        try:
            for path in part_paths:
                part = pl.read_parquet(path)
                table = part.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                # row_group_size=table.num_rows: pq.ParquetWriter.write_table's
                # default caps a row group at 1,048,576 rows, silently
                # splitting any part bigger than that into two-plus row
                # groups instead of the one-row-group-per-part this writer
                # is meant to produce (the same issue already fixed in
                # pipelines/build_gold_features.py's _stage_finalize, for
                # the same reason: downstream readers that iterate this
                # file's row groups one at a time - e.g.
                # join_by_native_row_groups, pipelines/train_model.py's
                # split extraction - assume that layout).
                writer.write_table(table, row_group_size=table.num_rows)
                del part, table
                gc.collect()
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            empty_result.write_parquet(out_path, compression="zstd")
        return None

    if not part_paths:
        return empty_result
    return pl.concat([pl.scan_parquet(p) for p in part_paths], how="vertical").collect()


def chunked_inner_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    on: list[str],
    chunk_rows: int,
    out_path: Path | None = None,
) -> pl.DataFrame | None:
    """Inner-joins `left` to `right` in row slices of `left` rather than
    all at once, concatenating the results (or, if `out_path` is given,
    writing them there directly instead - see `_finalize_batched_join`).

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
    at most one chunk's worth of data needs to be resident at a time.

    `right` is kept lazy and its (scan+filter+select) plan is re-run for
    every slice, rather than collected once up front and reused - that
    used to be the design here, on the assumption that the right side of
    a training-frame join (one horizon's observed-label rows) would
    always be small next to the wide left side. Against real Q1 data
    that assumption broke: with a 91-day quarter, most rows are far
    enough from the end of the ingested range to be observable at a
    14-day horizon, so the "small" side turned out to be ~26M rows -
    close to the ~30.6M-row left side - and collecting it once meant
    holding nearly as much data resident for the ENTIRE loop as slicing
    the left side was meant to avoid holding in the first place.
    Re-running `right`'s plan per slice costs some redundant scan+filter
    work (bounded, and pushed into the Parquet scan itself), in exchange
    for never holding more of `right` than the current slice's join
    actually touches - the same "redundant rescan, bounded memory" trade
    already made for Bronze in pipelines/build_silver.py's batch stage."""
    left_lazy = left.lazy()
    right_lazy = right.lazy()

    total_rows = left_lazy.select(pl.len()).collect().item()
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_paths = []
        for offset in range(0, total_rows, chunk_rows):
            part = (
                left_lazy.slice(offset, chunk_rows).join(right_lazy, on=on, how="inner").collect()
            )
            if part.height:
                path = Path(tmp_dir) / f"chunk_{offset}.parquet"
                part.write_parquet(path, compression="zstd")
                chunk_paths.append(path)
            del part
            gc.collect()

        # Preserve the joined schema rather than something a caller's
        # `.height == 0`/column check (or the written file, if `out_path`
        # is given) can't introspect.
        empty_result = left_lazy.slice(0, 0).join(right_lazy, on=on, how="inner").collect()
        return _finalize_batched_join(chunk_paths, empty_result, out_path=out_path)


def drive_batched_inner_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    on: list[str],
    n_batches: int,
    out_path: Path | None = None,
) -> pl.DataFrame | None:
    """Inner-joins `left` to `right` in batches of whole drives - both
    sides filtered to the SAME `hash(drive_id) % n_batches` bucket before
    joining, rather than slicing only `left` by row offset (see
    `chunked_inner_join`, and why it's the wrong tool when `right` isn't
    small).

    Row-offset slicing only bounds one side of the join. For every slice
    of `left`, the query engine still has to execute `right`'s ENTIRE
    plan to build the join's hash table - any drive's matching row could
    be anywhere in `right`, so nothing about a left row-offset lets it
    prune `right` at all. That's fine when `right` is genuinely small
    (build once, probe cheaply many times), but is no better than an
    unsliced join when `right` is comparably large: real Q1 data hit
    this directly in pipelines/train_model.py's training-frame join,
    where the label table's "one horizon, observed only" subset -
    assumed small when this was built against January-only data, where
    most rows were too close to the end of ingest to be observable -
    turned out to be ~26M rows, close to the ~30.6M-row gold features
    table, once a full quarter's observability made most rows keepable.

    Filtering BOTH sides to the same drive-hash bucket bounds both sides
    at once: batch `i`'s join is between two genuinely small slices
    (~1/n_batches of each table), not one small slice against a
    still-full-size complement. This is the same batch-by-drive shape
    already proven in pipelines/build_gold_features.py, build_silver.py,
    and build_labels.py, applied here to a join between two large tables
    instead of a single-table transform.

    Only valid when `on` includes "drive_id" (so hashing it is a
    meaningful partition key for both sides) and the join is row-wise -
    each left row matches at most one right row, so a drive's rows can
    be judged independently of every other drive's, exactly as required
    by every other batch-by-drive transform in this codebase.

    Row order is grouped by batch (all of batch 0's matched rows, then
    batch 1's, ...), not global - unlike `chunked_inner_join`, which
    preserves `left`'s row order because it only reindexes `left`. This
    does not affect correctness for callers here: nothing downstream of
    this join depends on row position for anything except reproducible
    random sampling (SHAP background, diagnostic row caps), and those
    stay reproducible - same code, same fixed seeds, same resulting
    sample - just not IDENTICAL to a row-offset-chunked join's sample,
    since the physical row order genuinely changed. The set and content
    of joined rows is unaffected either way.

    Each side's batch is `.collect()`-ed on its OWN, separately, before
    the join - the join itself is then a plain eager `DataFrame.join()`
    between two already-materialized frames, never a single combined
    lazy plan with both filters and the join collected together in one
    `.collect()` call. That matters, not just for style: against real Q1
    data, an earlier version of this function built exactly that
    combined plan (`left.filter(...).join(right.filter(...), ...)
    .collect()`) and it failed with a request to allocate
    ~354,295,353,634,353,658 bytes - not a real memory need at any
    conceivable scale, but the signature of a size-computation bug, the
    same category of failure `assemble_training_frame` hit once already
    tonight with `engine="streaming"`. Every OTHER batch-join in this
    codebase (e.g. `pivot_badness_wide`'s drive_day join-back, called
    from `pipelines/build_gold_features.py`'s per-batch pivot stage)
    already collects each side separately first and joins two eager
    frames - this was the one place that instead asked the query
    optimizer to fuse two hash-filters and a join into one lazy plan,
    which is the only thing that changed between "works" and that
    ~354-quintillion-byte failure.

    Pass `out_path` to write the combined result directly there instead
    of returning it as one eager DataFrame - see
    `_finalize_batched_join`. Required once the joined result can be
    tens of millions of rows (the training frame at fleet scale): even
    after every batch's own join is bounded, concatenating all of them
    back into one eager DataFrame at the end reintroduces the exact
    "materialize the whole thing" problem the batching was meant to
    avoid."""
    left_lazy = left.lazy()
    right_lazy = right.lazy()
    batch_of_drive = pl.col("drive_id").hash(seed=0) % n_batches

    with tempfile.TemporaryDirectory() as tmp_dir:
        batch_paths = []
        for batch_index in range(n_batches):
            left_batch = left_lazy.filter(batch_of_drive == batch_index).collect()
            right_batch = right_lazy.filter(batch_of_drive == batch_index).collect()
            part = left_batch.join(right_batch, on=on, how="inner")
            if part.height:
                path = Path(tmp_dir) / f"batch_{batch_index}.parquet"
                part.write_parquet(path, compression="zstd")
                batch_paths.append(path)
            del left_batch, right_batch, part
            gc.collect()

        # Preserve the joined schema rather than something a caller's
        # `.height == 0`/column check (or the written file, if `out_path`
        # is given) can't introspect.
        empty_left = left_lazy.filter(pl.lit(False)).collect()
        empty_right = right_lazy.filter(pl.lit(False)).collect()
        empty_result = empty_left.join(empty_right, on=on, how="inner")
        return _finalize_batched_join(batch_paths, empty_result, out_path=out_path)


def _as_dataframe(value: object, *, context: str) -> pl.DataFrame:
    """`pl.from_arrow(...)` is typed to return a DataFrame or Series
    depending on its input; a Parquet row group (or an Arrow Table more
    generally) always converts to a DataFrame, but that's a runtime fact
    about the input, not something the type checker can see. Raising
    explicitly (rather than `assert`, which vanishes under `-O`/
    `PYTHONOPTIMIZE=1`) turns a violated assumption into a clear error
    here instead of a confusing AttributeError several lines later."""
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{context}: expected a DataFrame from pl.from_arrow, got {type(value)}")
    return value


def join_by_native_row_groups(
    wide_path: Path,
    narrow: pl.DataFrame | pl.LazyFrame,
    *,
    on: list[str],
    max_rows_per_join: int | None = None,
    out_path: Path | None = None,
) -> pl.DataFrame | None:
    """Inner-joins the Parquet file at `wide_path` to `narrow`, reading
    `wide_path` by its OWN existing row groups rather than re-deriving
    batches with a fresh filter.

    This exists because `drive_batched_inner_join`'s `hash(drive_id) %
    n_batches` filter is not something Parquet row-group statistics can
    use for pruning - row groups aren't laid out by hash bucket - so
    evaluating it means scanning and decoding EVERY column of EVERY row
    of `wide_path` to find each batch's ~1/n_batches share, once per
    batch. For a ~196-column, ~10GB (one real quarter) gold feature
    table with n_batches in the teens, that is the wide table's full
    ~10GB read multiplied by n_batches - far more I/O and peak buffer
    use than the join itself needs, and the likely reason
    drive_batched_inner_join, even after being fixed to collect each
    side separately, still failed against real Q1 data on a small but
    real allocation with zero progress logged.

    `pipelines/build_gold_features.py`'s finalize stage writes the gold
    feature table with a `pyarrow.parquet.ParquetWriter`, one
    `write_table(..., row_group_size=table.num_rows)` call per batch of
    whole drives, so the file has exactly one row group per original
    batch there. That is NOT relied on for correctness here, only for
    how many times `narrow` gets re-scanned: Parquet row groups are
    always an exhaustive, non-overlapping partition of a file's rows
    regardless of how they were written, and the join is keyed on `on`
    against each row group's own physical rows, so even a row group that
    (against an older version of the writer, before it passed
    `row_group_size` explicitly - real batches routinely exceeded
    pyarrow's un-configured 1,048,576-row default cap and silently split
    into more row groups than intended) happens to straddle a drive's
    date range still produces a correct result - verified directly with
    a table deliberately split at arbitrary, non-drive-aligned row
    counts. Reading by row group is a direct, targeted read of just that
    row group's bytes - no filter evaluation, no re-reading rows that
    belong to a different one - so the WHOLE file is read exactly once
    in total, not once per row group.

    `narrow` (typically one horizon's observed-label rows) is re-scanned
    and filtered to each row group's own drive_ids via `.is_in(...)` -
    the same "redundant rescan of a NARROW source, once per batch"
    trade already made for Bronze in pipelines/build_silver.py's batch
    stage, which is far cheaper here than the wide side's would have
    been precisely because `narrow` is narrow.

    `max_rows_per_join`, when given, further slices each row group's
    already-in-memory DataFrame into row-count-bounded pieces before
    joining - free to do (no re-read: the row group is already
    resident), and it's what actually gives
    `resource_limits.training_join_chunk_rows` an effect on peak memory
    here. Without it, a caller who lowers that config expecting smaller
    join batches would see no change at all, since row-group size is
    governed entirely by `feature_batch_target_rows` (a different
    config key, read by a different pipeline) - `pipelines/train_model.py`
    passes both.

    Falls back to a single row group (the whole file) if `wide_path`
    happens to have been written as one - a small dataset that never
    needed batching in build_gold_features.py, or any other single-row-
    group Parquet file, both handled correctly, just with no batching
    benefit.

    Pass `out_path` to write the combined result directly there instead
    of returning it as one eager DataFrame - see
    `_finalize_batched_join`. This is the reason this parameter exists
    at all: against real Q1 data, every row group processed
    successfully (confirming the row-group read and per-row-group join
    were both correctly bounded), and the crash still happened - in the
    final `pl.concat([...]).collect()` that used to run after this
    loop, reassembling all ~30 batches back into one eager ~26-30M-row
    DataFrame. `pipelines/train_model.py`'s caller was just going to
    `.write_parquet()` that DataFrame immediately anyway, so writing
    each batch directly to the destination avoids that pointless full
    materialization entirely."""
    narrow_lazy = narrow.lazy()
    parquet_file = pq.ParquetFile(wide_path)
    logger.info(
        "join_by_native_row_groups_started",
        wide_path=str(wide_path),
        num_row_groups=parquet_file.num_row_groups,
        max_rows_per_join=max_rows_per_join,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        batch_paths = []
        for row_group_index in range(parquet_file.num_row_groups):
            t0 = time.perf_counter()
            wide_batch = _as_dataframe(
                pl.from_arrow(parquet_file.read_row_group(row_group_index)),
                context=f"row group {row_group_index} of {wide_path}",
            )
            # `.implode()`: passing a bare Series of the same dtype as the
            # column being checked is deprecated as of Polars 1.44 (it's
            # ambiguous whether it means "check membership against these
            # values" or an element-wise comparison) - imploding it into
            # a single list value makes "these are the values to check
            # membership against" explicit, the same semantics a plain
            # `list` (e.g. from `.to_list()`) already had, without paying
            # for that list's per-value Python object boxing.
            batch_drive_ids = wide_batch["drive_id"].unique().implode()
            narrow_batch = narrow_lazy.filter(pl.col("drive_id").is_in(batch_drive_ids)).collect()

            sub_size = max_rows_per_join or wide_batch.height
            for offset in range(0, wide_batch.height, sub_size):
                sub_wide = wide_batch.slice(offset, sub_size)
                part = sub_wide.join(narrow_batch, on=on, how="inner")
                if part.height:
                    path = Path(tmp_dir) / f"batch_{row_group_index}_{offset}.parquet"
                    part.write_parquet(path, compression="zstd")
                    batch_paths.append(path)
                del sub_wide, part
            logger.info(
                "join_by_native_row_groups_row_group_done",
                row_group_index=row_group_index,
                num_row_groups=parquet_file.num_row_groups,
                elapsed_seconds=round(time.perf_counter() - t0, 2),
                row_group_rows=wide_batch.height,
                matched_row_count=narrow_batch.height,
            )
            del wide_batch, narrow_batch
            gc.collect()

        empty_wide = _as_dataframe(
            pl.from_arrow(parquet_file.schema_arrow.empty_table()),
            context=f"empty-schema fallback for {wide_path}",
        )
        empty_narrow = narrow_lazy.filter(pl.lit(False)).collect()
        empty_result = empty_wide.join(empty_narrow, on=on, how="inner")
        return _finalize_batched_join(batch_paths, empty_result, out_path=out_path)


def assemble_training_frame(
    gold_features: pl.DataFrame | pl.LazyFrame,
    labels: pl.DataFrame | pl.LazyFrame,
    *,
    horizon_days: int,
    label_columns: list[str] | None = None,
    chunk_rows: int | None = None,
    gold_features_path: Path | None = None,
    out_path: Path | None = None,
) -> pl.DataFrame | None:
    """Joins gold features to the label table for one horizon, keeping only
    rows with an observed (non-censored) label.

    `gold_features` may be passed as a `pl.scan_parquet(...)` LazyFrame
    (`pipelines/train_model.py` does this), and `chunk_rows` then joins it
    in batches of whole drives - `hash(drive_id) % n_batches`, where
    `n_batches` is derived from `chunk_rows` and gold features' own row
    count - instead of all at once. See `drive_batched_inner_join` for why
    both sides of this specific join need to be batched together, not
    just `gold_features`: every one of gold features' ~196 columns is
    needed as a model feature (no predicate to push into that scan), and
    the label table's "one horizon, observed only" subset is NOT small
    either at real fleet scale (confirmed against real Q1 data - see
    `drive_batched_inner_join`'s docstring), so an unbatched join, or a
    join that only bounds one side, both hit the memory cap with far too
    much resident at once.

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
    callers are unaffected.

    Pass `gold_features_path` (the actual file `gold_features` was
    scanned from) alongside `chunk_rows` to join via
    `join_by_native_row_groups` instead of `drive_batched_inner_join` -
    reads the wide gold feature table exactly once in total, by its own
    existing per-batch row groups, rather than re-scanning it once per
    batch with a fresh hash filter (see `join_by_native_row_groups`'s
    docstring for why that rescan was the actual bottleneck even after
    `drive_batched_inner_join` collected each side separately).
    `chunk_rows` still bounds peak memory in this path too - passed
    through as `max_rows_per_join`, it further slices each row group's
    already-in-memory batch before joining, rather than being silently
    ignored once a path is available. `drive_batched_inner_join` remains
    the fallback when no path is given - e.g. for callers/tests using an
    in-memory or non-Parquet `gold_features` - since it needs an actual
    file to read row groups from.

    Pass `out_path` to write the result directly there instead of
    returning it as one eager DataFrame (returns `None` when given) -
    see `_finalize_batched_join`. `pipelines/train_model.py` always
    passes this: at fleet scale the joined training frame can be tens
    of millions of rows, and it was just going to `.write_parquet()`
    whatever this function returned anyway, so materializing it as one
    eager object first was pure waste - and, against real Q1 data, the
    actual remaining crash after every other fix in this join's history:
    every batch's own join succeeded, and the final
    `pl.concat([...]).collect()` that used to run after them (silently
    reassembling the whole result) is what failed."""
    horizon_labels = labels.lazy().filter(
        (pl.col("horizon_days") == horizon_days) & pl.col("label").is_not_null()
    )
    if label_columns is not None:
        horizon_labels = horizon_labels.select(label_columns)

    if chunk_rows is None:
        result = (
            gold_features.lazy()
            .join(horizon_labels, on=["drive_id", "date"], how="inner")
            .collect()
        )
        if out_path is not None:
            result.write_parquet(out_path, compression="zstd")
            return None
        return result

    if gold_features_path is not None:
        return join_by_native_row_groups(
            gold_features_path,
            horizon_labels,
            on=["drive_id", "date"],
            max_rows_per_join=chunk_rows,
            out_path=out_path,
        )

    total_rows = gold_features.lazy().select(pl.len()).collect().item()
    n_batches = max(1, math.ceil(total_rows / chunk_rows))
    return drive_batched_inner_join(
        gold_features,
        horizon_labels,
        on=["drive_id", "date"],
        n_batches=n_batches,
        out_path=out_path,
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
        df.select([pl.col(c).cast(pl.Float32) for c in feature_columns]).fill_null(0.0).to_numpy()
    )
