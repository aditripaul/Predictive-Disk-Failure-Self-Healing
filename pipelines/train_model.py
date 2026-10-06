"""Entry point for `make train`.

Assembles the training frame for the primary horizon, trains a LightGBM
model with class-imbalance weighting, tunes a precision-first threshold on
the validation split, evaluates on validation and test, and logs metrics
plus the model to MLflow.

Note (2026-09-30): the memory cap is now enforced on resident memory
(RSS), not on `RLIMIT_AS` virtual address space - see
src/resource_limits.py. Every crash described below, including the
last one (a few-hundred-KB allocation failing in the orchestrator right
after LightGBM training, at far less than 20GB of real use), was the
virtual-address cap tripping on reserved-but-unused address space, not
on memory actually in use. The stage split below is kept because it
still lowers real peak memory, but it is no longer load-bearing for the
cap itself.

Runs as FOUR separate processes in sequence - assemble, then one
extract-split per split (train, validation, test), then this top-level
orchestrator (no `--stage`) - all sharing one `--work-dir`. Polars is
built on jemalloc, which on 64-bit Linux defaults to retaining freed
virtual memory for reuse rather than returning it to the OS. RLIMIT_AS
(this pipeline's memory cap, src/resource_limits.py) constrains mapped
address space, not resident memory, so within one process it tracks the
high-water mark of everything that process has EVER allocated, not
what's currently live - deleting a large object and calling gc.collect()
does NOT lower this ceiling. Exiting a process unconditionally unmaps its
entire address space regardless of what the allocator was retaining, so
that's the only thing that reliably works here. Each stage below exists
because folding it into the process before it kept failing on real
(fleet-scale) data, even though EVERY one of these steps, in isolation,
fits the memory cap comfortably - this took three rounds to get right,
each one isolating a step that turned out to still be sharing a process
with another heavy step (see the history at the end of this docstring):

1. `--stage assemble`: joins gold features (~196 columns, ~10GB for one
   month) to the label table (src/models/features.py::
   assemble_training_frame, itself internally batched) and writes the
   joined frame straight to `work_dir/frame.parquet`, never held as one
   eager DataFrame in this process. Also validates split non-emptiness
   and writes `frame_meta.json` (feature_columns, each split's raw row
   count) - both cheap, narrow-column-only reads that don't touch the
   frame's full width. Exits immediately after.
2. `--stage extract-split --split {train,validation,test}`, run once per
   split: reads `frame.parquet` and `frame_meta.json` (written by a
   PRIOR, already-exited process) and collects JUST that one split,
   projected at scan time to only the columns it needs. Train is
   additionally capped (DEFAULT_MAX_TRAIN_ROWS / `model.max_train_rows`,
   _subsampled_train_lazy - every failure/positive-label row is kept,
   negative rows randomly subsampled down to the cap) because even a
   single clean copy of a real quarter's full train split no longer fits
   the cap on its own. Writes that split's own small, already-converted
   artifact under `work_dir` (`x_train.npy`/`y_train.npy` for train,
   `x_val.npy`/`y_val.npy` and `x_test.npy`/`y_test.npy` for validation and
   test, each with a row-aligned `<split>_ids.parquet`, plus a
   `{split}_meta.json` with its post-cap row count) and
   exits - so no two splits' collects ever share a process, any more than
   the join and a split's collect do.
3. The top-level orchestrator (this function): never scans or filters
   either the gold tables or the assembled frame itself - it only loads
   the already-small, already-final artifacts steps 1-2 wrote (a few GB
   `np.load` and a narrow Parquet read, not a scan+filter+cast+collect
   pipeline), then trains the model, tunes the threshold, evaluates, and
   runs SHAP.

History, since the reasoning generalizes beyond this specific pipeline:
attempt 1 put the join alone in its own subprocess, leaving `main()` to
collect all three splits itself - failed on real data because collecting
all three as full eager DataFrames, then a second numpy copy of each
while the DataFrame was still alive, needed several times the size of
even the largest split. Attempt 2 narrowed and sequenced that collection
(one split at a time, minimal columns, immediately freed) but still ran
it in `main()` - failed anyway, because each split's collect inherited
whatever high-water mark the earlier splits' collects had left behind in
that same process, even though any one split alone fit comfortably.
Attempt 3 moved ALL split collection into the join's own subprocess -
failed again, because the join's OWN retained memory was enough to starve
the very first split collected right after it. Attempt 4 isolated EVERY
distinct heavy Polars collect into its own process - the join, and each
split separately - and got further, but a plain `scan_parquet(...)
.filter(...).collect()` still failed even in a fully isolated process
(attempt 5's fix: row-group-native reads, `_row_group_parts`), and even
THAT still failed once fixed to spill each row group to disk instead of
accumulating results in memory (attempt 6). Peak-RSS instrumentation
(`resource.getrusage` checkpoints, since removed) finally pinned
attempt 6's failure exactly: forming ONE
combined Polars DataFrame from all the spilled parts, then converting
THAT to a numpy array, needed both the ~6GB combined DataFrame and a
further ~3.7GB array alive at once - together enough to fail even though
either one alone fit easily. Attempt 7 (`_build_feature_arrays`) builds
the numpy array directly from the spilled parts instead, one part at a
time, never forming a combined DataFrame at all.

The lesson generalizes beyond "isolate the heavy step" (attempts 1-4) to
"the SHAPE of how a result is built matters, not just which process it
runs in" (attempts 5-7): reading a wide file needs explicit row-group
discipline, a loop must spill and drop each iteration's result rather
than accumulate them, and converting a large result to a different
representation (Parquet -> Polars -> numpy) can itself double memory if
the intermediate representation lingers - build directly into the final
form instead of combining-then-converting through one.

Attempt 7 fixed the memory problem (train, then validation, completed
successfully for the first time), but immediately surfaced an unrelated
one: `np.save`'s underlying `array.tofile()` failed with a PARTIAL write
- the OS ran out of disk space partway through, independent of this
pipeline's own memory cap. `work_dir` was living under `tempfile`'s
default location (`/tmp`), which turned out too small to hold
`frame.parquet` (tens of GB at fleet scale) plus every split's own
artifacts at once. Fixed by putting `work_dir` under `resource_limits.
scratch_dir` (`_scratch_base`) instead - `<gold_dir>/../tmp` by default,
a filesystem already proven to have room for comparably large files -
and by spilling each split's own row-group parts into a subdirectory of
`work_dir` rather than a separate `tempfile.TemporaryDirectory()` (which
would have defaulted right back to `/tmp`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml

import mlflow
from src.labels.dataset_version import latest_dataset_version
from src.labels.event_types import FAILURE_EVENT_TYPES
from src.logging_config import configure_logging, get_logger
from src.models.evaluation import (
    compute_auprc,
    compute_warning_lead_time_days,
    drive_level_metrics,
    drive_level_table,
    evaluate_at_threshold,
    precision_at_recall_table,
)
from src.models.explainability import (
    build_explainer,
    compute_shap_values,
    global_feature_importance,
)
from src.models.features import (
    _as_dataframe,
    _finalize_batched_join,
    assemble_training_frame,
    feature_matrix,
    select_feature_columns,
)
from src.models.hyperparameter_tuning import tune_lightgbm_hyperparameters
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.model_card import build_model_card, render_model_card_markdown
from src.models.smote import apply_smote
from src.models.threshold import tune_action_tiers, tune_drive_level_threshold
from src.models.training import (
    early_stopping_subset,
    predict_proba_positive,
    train_lightgbm,
)
from src.models.two_stage import fit_second_stage, lightgbm_fitter, log_two_stage
from src.models.xgboost_training import train_xgboost
from src.reporting.figure_data import gain_importance, roc_points
from src.resource_limits import apply_memory_limit_from_config

SHAP_BACKGROUND_SAMPLE_SIZE = 100

#: Fallback when `resource_limits.training_join_chunk_rows` is absent from
#: configs/data.yaml.
DEFAULT_TRAINING_JOIN_CHUNK_ROWS = 2_000_000

#: Fallback when `model.max_train_rows` is absent from configs/model.yaml.
#: At ~185 float32 feature columns, 5,000,000 rows is ~3.7GB as a single
#: matrix - comfortable under the 20GB default memory cap even with the
#: transient second copy `feature_matrix`'s to_numpy() briefly needs plus
#: LightGBM's own histogram-building overhead. A real quarter's train
#: split is tens of millions of rows, well above this - see
#: _subsampled_train_lazy for how the cap is applied. Set
#: `model.max_train_rows: null` explicitly in configs/model.yaml to train
#: on every row instead (only advisable with a much higher memory cap or
#: a smaller dataset).
DEFAULT_MAX_TRAIN_ROWS = 5_000_000

#: Label-table columns the training pipeline actually uses: the target
#: (`label`), the split assignment, and `event_type`/`days_to_event` for
#: the warning-lead-time metric - plus the join keys. The label table has
#: 12 columns and is ~3x the drive-day count (one row per horizon), so
#: projecting to these before joining avoids carrying ~5.6GB of unused
#: columns through the join at fleet scale.
LABEL_COLUMNS_USED = [
    "drive_id",
    "date",
    "label",
    "split",
    "event_type",
    "days_to_event",
]

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"train_model_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def _subsample_rows(
    x: np.ndarray, y: np.ndarray, *, max_rows: int | None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` unchanged when they already fit within `max_rows`, else a
    fixed-seed random row sample of exactly `max_rows` rows.

    Returns the inputs themselves (not a copy) in the common case, so a
    dataset under the cap costs nothing - the cap only ever materializes a
    sample at fleet scale. `max_rows=None` disables the cap entirely."""
    if max_rows is None or x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    selected = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[selected], y[selected]


def _require_gold_inputs(data_config: dict) -> tuple[Path, Path]:
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    labels_path = gold_dir / "labels" / "part.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing gold features or labels; run `make build-features` and "
            "`make build-labels` first."
        )
    return features_path, labels_path


def _scratch_base(data_config: dict) -> Path:
    """Base directory for `main()`'s `work_dir` (the assembled training
    frame plus every split's intermediate artifacts - tens of GB
    combined at fleet scale, all present at once for as long as
    extraction is running).

    `resource_limits.scratch_dir` if set; otherwise `<gold_dir>/../tmp`,
    NOT `tempfile`'s own default location (`/tmp`, via `tempfile.
    mkdtemp()`/`TemporaryDirectory()` with no `dir=` argument) - on real
    Q1 data, `/tmp` turned out to be a small enough filesystem, unrelated
    to this pipeline's own memory cap, that writing x_val.npy ran out of
    disk space mid-write. `gold_dir`'s own filesystem is a safer default
    since the gold feature/label tables it already holds are comparable
    in size to what this needs."""
    configured = data_config.get("resource_limits", {}).get("scratch_dir")
    base = Path(configured) if configured else Path(data_config["gold_dir"]).parent / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_id_columns(part_paths: list[Path], columns: list[str], out_path: Path) -> None:
    """Writes just `columns` (drive_id, days_to_event, ...) from the spilled
    part files to one small Parquet file, row-aligned with the arrays
    `_build_feature_arrays` builds from the same parts (same order). Needed
    for drive-level evaluation and per-drive weighting; read narrow, one
    part at a time, so it adds nothing to the peak."""
    frames = [pl.read_parquet(path, columns=columns) for path in part_paths]
    (pl.concat(frames) if frames else pl.DataFrame({c: [] for c in columns})).write_parquet(
        out_path, compression="zstd"
    )


def _stage_assemble(work_dir: Path, horizon_days: int | None = None) -> None:
    """Subprocess stage: joins gold features to the label table for the
    primary horizon and writes the result to `work_dir / "frame.parquet"`.

    Deliberately does ONLY the join, not split extraction too - an
    earlier version of this function did both in one process, on the
    theory that isolating "assemble + splits" from `main()` would be
    enough. On real data it wasn't: the join's own high-water mark
    (jemalloc, RLIMIT_AS - see the module docstring) was still large
    enough that the very first split collected right after it, in the
    SAME process, failed to allocate a few GB it needed. Exiting a
    process unconditionally unmaps its entire address space regardless
    of what the allocator was retaining, so the join gets its own
    process here, and split extraction (_stage_extract_splits) gets a
    separate one of its own."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_path, labels_path = _require_gold_inputs(data_config)

    # Scanned lazily, not pl.read_parquet'd: gold features is ~196 columns
    # and ~10GB for one month of real data, but assemble_training_frame's
    # join only keeps one horizon's observed-label rows - reading the whole
    # table eagerly first would materialize all ~10GB before the join gets
    # a chance to discard most of it (see src/models/features.py).
    gold_features = pl.scan_parquet(features_path)
    # Scanned lazily and projected to LABEL_COLUMNS_USED inside
    # assemble_training_frame, rather than pl.read_parquet'd: the label
    # table is one row per drive-day per horizon (31.4M rows, ~5.6GB for
    # one month of real data) and only one horizon and 6 of its 12
    # columns are ever used, so reading it whole put ~5GB of dead columns
    # alive alongside the join's own output.
    labels = pl.scan_parquet(labels_path)

    if horizon_days is None:
        horizon_days = model_config["primary_horizon_days"]
    work_dir.mkdir(parents=True, exist_ok=True)
    frame_path = work_dir / "frame.parquet"
    t0 = time.perf_counter()
    # gold_features_path lets assemble_training_frame join by gold
    # features' own existing row groups (one per batch, already written
    # by build_gold_features.py) instead of re-deriving batches with a
    # fresh hash filter, which - even after being fixed to collect each
    # side of the join separately - still meant rescanning the entire
    # ~196-column, ~10GB (one real quarter) file once per batch just to
    # evaluate an unprunable filter. See src/models/features.py's
    # join_by_native_row_groups for the full explanation.
    #
    # out_path=frame_path: the joined training frame can itself be tens
    # of millions of rows at fleet scale, and this function was just
    # going to write it to frame_path immediately after anyway - passing
    # the destination straight through means it's written incrementally,
    # one batch at a time, and never exists as a single eager DataFrame
    # in this process at all. Against real Q1 data, this was the actual
    # remaining crash after every earlier fix in this join's history:
    # every batch's own join succeeded, and reassembling them all into
    # one eager DataFrame right before writing it straight back out
    # (which frame.write_parquet(...) below used to do) is what failed.
    assemble_training_frame(
        gold_features,
        labels,
        horizon_days=horizon_days,
        label_columns=LABEL_COLUMNS_USED,
        chunk_rows=data_config.get("resource_limits", {}).get(
            "training_join_chunk_rows", DEFAULT_TRAINING_JOIN_CHUNK_ROWS
        ),
        gold_features_path=features_path,
        out_path=frame_path,
    )
    # row_count/column_count read back from the written file - cheap,
    # metadata-only for the count - rather than from the DataFrame this
    # function no longer returns.
    row_count = pl.scan_parquet(frame_path).select(pl.len()).collect().item()
    column_count = len(pl.scan_parquet(frame_path).collect_schema().names())
    _log_stage("training_frame_assembled", t0, row_count=row_count, column_count=column_count)

    # Split row counts read via a projection to just `split` - one narrow
    # column, cheap regardless of the frame's ~196-column width - rather
    # than collecting any split to check it's non-empty. Failing fast
    # here, before any of the three extract-split subprocesses even
    # start, avoids launching (and paying the join-output rescan cost of)
    # subprocesses that would just raise the same error anyway.
    split_counts_df = (
        pl.scan_parquet(frame_path).group_by("split").agg(pl.len().alias("count")).collect()
    )
    if split_counts_df.height == 0:
        raise ValueError(
            f"No rows with an observed (non-censored) {horizon_days}-day label. "
            "This is expected against the synthetic stub dataset, which has too "
            "short a history for any row to reach horizon observability - it will "
            "resolve once real Backblaze data (Phase 1) is ingested."
        )
    split_counts = dict(
        zip(split_counts_df["split"].to_list(), split_counts_df["count"].to_list(), strict=True)
    )
    empty_splits = [
        name for name in ("train", "validation", "test") if split_counts.get(name, 0) == 0
    ]
    if empty_splits:
        raise ValueError(
            f"Split(s) {empty_splits} have no observed-label rows for "
            f"horizon={horizon_days}d. Ensure the chronological split boundaries "
            "in configs/model.yaml align with the ingested data's date range."
        )

    # A schema-only read (no rows) is enough for select_feature_columns,
    # which only inspects `.columns`/`.dtypes`. Written out here so each
    # extract-split subprocess below doesn't need to re-derive it (cheap
    # either way, but this keeps it computed in exactly one place).
    feature_columns = select_feature_columns(pl.scan_parquet(frame_path).limit(0).collect())
    (work_dir / "frame_meta.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "split_counts": split_counts,
                "horizon_days": horizon_days,
            }
        )
    )


def _split_period_predicate(
    split_name: str, model_config: dict, horizon_days: int
) -> pl.Expr | None:
    """The dates each split may use, beyond the label table's own `split`
    column. All three default on; `splits.purge_label_window: false` or
    `splits.validation_after_train_only: false` in configs/model.yaml
    restore the earlier behaviour.

    - **train:** only rows dated at least `horizon_days` before `train_end`.
      A later row's label is decided by what happens after `train_end`,
      inside the validation period, which a model trained on `train_end`
      could not know. It also made validation look easier than the future:
      the same drive's rows a few days apart sat in train and validation
      with nearly the same features and the same outcome.
    - **validation:** only rows after `train_end`. The label table also
      puts the held-out drives' rows from the training period here; those
      share their dates with the training rows, so thresholds and early
      stopping chosen on them did not carry over to test.
    - **test:** only rows up to `test_end`. Later, a healthy drive-day has
      no label (its horizon runs past the data) and is dropped, but a
      failing one keeps its positive label, so those dates would add
      failing drives with no healthy rows beside them. `test_end` must be
      at least the horizon before the last date in the data."""
    splits = model_config.get("splits", {})

    def boundary(name: str) -> dt.date | None:
        value = splits.get(name)
        return dt.date.fromisoformat(str(value)) if value else None

    train_end, test_end = boundary("train_end"), boundary("test_end")
    if split_name == "train" and train_end and splits.get("purge_label_window", True):
        return pl.col("date") <= train_end - dt.timedelta(days=horizon_days)
    if split_name == "validation" and train_end and splits.get("validation_after_train_only", True):
        return pl.col("date") > train_end
    if split_name == "test" and test_end:
        return pl.col("date") <= test_end
    return None


def _row_group_parts(
    frame_path: Path,
    *,
    split_name: str,
    select_columns: list[str],
    read_columns: list[str],
    extra_predicate: pl.Expr | None,
    tmp_dir: Path,
) -> list[Path]:
    """Reads `frame_path` ONE PARQUET ROW GROUP AT A TIME via
    `pyarrow.parquet.ParquetFile.read_row_group` - the same technique
    `src/models/features.py::join_by_native_row_groups` already uses for
    this exact file for this exact reason - filtering each row group down
    to `split_name` (and `extra_predicate`, if given) before moving to
    the next, rather than a plain `scan_parquet(...).filter(...).collect()`
    that leaves Polars' own query engine to decide how many row groups to
    decode concurrently.

    Each row group's filtered/selected slice is spilled to its own small
    temp Parquet file (under `tmp_dir`, caller-owned) and dropped
    immediately, exactly like `join_by_native_row_groups` - NOT
    accumulated as a live DataFrame in a Python list across the whole
    loop, which an earlier version of this function did. On real data
    that was enough, on its own, to fail a later step even for train's
    OWN split (well within the memory cap on its own): every row group's
    slice - summing to the split's entire final size - stayed alive as
    Python objects for the whole loop, on top of the per-row-group decode
    work, rather than each one being written out and freed before the
    next row group is read.

    `read_columns` are the raw columns read from each row group (must
    cover `select_columns`, `"split"`, and anything `extra_predicate`
    references); `select_columns` are what survives into each part file.
    Returns the spilled parts' paths - see `_collect_split_rows` (small
    results: combine into one returned/written DataFrame via
    `_finalize_batched_join`) and `_build_feature_arrays` (large results:
    build a numpy array directly from the parts, without ever forming a
    combined DataFrame at all) for what to do with them."""
    parquet_file = pq.ParquetFile(frame_path)
    part_paths: list[Path] = []
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i, columns=read_columns)
        batch = _as_dataframe(pl.from_arrow(table), context=f"row group {i} of {frame_path}")
        batch = batch.filter(pl.col("split") == split_name)
        if extra_predicate is not None:
            batch = batch.filter(extra_predicate)
        if batch.height:
            part = batch.select(select_columns)
            path = tmp_dir / f"part_{i}.parquet"
            part.write_parquet(path, compression="zstd")
            part_paths.append(path)
            del part
        del table, batch
        gc.collect()
    return part_paths


def _collect_split_rows(
    frame_path: Path,
    *,
    split_name: str,
    select_columns: list[str],
    read_columns: list[str],
    tmp_dir: Path,
    extra_predicate: pl.Expr | None = None,
    out_path: Path | None = None,
) -> pl.DataFrame | None:
    """Collects one split via `_row_group_parts`, then combines the
    spilled parts through `_finalize_batched_join` - the same proven path
    `join_by_native_row_groups` uses for its own, much larger, result.

    `tmp_dir` (caller-provided, not created here via `tempfile`'s own
    default location) is where the spilled parts live - see
    `_stage_extract_split`, which derives it from `work_dir` so every
    scratch file this pipeline writes lands on the same, deliberately
    chosen filesystem (`resource_limits.scratch_dir`) rather than
    whatever the system's default temp directory happens to be.

    Pass `out_path` to write the combined result directly there (one
    part at a time, never forming a single combined DataFrame in this
    process - see `_finalize_batched_join`) instead of returning it. No
    split uses this function any more (test used to, to write a
    `test_df.parquet`): all three build arrays instead, for
    the same reason: they need a numpy array, not a DataFrame or a file,
    and forming a combined DataFrame only to immediately convert it to a
    numpy array meant both the combined DataFrame AND the array it
    produced needed to be resident at once - on real data, together
    enough to fail an allocation smaller than either one alone. See
    `_build_feature_arrays`, which builds directly into the final array
    instead."""
    parquet_file = pq.ParquetFile(frame_path)
    part_paths = _row_group_parts(
        frame_path,
        split_name=split_name,
        select_columns=select_columns,
        read_columns=read_columns,
        extra_predicate=extra_predicate,
        tmp_dir=tmp_dir,
    )
    empty = _as_dataframe(
        pl.from_arrow(parquet_file.schema_arrow.empty_table()),
        context=f"empty-schema fallback for {frame_path}",
    ).select(select_columns)
    result = _finalize_batched_join(part_paths, empty, out_path=out_path)
    if out_path is not None:
        return None
    return _as_dataframe(result, context=f"_collect_split_rows({split_name}) of {frame_path}")


def _build_feature_arrays(
    part_paths: list[Path],
    feature_columns: list[str],
    *,
    label_column: str = "label",
    out_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds `(x, y)` directly from `_row_group_parts`' spilled parts,
    filling ONE pre-allocated array part by part - never forming a single
    combined Polars DataFrame first, unlike `_collect_split_rows`'s
    `out_path=None` path. On real data, that combined DataFrame (~6GB for
    a 5-million-row, ~187-column train split - every part concatenated
    into one Polars frame) plus the further numpy array `feature_matrix`
    would then build FROM that combined frame were, together, enough to
    fail an allocation (~3.7GB) smaller than either one alone. Building
    directly into the final array instead means the only large thing
    resident at any point is the array itself, plus one small part's own
    data - never a second nearly-full-size copy alongside it.

    With `out_path`, the matrix is written straight to that `.npy` file
    through a memory map instead of being allocated in RAM, and the returned
    `x` is the map. The validation and test splits are not row-capped, so with
    several quarters of data their matrices are larger than the train matrix
    (about 11 GB and 6 GB for two quarters) and are never held in RAM whole -
    neither here nor by the orchestrator, which maps them read-only and
    scores them in chunks."""
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in part_paths)
    shape = (total_rows, len(feature_columns))
    x: np.ndarray
    if out_path is not None:
        x = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
    else:
        x = np.empty(shape, dtype=np.float32)
    if not part_paths:
        return x, np.empty((0,))
    y_parts: list[np.ndarray] = []
    offset = 0
    for path in part_paths:
        part_df = pl.read_parquet(path)
        n = part_df.height
        x[offset : offset + n] = feature_matrix(part_df, feature_columns)
        y_parts.append(part_df[label_column].to_numpy())
        offset += n
        del part_df
        gc.collect()
    if isinstance(x, np.memmap):
        x.flush()
    return x, np.concatenate(y_parts)


def _stage_extract_split(work_dir: Path, split_name: str) -> None:
    """Subprocess stage: reads `work_dir / "frame.parquet"` (written by a
    PRIOR, already-exited `_stage_assemble` run) and collects ONE split -
    `split_name`, one of "train"/"validation"/"test" - into a small,
    already-converted artifact under `work_dir`: `x_train.npy`/
    `y_train.npy` for train, `x_val.npy`/`y_val.npy` and `x_test.npy`/
    `y_test.npy` for validation and test (each with a row-aligned
    `<split>_ids.parquet`), plus a `{split_name}_meta.json`
    with that split's post-cap row count.

    Runs in its own process, separate from `_stage_assemble`, `main()`,
    AND the other two splits: an earlier version collected all three
    splits in one shared "extract-splits" subprocess (itself already
    split out from `_stage_assemble` for the same reason), and on real
    data even THAT failed - train's own collect succeeded, but
    validation's collect, right after it in the same process, still
    failed to allocate memory it needed on its own. See the module
    docstring: every distinct heavy Polars collect needs its own
    process, not just every "phase" of the pipeline. Collects via
    row-group-native reads (`_row_group_parts`), not a plain
    scan+filter+collect - see that function's docstring for why a fully
    isolated subprocess still wasn't enough on its own. Train and
    validation build their numpy arrays directly from the spilled parts
    (`_build_feature_arrays`), never forming a combined Polars DataFrame
    first - see that function's docstring for why even THAT (row-group-
    native, spilled-to-disk-per-part) still wasn't enough on its own."""
    configure_logging()
    apply_memory_limit_from_config()
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    frame_path = work_dir / "frame.parquet"
    frame_meta = json.loads((work_dir / "frame_meta.json").read_text())
    feature_columns = frame_meta["feature_columns"]

    t0 = time.perf_counter()

    if split_name == "train":
        # Capped - see DEFAULT_MAX_TRAIN_ROWS and _train_sample_predicate
        # - because even a single clean copy of every real quarter's
        # full train split no longer fits in the memory cap on its own.
        # Counts read via the existing lazy scan first - cheap and narrow
        # (just split+label), unlike the wide per-row-group read below,
        # so it doesn't need the same row-group discipline.
        max_train_rows = model_config["model"].get("max_train_rows", DEFAULT_MAX_TRAIN_ROWS)
        period = _split_period_predicate("train", model_config, frame_meta["horizon_days"])
        train_rows = pl.scan_parquet(frame_path).filter(pl.col("split") == "train")
        if period is not None:
            train_rows = train_rows.filter(period)
        label_counts_df = train_rows.group_by("label").agg(pl.len().alias("count")).collect()
        label_counts = dict(
            zip(
                label_counts_df["label"].to_list(),
                label_counts_df["count"].to_list(),
                strict=True,
            )
        )
        rows_before_cap = int(sum(label_counts.values()))
        if rows_before_cap == 0:
            raise ValueError(
                "No train rows are left after dropping those whose label window runs past "
                f"train_end (horizon={frame_meta['horizon_days']}d). Move splits.train_end "
                "later, or set splits.purge_label_window: false in configs/model.yaml."
            )
        sample = _train_sample_predicate(label_counts, max_rows=max_train_rows, seed=0)
        predicate = period if sample is None else sample if period is None else (period & sample)
        # event_type/days_to_event are not model inputs: they give each train
        # row its time to failure, for the survival variants in
        # pipelines/experiment_model.py.
        train_id_columns = ["drive_id", "date", "drive_model", "event_type", "days_to_event"]
        select_columns = [*feature_columns, "label", *train_id_columns]
        # A subdirectory of work_dir, not tempfile's own default location
        # (typically /tmp): every scratch file this pipeline writes needs
        # to land on the same, deliberately chosen filesystem - see
        # main()'s work_dir setup and resource_limits.scratch_dir. On
        # real data, /tmp itself turned out too small to hold
        # frame.parquet plus every split's own artifacts at once.
        part_tmp_dir = work_dir / "_train_parts"
        part_tmp_dir.mkdir()
        try:
            part_paths = _row_group_parts(
                frame_path,
                split_name="train",
                select_columns=select_columns,
                read_columns=sorted({*select_columns, "split"}),
                extra_predicate=predicate,
                tmp_dir=part_tmp_dir,
            )
            _, y_train = _build_feature_arrays(
                part_paths, feature_columns, out_path=work_dir / "x_train.npy"
            )
            _write_id_columns(part_paths, train_id_columns, work_dir / "train_ids.parquet")
        finally:
            shutil.rmtree(part_tmp_dir, ignore_errors=True)
        row_count = len(y_train)
        np.save(work_dir / "y_train.npy", y_train)
        meta = {"row_count": row_count, "row_count_before_cap": rows_before_cap}
    else:
        # Validation and test are extracted the same way: the feature matrix
        # goes straight to `x_<name>.npy` through a memory map, the labels to
        # `y_<name>.npy`, and the columns evaluation needs beside the scores
        # (drive, date, drive model, event type, days to the event) to a small
        # row-aligned `<split>_ids.parquet`.
        short_name = "val" if split_name == "validation" else "test"
        id_columns = ["drive_id", "date", "drive_model", "event_type", "days_to_event"]
        select_columns = [*feature_columns, "label", *id_columns]
        part_tmp_dir = work_dir / f"_{split_name}_parts"
        part_tmp_dir.mkdir()
        try:
            part_paths = _row_group_parts(
                frame_path,
                split_name=split_name,
                select_columns=select_columns,
                read_columns=sorted({*select_columns, "split"}),
                extra_predicate=_split_period_predicate(
                    split_name, model_config, frame_meta["horizon_days"]
                ),
                tmp_dir=part_tmp_dir,
            )
            _, y_split = _build_feature_arrays(
                part_paths, feature_columns, out_path=work_dir / f"x_{short_name}.npy"
            )
            _write_id_columns(part_paths, id_columns, work_dir / f"{split_name}_ids.parquet")
        finally:
            shutil.rmtree(part_tmp_dir, ignore_errors=True)
        row_count = len(y_split)
        if row_count == 0:
            raise ValueError(
                f"The {split_name} split has no rows inside its date range "
                f"(horizon={frame_meta['horizon_days']}d). Check splits.train_end / "
                "validation_end / test_end in configs/model.yaml against the data's dates."
            )
        np.save(work_dir / f"y_{short_name}.npy", y_split)
        meta = {"row_count": row_count}

    (work_dir / f"{split_name}_meta.json").write_text(json.dumps(meta))
    _log_stage(f"{split_name}_split_extracted", t0, row_count=row_count)


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage - a new
    process, and therefore a new address space, regardless of what the
    calling process's allocator has retained. See the module docstring."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def _subsampled_train_lazy(
    train_lazy: pl.LazyFrame, *, max_rows: int | None, seed: int = 0
) -> pl.LazyFrame:
    """Caps the primary model's training row count for fleet-scale data,
    where a single copy of the full train split's feature matrix - let
    alone the second, similarly sized copy `feature_matrix`'s to_numpy()
    transiently needs - no longer fits in the memory cap regardless of how
    efficiently the rest of the pipeline avoids redundant copies. See
    DEFAULT_MAX_TRAIN_ROWS.

    Keeps every positive (failure) row - disk failure is the rare class,
    and subsampling can't afford to lose it - and randomly keeps a
    `hash(drive_id, date)`-derived fraction of the negative rows to reach
    `max_rows`. This is the same deterministic, reproducible,
    engine-native sampling idiom used for drive-batching elsewhere in this
    codebase (see src/models/features.py's `hash(drive_id) % n_batches`),
    rather than reading the label column into Python to sample row
    indices, which would itself require materializing a column across the
    whole (uncapped) train split just to decide how to cap it."""
    if max_rows is None:
        return train_lazy
    label_counts = train_lazy.group_by("label").agg(pl.len().alias("count")).collect()
    counts = dict(
        zip(label_counts["label"].to_list(), label_counts["count"].to_list(), strict=True)
    )
    predicate = _train_sample_predicate(counts, max_rows=max_rows, seed=seed)
    return train_lazy if predicate is None else train_lazy.filter(predicate)


def _train_sample_predicate(
    label_counts: dict[int, int], *, max_rows: int | None, seed: int
) -> pl.Expr | None:
    """The filter expression `_subsampled_train_lazy` (a lazy scan) and
    `_stage_extract_split`'s row-group-native path (an eager per-row-group
    loop) both use to decide which train rows to keep - factored out so
    the two call sites sample identically rather than maintaining the
    hashing logic twice. Returns `None` when no cap applies (label_counts'
    total is already <= max_rows, or max_rows is None) - see
    _subsampled_train_lazy for what the expression means."""
    if max_rows is None:
        return None
    n_positive = label_counts.get(1, 0)
    n_negative = label_counts.get(0, 0)
    if n_positive + n_negative <= max_rows:
        return None
    target_negative = max(0, max_rows - n_positive)
    keep_fraction = min(1.0, target_negative / n_negative) if n_negative else 1.0
    hash_bucket = 1_000_000
    row_hash = (pl.col("drive_id").hash(seed=seed) ^ pl.col("date").hash(seed=seed)) % hash_bucket
    return (pl.col("label") == 1) | (row_hash < int(keep_fraction * hash_bucket))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["assemble", "extract-split"])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument(
        "--horizon-days",
        type=int,
        help="Train for this label horizon instead of configs/model.yaml primary_horizon_days "
        "(must be one of horizons_days, which build-labels already computed). The run is "
        "logged with this horizon_days param; score_fleet only uses runs whose horizon "
        "matches primary_horizon_days.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Also train a second model that re-ranks the first model's candidates "
        "(src/models/two_stage.py), for this run only. Same as model.two_stage.enabled: "
        "true in configs/model.yaml.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the scratch work dir (x_*/y_*.npy, *_ids.parquet) after "
        "the run, for pipelines/experiment_model.py. Delete it yourself when done.",
    )
    args = parser.parse_args()

    if args.stage == "assemble":
        _stage_assemble(args.work_dir, args.horizon_days)
        return
    if args.stage == "extract-split":
        _stage_extract_split(args.work_dir, args.split)
        return

    # No --stage: the top-level orchestrator. It never itself scans
    # gold_features, the label table, or the assembled training frame -
    # the assemble and extract-split subprocesses it spawns do all of
    # that - so its own address space stays clean for model training and
    # SHAP, no matter how large the join or the splits were. See the
    # module docstring.
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())
    _require_gold_inputs(data_config)

    horizon_days = args.horizon_days or model_config["primary_horizon_days"]
    if horizon_days not in model_config["horizons_days"]:
        raise SystemExit(
            f"--horizon-days {horizon_days} has no labels: build-labels computes "
            f"horizons_days={model_config['horizons_days']} (configs/model.yaml)."
        )
    work_dir = Path(tempfile.mkdtemp(prefix="train_model_frame_", dir=_scratch_base(data_config)))
    try:
        t0 = time.perf_counter()
        # FOUR separate subprocesses, not one doing everything or even
        # one per "phase": the join's own retained virtual memory
        # (jemalloc, RLIMIT_AS - see the module docstring) was still
        # large enough, when all three splits were collected in one
        # shared subprocess right after it, that even train's own
        # (smaller, capped) collect starved validation's collect right
        # after it in that same process - even though every one of these
        # collects fits the memory cap comfortably on its own. Each
        # subprocess here gets a fully clean address space, guaranteed by
        # starting a new process rather than by anything any stage does
        # internally.
        _run_stage(
            "--stage",
            "assemble",
            "--work-dir",
            str(work_dir),
            "--horizon-days",
            str(horizon_days),
        )
        for split_name in ("train", "validation", "test"):
            _run_stage(
                "--stage", "extract-split", "--work-dir", str(work_dir), "--split", split_name
            )

        frame_meta = json.loads((work_dir / "frame_meta.json").read_text())
        feature_columns = frame_meta["feature_columns"]
        train_meta = json.loads((work_dir / "train_meta.json").read_text())
        validation_meta = json.loads((work_dir / "validation_meta.json").read_text())
        test_meta = json.loads((work_dir / "test_meta.json").read_text())

        x_train = np.load(work_dir / "x_train.npy")
        y_train = np.load(work_dir / "y_train.npy")

        mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(model_config["mlflow"]["experiment_name"])

        # Mapped read-only, not loaded: validation and test are not row-capped
        # and are scored in chunks (src/models/training.py::
        # predict_proba_positive), so only the train matrix is held in RAM.
        x_val = np.load(work_dir / "x_val.npy", mmap_mode="r")
        y_val = np.load(work_dir / "y_val.npy")
        # Distinct from each _stage_extract_split's own
        # "{split}_split_extracted" log: this one times the whole round
        # trip through all four subprocesses (spawn, join, three split
        # extractions, and loading the compact artifacts back) as seen
        # from main(), not any single stage's own work.
        _log_stage(
            "splits_loaded",
            t0,
            train=train_meta["row_count"],
            train_before_cap=train_meta.get(
                "row_count_before_cap", frame_meta["split_counts"]["train"]
            ),
            validation=validation_meta["row_count"],
            test=test_meta["row_count"],
        )

        model_type = model_config["model"].get("type", "lightgbm")
        diagnostics_cfg = model_config.get("diagnostics", {})

        with mlflow.start_run():
            results: dict = {
                "horizon_days": horizon_days,
                "feature_columns": feature_columns,
                "model_type": model_type,
            }

            hp_search_cfg = model_config.get("hyperparameter_search", {})
            model_params = model_config["model"]["params"]
            if model_type == "xgboost" and hp_search_cfg.get("enabled", False):
                raise ValueError(
                    "hyperparameter_search is only implemented for model.type: lightgbm "
                    "(src/models/hyperparameter_tuning.py); disable it or switch model.type."
                )
            if model_type == "lightgbm" and hp_search_cfg.get("enabled", False):
                search_result = tune_lightgbm_hyperparameters(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    base_params=model_config["model"]["params"],
                    search_space=hp_search_cfg["search_space"],
                    n_trials=hp_search_cfg.get("n_trials", 20),
                    subsample_fraction=hp_search_cfg.get("subsample_fraction", 0.3),
                    seed=hp_search_cfg.get("seed", 0),
                    positive_weight_power=model_config["model"].get("positive_weight_power"),
                )
                model_params = search_result["best_params"]
                results["hyperparameter_search"] = {
                    "best_params": search_result["best_params"],
                    "best_value": search_result["best_value"],
                    "n_trials": search_result["n_trials"],
                    "subsample_size": search_result["subsample_size"],
                }
                mlflow.log_params(
                    {f"tuned_{k}": v for k, v in search_result["best_params"].items()}
                )

            # Retrain on the FULL training partition with the (possibly tuned)
            # params - the search above only ever sees a subsample.
            t0 = time.perf_counter()
            if model_type == "xgboost":
                model = train_xgboost(x_train, y_train, params=model_params)
            elif model_type == "lightgbm":
                early_stopping_rounds = model_config["model"].get("early_stopping_rounds")
                eval_idx = (
                    early_stopping_subset(
                        y_val,
                        max_negatives=model_config["model"].get(
                            "early_stopping_max_negatives", 1_500_000
                        ),
                    )
                    if early_stopping_rounds
                    else None
                )
                model = train_lightgbm(
                    x_train,
                    y_train,
                    params=model_params,
                    positive_weight_power=model_config["model"].get("positive_weight_power"),
                    eval_x=x_val[eval_idx] if eval_idx is not None else None,
                    eval_y=y_val[eval_idx] if eval_idx is not None else None,
                    early_stopping_rounds=early_stopping_rounds,
                )
                del eval_idx
            else:
                raise ValueError(
                    f"Unknown model.type: {model_type!r} (expected lightgbm or xgboost)"
                )
            _log_stage("model_trained", t0, model_type=model_type, train_row_count=x_train.shape[0])

            val_scores = predict_proba_positive(model, x_val)
            val_drive_ids = pl.read_parquet(work_dir / "validation_ids.parquet")[
                "drive_id"
            ].to_numpy()

            # Optional second stage (src/models/two_stage.py): a second model
            # re-ranks the rows Stage 1 scores highest. `model` stays Stage 1
            # (logged, explained and plotted as before); the scores every
            # threshold and report below is built on become the combined ones.
            two_stage_cfg = model_config["model"].get("two_stage", {})
            two_stage_model = None
            stage1_val_scores = val_scores
            if args.two_stage or two_stage_cfg.get("enabled", False):
                if model_type != "lightgbm":
                    raise ValueError("model.two_stage is only implemented for model.type: lightgbm")
                t0 = time.perf_counter()
                two_stage_model, two_stage_report = fit_second_stage(
                    lightgbm_fitter(
                        model_params,
                        positive_weight_power=model_config["model"].get("positive_weight_power"),
                        early_stopping_rounds=model_config["model"].get("early_stopping_rounds"),
                        seed=two_stage_cfg.get("seed", 0),
                    ),
                    model,
                    x_train=x_train,
                    y_train=y_train,
                    train_drives=pl.read_parquet(work_dir / "train_ids.parquet")[
                        "drive_id"
                    ].to_numpy(),
                    x_val=x_val,
                    y_val=y_val,
                    val_drives=val_drive_ids,
                    val_stage1_scores=stage1_val_scores,
                    candidate_recall=two_stage_cfg.get("candidate_recall", 0.5),
                    n_folds=two_stage_cfg.get("n_folds", 3),
                    seed=two_stage_cfg.get("seed", 0),
                )
                results["two_stage"] = two_stage_report
                _log_stage("two_stage_trained", t0, **two_stage_report)
                if two_stage_model is not None:
                    val_scores = two_stage_model.final_scores(x_val, stage1_val_scores)

            # Chosen on the DRIVE-level precision/recall curve, honoring both
            # the precision target and the recall range; if the goal is
            # unreachable, the best achievable point inside the recall range
            # (never a single-drive, ~0%-recall fallback). See
            # src/models/threshold.py::tune_drive_level_threshold.
            threshold_result = tune_drive_level_threshold(
                val_drive_ids,
                y_val,
                val_scores,
                target_precision=model_config["threshold"]["target_precision"],
                target_recall_range=tuple(model_config["threshold"]["target_recall_range"]),
            )
            results["threshold"] = threshold_result
            results["validation_metrics"] = evaluate_at_threshold(
                y_val, val_scores, threshold_result["threshold"]
            )

            x_test = np.load(work_dir / "x_test.npy", mmap_mode="r")
            y_test = np.load(work_dir / "y_test.npy")
            test_ids = pl.read_parquet(work_dir / "test_ids.parquet")
            test_scores = predict_proba_positive(model, x_test)
            if two_stage_model is not None:
                # Stage 1 alone on the same drives, so every report shows what
                # the second stage changed.
                results["two_stage"]["stage1_precision_at_recall"] = precision_at_recall_table(
                    val_drive_ids,
                    y_val,
                    stage1_val_scores,
                    test_ids["drive_id"].to_numpy(),
                    y_test,
                    test_scores,
                )
                test_scores = two_stage_model.final_scores(x_test, test_scores)
            del stage1_val_scores
            results["test_metrics"] = evaluate_at_threshold(
                y_test, test_scores, threshold_result["threshold"]
            )
            # One threshold per agent action tier, each tied to the drive-level
            # precision that action can tolerate (configs/model.yaml
            # threshold.action_tier_precision). Evaluated on test below.
            tier_targets = model_config["threshold"]["action_tier_precision"]
            tier_thresholds = tune_action_tiers(
                val_drive_ids, y_val, val_scores, precision_targets=tier_targets
            )
            # The model goal (precision >= 95%, recall 35-50%) is judged per
            # DRIVE, not per drive-day: a failing drive contributes ~horizon
            # near-identical positive rows, so the row-level numbers above
            # over-count it. See src/models/evaluation.py::drive_level_table.
            results["validation_drive_level"] = drive_level_metrics(
                val_drive_ids,
                y_val,
                val_scores,
                threshold_result["threshold"],
            )
            results["test_drive_level"] = drive_level_metrics(
                test_ids["drive_id"].to_numpy(),
                y_test,
                test_scores,
                threshold_result["threshold"],
            )
            results["precision_at_recall"] = precision_at_recall_table(
                val_drive_ids,
                y_val,
                val_scores,
                test_ids["drive_id"].to_numpy(),
                y_test,
                test_scores,
            )
            # Which data this run saw. Results moved by ~0.04 AUPRC between two
            # builds of the "same" quarter, so a number without its build is
            # not comparable with another.
            dataset_version_record = latest_dataset_version(Path(data_config["audit_dir"]))
            results["data_build"] = {
                "dataset_version": (
                    dataset_version_record["version_id"] if dataset_version_record else None
                ),
                "rows": {
                    "train": train_meta["row_count"],
                    "train_before_cap": train_meta.get(
                        "row_count_before_cap", frame_meta["split_counts"]["train"]
                    ),
                    "validation": validation_meta["row_count"],
                    "test": test_meta["row_count"],
                },
                "failing_drives": {
                    "validation": results["validation_drive_level"]["failing_drive_count"],
                    "test": results["test_drive_level"]["failing_drive_count"],
                },
                "feature_count": len(feature_columns),
            }
            results["action_tiers"] = {
                tier: (
                    None
                    if threshold is None
                    else {
                        "threshold": threshold,
                        "target_precision": tier_targets[tier],
                        "validation": drive_level_metrics(
                            val_drive_ids, y_val, val_scores, threshold
                        ),
                        "test": drive_level_metrics(
                            test_ids["drive_id"].to_numpy(), y_test, test_scores, threshold
                        ),
                    }
                )
                for tier, threshold in tier_thresholds.items()
            }

            # Logistic Regression sanity baseline (docs/project_plan.md Phase 6
            # Model Candidates: "Interpretable sanity baseline") - confirms the
            # primary model is actually adding value over a simple linear model,
            # rather than assuming it. Never used for production decisions.
            #
            # Fitted on at most `diagnostics.baseline_max_rows` rows: sklearn's
            # StandardScaler and LogisticRegression each force a float64 copy
            # of their input, so an uncapped fit needs ~2x the (already
            # fleet-scale) training matrix on top of the primary model's own.
            # This is a non-production sanity check, so a bounded, fixed-seed
            # sample is the right trade; the cap is high enough that smaller
            # datasets are fitted in full and unaffected.
            t0 = time.perf_counter()
            x_baseline, y_baseline = _subsample_rows(
                x_train, y_train, max_rows=diagnostics_cfg.get("baseline_max_rows", 1_000_000)
            )
            baseline_model = train_logistic_regression_baseline(x_baseline, y_baseline)
            baseline_val_scores = predict_proba_positive(baseline_model, x_val)
            baseline_test_scores = predict_proba_positive(baseline_model, x_test)
            # Figure data for `make plots` (the paper's ROC and variable-importance
            # figures): ROC curves of the primary model and the baseline on test,
            # row level and drive level, and the primary model's gain importance.
            test_drive_ids = test_ids["drive_id"].to_numpy()
            results["roc_curves"] = {
                "lightgbm_test_row": roc_points(y_test, test_scores),
                "logistic_test_row": roc_points(y_test, baseline_test_scores),
                "lightgbm_test_drive": roc_points(
                    *drive_level_table(test_drive_ids, y_test, test_scores)
                ),
                "logistic_test_drive": roc_points(
                    *drive_level_table(test_drive_ids, y_test, baseline_test_scores)
                ),
            }
            if model_type == "lightgbm":
                importance_gains = model.booster_.feature_importance("gain")
            else:
                importance_gains = np.asarray(model.feature_importances_)
            results["feature_importance"] = gain_importance(feature_columns, importance_gains)
            results["logistic_regression_baseline"] = {
                "validation_auprc": compute_auprc(y_val, baseline_val_scores),
                "test_auprc": compute_auprc(y_test, baseline_test_scores),
                "train_row_count": int(x_baseline.shape[0]),
            }
            _log_stage("baseline_trained", t0, baseline_row_count=int(x_baseline.shape[0]))
            del x_baseline, y_baseline
            gc.collect()
            mlflow.log_metric(
                "baseline_validation_auprc",
                results["logistic_regression_baseline"]["validation_auprc"],
            )
            mlflow.log_metric(
                "baseline_test_auprc", results["logistic_regression_baseline"]["test_auprc"]
            )

            # Subsampled SMOTE comparison (docs/dataset_strategy.md section 15
            # Imbalance Mitigations #3) - a SECOND, comparison-only model, never
            # used in place of the primary (class-weighted) one. SMOTE-resampled
            # data is already balanced, so class weighting is turned off for
            # this comparison fit to avoid double-compensating.
            smote_cfg = model_config.get("smote_comparison", {})
            if smote_cfg.get("enabled", False):
                x_smote, y_smote = apply_smote(
                    x_train,
                    y_train,
                    subsample_fraction=smote_cfg.get("subsample_fraction", 1.0),
                    seed=smote_cfg.get("seed", 0),
                )
                smote_params = dict(model_params)
                if model_type == "lightgbm":
                    smote_params["is_unbalance"] = False
                smote_model = (
                    train_xgboost(x_smote, y_smote, params=smote_params)
                    if model_type == "xgboost"
                    else train_lightgbm(x_smote, y_smote, params=smote_params)
                )
                smote_val_auprc = compute_auprc(y_val, predict_proba_positive(smote_model, x_val))
                class_weighting_val_auprc = results["validation_metrics"]["auprc"]
                results["smote_comparison"] = {
                    "smote_validation_auprc": smote_val_auprc,
                    "class_weighting_validation_auprc": class_weighting_val_auprc,
                    "recommendation": (
                        "smote"
                        if smote_val_auprc > class_weighting_val_auprc
                        else "class_weighting"
                    ),
                    "smote_resampled_row_count": len(y_smote),
                }
                mlflow.log_metric("smote_validation_auprc", smote_val_auprc)
                logger.info("smote_comparison", **results["smote_comparison"])

            # Warning lead time (docs/dataset_strategy.md section 15): restrict
            # to drive-days belonging to drives with a genuine failure event
            # before asking "how early did the score cross threshold".
            test_with_scores = test_ids.with_columns(pl.Series("_p_fail_score", test_scores))
            failing_test_rows = test_with_scores.filter(
                pl.col("event_type").is_in(list(FAILURE_EVENT_TYPES))
            )
            results["test_warning_lead_time"] = compute_warning_lead_time_days(
                failing_test_rows,
                score_column="_p_fail_score",
                threshold=threshold_result["threshold"],
            )
            del test_with_scores, failing_test_rows
            gc.collect()

            mlflow.log_params({"horizon_days": horizon_days, **model_params})
            mlflow.log_metric("validation_auprc", results["validation_metrics"]["auprc"])
            mlflow.log_metric("validation_precision", results["validation_metrics"]["precision"])
            mlflow.log_metric("validation_recall", results["validation_metrics"]["recall"])
            mlflow.log_metric("test_auprc", results["test_metrics"]["auprc"])
            for split_key in ("validation", "test"):
                drive_level = results[f"{split_key}_drive_level"]
                for metric in ("precision", "recall", "auprc"):
                    mlflow.log_metric(f"{split_key}_drive_{metric}", drive_level[metric])
            mlflow.log_metric(
                "test_expected_calibration_error",
                results["test_metrics"]["calibration"]["expected_calibration_error"],
            )
            mlflow.log_metric(
                "test_brier_score", results["test_metrics"]["calibration"]["brier_score"]
            )
            if results["test_warning_lead_time"]["mean_lead_time_days"] is not None:
                mlflow.log_metric(
                    "test_mean_warning_lead_time_days",
                    results["test_warning_lead_time"]["mean_lead_time_days"],
                )
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.lightgbm.log_model(model, name="model")
            mlflow.log_param("two_stage", two_stage_model is not None)
            if two_stage_model is not None:
                log_two_stage(two_stage_model, results["two_stage"])

            # SHAP global feature importance (docs/design_goal.md "Explainability
            # by default"): background sample keeps TreeExplainer fast even on a
            # large training set.
            # Explained on at most `diagnostics.shap_max_rows` rows: the global
            # importance below is a mean of |shap value| per feature, which
            # converges long before millions of rows, while TreeExplainer
            # materializes a full (rows x features) float64 array to get there.
            # Optional (configs/model.yaml diagnostics.shap_enabled, off by
            # default): slow on fleet-scale data, and SHAP's additivity check
            # fails outright when the model has extreme leaf values.
            feature_importance: list[dict[str, float | str]] | None = None
            if diagnostics_cfg.get("shap_enabled", False):
                t0 = time.perf_counter()
                rng = np.random.default_rng(0)
                background_size = min(SHAP_BACKGROUND_SAMPLE_SIZE, x_train.shape[0])
                background = x_train[
                    rng.choice(x_train.shape[0], size=background_size, replace=False)
                ]
                x_shap, _ = _subsample_rows(
                    x_val, y_val, max_rows=diagnostics_cfg.get("shap_max_rows", 200_000)
                )
                explainer = build_explainer(model, background)
                shap_values = compute_shap_values(explainer, x_shap)
                feature_importance = global_feature_importance(shap_values, feature_columns)
                results["shap_global_feature_importance"] = feature_importance[:20]
                _log_stage("shap_computed", t0, explained_row_count=int(x_shap.shape[0]))
                del x_shap, shap_values
                gc.collect()
            else:
                logger.info("shap_skipped", reason="diagnostics.shap_enabled is false")

            audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
            audit_dir.mkdir(parents=True, exist_ok=True)
            report_path = audit_dir / "model_evaluation_report.json"
            report_path.write_text(json.dumps(results, indent=2, default=str))

            # Read back by `make score-fleet` from this run's artifacts.
            action_tiers_path = audit_dir / "action_tiers.json"
            action_tiers_path.write_text(json.dumps(tier_thresholds, indent=2))
            mlflow.log_artifact(str(action_tiers_path))

            shap_report_path = audit_dir / "shap_feature_importance.json"
            if feature_importance is not None:
                shap_report_path.write_text(json.dumps(feature_importance, indent=2, default=str))
                mlflow.log_artifact(str(shap_report_path))
            else:
                # Never leave an earlier run's SHAP report next to this
                # run's evaluation report - generate_performance_plots.py
                # would plot it as if it described this model.
                shap_report_path.unlink(missing_ok=True)

            model_card = build_model_card(
                horizon_days=horizon_days,
                model_params=model_params,
                model_version=model_config["version"],
                model_type=model_type,
                feature_registry_version=features_config["version"],
                dataset_version=(
                    dataset_version_record["version_id"] if dataset_version_record else None
                ),
                feature_columns=feature_columns,
                threshold_result=threshold_result,
                validation_metrics=results["validation_metrics"],
                test_metrics=results["test_metrics"],
                drive_level_metrics={
                    "validation": results["validation_drive_level"],
                    "test": results["test_drive_level"],
                },
                action_tiers=results["action_tiers"],
                precision_at_recall=results["precision_at_recall"],
                two_stage=results.get("two_stage"),
                shap_top_features=(
                    feature_importance[:20] if feature_importance is not None else None
                ),
                train_row_count=x_train.shape[0],
                test_warning_lead_time=results["test_warning_lead_time"],
                logistic_regression_baseline=results["logistic_regression_baseline"],
                smote_comparison=results.get("smote_comparison"),
            )
            model_cards_dir = Path(data_config["audit_dir"]) / "model_cards"
            model_cards_dir.mkdir(parents=True, exist_ok=True)
            model_card_stem = f"v{model_config['version']}_h{horizon_days}d"
            model_card_json_path = model_cards_dir / f"{model_card_stem}.json"
            model_card_md_path = model_cards_dir / f"{model_card_stem}.md"
            model_card_json_path.write_text(json.dumps(model_card, indent=2, default=str))
            model_card_md_path.write_text(render_model_card_markdown(model_card))
            mlflow.log_artifact(str(model_card_json_path))
            mlflow.log_artifact(str(model_card_md_path))

            logger.info(
                "model_trained", horizon_days=horizon_days, train_row_count=x_train.shape[0]
            )
            logger.info("threshold_tuned", **threshold_result)
            logger.info("validation_metrics", **results["validation_metrics"])
            logger.info("test_metrics", **results["test_metrics"])
            logger.info("validation_drive_level", **results["validation_drive_level"])
            logger.info("test_drive_level", **results["test_drive_level"])
            for row in results["precision_at_recall"]:
                if row["threshold"] is None:
                    logger.info("precision_at_recall", target_recall=row["target_recall"])
                    continue
                logger.info(
                    "precision_at_recall",
                    target_recall=row["target_recall"],
                    threshold=row["threshold"],
                    validation_precision=row["validation"]["precision"],
                    test_precision=row["test"]["precision"],
                    test_recall=row["test"]["recall"],
                    test_caught=row["test"]["caught_drive_count"],
                    test_false_alarms=row["test"]["false_alarm_drive_count"],
                    test_lift=row["test"]["lift"],
                    # Same alerts on test sets with more failing drives.
                    test_precision_at_failure_rate=row["test"]["precision_at_failure_rate"],
                )
            for row in results.get("two_stage", {}).get("stage1_precision_at_recall", []):
                if row["threshold"] is None:
                    continue
                logger.info(
                    "stage1_alone_precision_at_recall",
                    target_recall=row["target_recall"],
                    test_precision=row["test"]["precision"],
                    test_recall=row["test"]["recall"],
                    test_caught=row["test"]["caught_drive_count"],
                    test_false_alarms=row["test"]["false_alarm_drive_count"],
                )
            logger.info("data_build", **results["data_build"])
            for tier, info in results["action_tiers"].items():
                if info is None:
                    logger.info("action_tier", tier=tier, reachable=False)
                    continue
                logger.info(
                    "action_tier",
                    tier=tier,
                    threshold=info["threshold"],
                    target_precision=info["target_precision"],
                    validation_precision=info["validation"]["precision"],
                    validation_recall=info["validation"]["recall"],
                    test_precision=info["test"]["precision"],
                    test_recall=info["test"]["recall"],
                )
            logger.info("logistic_regression_baseline", **results["logistic_regression_baseline"])
            logger.info("test_warning_lead_time", **results["test_warning_lead_time"])
            logger.info("evaluation_report_written", path=str(report_path))
            if feature_importance is not None:
                logger.info("top_shap_features", features=feature_importance[:5])
                logger.info("shap_feature_importance_written", path=str(shap_report_path))
            logger.info("model_card_written", path=str(model_card_md_path))
    finally:
        if args.keep_work_dir:
            logger.info("work_dir_kept", path=str(work_dir))
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
