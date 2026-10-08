"""Entry point for `make build-labels`.

Reads gold features and silver drive metadata, classifies removal event
types, computes leakage-free failure labels for every configured horizon,
applies the chronological (+ optional drive-holdout) split, and writes the
label table, split assignment, and imbalance report.

Labels are computed and split in batches of whole drives, each in its own
subprocess, matching pipelines/build_gold_features.py's architecture.

This started as a single subprocess covering the whole pipeline, then two
subprocesses (one for label computation, one for split application) once
that single subprocess crashed against real Q1 data right after
`labels_computed` (91,792,452 rows) - the theory being that whatever
high-water mark computing those labels left behind (Polars is built on
jemalloc, which on 64-bit Linux retains freed virtual memory for reuse
rather than returning it to the OS, so RLIMIT_AS tracks a process's
cumulative allocations, not what's currently live) was compounding with
the next step's own needs.

That theory turned out to be wrong for this crash: a brand-new `_stage_split`
subprocess, having done nothing but read the raw label file back, crashed
on essentially the same transformation almost immediately - proving the
split-application step itself, independent of any prior process history,
needs more memory than fits at ~92M rows (one real quarter, 3 horizons).
Process isolation cannot fix a real per-operation memory requirement -
only bounding the amount of data any one call to that operation touches
can, which is exactly what pipelines/build_gold_features.py's per-drive
batching does for its pivot. So this pipeline now does the same: batch by
`hash(drive_id)` (every join here - compute_labels, the drive-holdout/
vendor-holdout assignment - is keyed on drive_id alone against a
one-row-per-drive table, so it can never fan out and a drive's full
history always lands in one batch), compute labels AND apply every split
for one batch at a time, in its own subprocess, and combine the batch
outputs at the end.

The one thing that must be computed once, globally, rather than per
batch: which ~10% of drives are held out (`apply_drive_level_holdout`).
Sampling independently per batch would give each batch its own random
~10% rather than sharing one fleet-wide holdout set, so
`_stage_prepare` samples it once (cheaply - from drive_metadata's small
one-row-per-drive table, not the full label table) and every batch is
given the same precomputed set.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
import yaml

from src.labels.dataset_version import build_dataset_version_record, write_dataset_version
from src.labels.event_types import classify_event_types
from src.labels.imbalance import class_distribution_report, compute_scale_pos_weight
from src.labels.labeling import compute_labels
from src.labels.schema_validation import validate_gold_labels
from src.labels.splits import (
    add_chronological_split,
    apply_drive_level_holdout,
    apply_vendor_holdout,
    sample_holdout_drive_ids,
)
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

#: Target drive-day rows per label batch. The label table itself is ~3x
#: this per batch (one row per horizon), still comfortably bounded.
DEFAULT_BATCH_TARGET_ROWS = 2_000_000

PREPARE_MANIFEST_NAME = "prepare_manifest.json"

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_labels_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def _require_inputs(features_path: Path, metadata_path: Path) -> None:
    if not features_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing gold features or drive metadata; run `make build-features` first."
        )


def _load_configs() -> tuple[dict, dict]:
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    return data_config, model_config


def _stage_prepare(tmp_dir: Path) -> None:
    """Subprocess stage: determines batch count and samples the one
    fleet-wide drive-holdout set every batch will share (see the module
    docstring for why this must be global, not per batch).

    Both are cheap: drive_metadata is one row per drive (351k for one
    real quarter, trivial to hold and sample from directly), and the
    batch count only needs gold features' row count, which resolves from
    Parquet footer metadata without reading any column data."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config, _ = _load_configs()

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    _require_inputs(features_path, metadata_path)

    t0 = time.perf_counter()
    drive_day_row_count = pl.scan_parquet(features_path).select(pl.len()).collect().item()
    _log_stage("drive_day_row_count_determined", t0, row_count=drive_day_row_count)

    t0 = time.perf_counter()
    drive_metadata = pl.read_parquet(metadata_path)
    holdout_drive_ids = sample_holdout_drive_ids(drive_metadata["drive_id"])
    _log_stage(
        "holdout_drive_ids_sampled",
        t0,
        drive_count=drive_metadata.height,
        holdout_count=len(holdout_drive_ids),
    )

    batch_target_rows = int(
        data_config.get("resource_limits", {}).get(
            "label_batch_target_rows", DEFAULT_BATCH_TARGET_ROWS
        )
    )
    n_batches = max(1, math.ceil(drive_day_row_count / batch_target_rows))
    (tmp_dir / PREPARE_MANIFEST_NAME).write_text(
        json.dumps({"n_batches": n_batches, "holdout_drive_ids": holdout_drive_ids})
    )


def _stage_batch(tmp_dir: Path, batch_index: int, batch_out_path: Path) -> None:
    """Subprocess stage: computes labels for every configured horizon and
    applies every split, for one batch of whole drives - every gold
    features/drive_metadata row whose drive_id hashes to `batch_index` -
    and writes the result to `batch_out_path`."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config, model_config = _load_configs()

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    _require_inputs(features_path, metadata_path)

    manifest = json.loads((tmp_dir / PREPARE_MANIFEST_NAME).read_text())
    n_batches = manifest["n_batches"]
    holdout_drive_ids = manifest["holdout_drive_ids"]
    batch_of_drive = pl.col("drive_id").hash(seed=0) % n_batches

    # Only drive_id/date/source_dataset are ever needed from the gold
    # features table here - never read it in full, and only this batch's
    # own slice at that. It's ~196 columns and ~10GB for one month of
    # real data; a narrow, batch-filtered scan avoids ever materializing
    # any of that.
    t0 = time.perf_counter()
    features_columns = pl.scan_parquet(features_path).collect_schema().names()
    drive_day_columns = [c for c in ("drive_id", "date", "source_dataset") if c in features_columns]
    drive_days = (
        pl.scan_parquet(features_path)
        .select(drive_day_columns)
        .filter(batch_of_drive == batch_index)
        .collect()
    )
    _log_stage(
        "drive_days_read",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=drive_days.height,
    )

    # drive_metadata is small enough (one row per drive) to read whole
    # rather than filtering it to the batch too - classify_event_types'
    # as_of_date needs the fleet-wide max date anyway (drive_days here is
    # already just this batch, so its own max would be wrong if this
    # batch happens not to include the fleet's most-recently-observed
    # drive), and re-reading 351k small rows per batch is cheap regardless.
    t0 = time.perf_counter()
    drive_metadata = pl.read_parquet(metadata_path)
    as_of_date = pl.scan_parquet(features_path).select(pl.col("date").max()).collect().item()
    drive_metadata = classify_event_types(drive_metadata, as_of_date=as_of_date)
    _log_stage("drive_metadata_read", t0, batch_index=batch_index, row_count=drive_metadata.height)

    # An empty batch (drive_days.is_empty()) is a normal, expected outcome
    # of hashing rows into batches (see the analogous case in
    # build_silver.py) - every step below is 0-row-safe (verified
    # directly: compute_labels/the split functions/validate_gold_labels
    # all produce a correctly-shaped, empty-but-valid result), so it
    # needs no special case here.
    t0 = time.perf_counter()
    labels = compute_labels(drive_days, drive_metadata, horizons_days=model_config["horizons_days"])
    _log_stage(
        "labels_computed",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=labels.height,
    )

    splits_cfg = model_config["splits"]
    t0 = time.perf_counter()
    labels = add_chronological_split(
        labels,
        train_end=splits_cfg["train_end"],
        validation_end=splits_cfg["validation_end"],
        sealed_start=splits_cfg.get("sealed_start"),
    )
    labels = apply_drive_level_holdout(labels, holdout_drive_ids=holdout_drive_ids)
    if "source_dataset" in drive_days.columns:
        labels = labels.join(
            drive_days.select(["drive_id", "date", "source_dataset"]),
            on=["drive_id", "date"],
            how="left",
        )
    labels = apply_vendor_holdout(labels)
    _log_stage(
        "splits_applied",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=labels.height,
    )

    t0 = time.perf_counter()
    labels = validate_gold_labels(labels, horizons_days=model_config["horizons_days"])
    _log_stage("labels_validated", t0, batch_index=batch_index, batch_count=n_batches)

    batch_out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    labels.write_parquet(batch_out_path, compression="zstd")
    _log_stage(
        "batch_written",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=labels.height,
    )


def _stage_finalize(batch_paths: list[Path], labels_path: Path) -> None:
    """Subprocess stage: concatenates the per-batch label outputs - all
    sharing an identical schema by construction - into the final label
    table, via a lazy scan+sink rather than an eager concat, so combining
    them stays streaming."""
    configure_logging()
    apply_memory_limit_from_config()
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pl.concat([pl.scan_parquet(p) for p in batch_paths], how="vertical").sink_parquet(
        labels_path, compression="zstd"
    )
    row_count = pl.scan_parquet(labels_path).select(pl.len()).collect().item()
    _log_stage("labels_combined", t0, row_count=row_count)


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage - a new
    process, and therefore a new address space, regardless of what the
    calling process's allocator has retained. See the module docstring."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "batch", "finalize"])
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--batch-out-path", type=Path)
    parser.add_argument("--batch-paths", type=Path, nargs="*")
    parser.add_argument("--labels-path", type=Path)
    args = parser.parse_args()

    if args.stage == "prepare":
        _stage_prepare(args.tmp_dir)
        return
    if args.stage == "batch":
        _stage_batch(args.tmp_dir, args.batch_index, args.batch_out_path)
        return
    if args.stage == "finalize":
        _stage_finalize(args.batch_paths, args.labels_path)
        return

    # No --stage: the top-level orchestrator. It never itself reads gold
    # features, drive_metadata, or the full label table - only the
    # subprocesses it spawns do - so its own address space stays flat
    # regardless of fleet size or batch count.
    configure_logging()
    data_config, model_config = _load_configs()
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    _require_inputs(features_path, metadata_path)

    labels_dir = gold_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / "part.parquet"

    tmp_dir = Path(tempfile.mkdtemp(prefix="build_labels_"))
    try:
        _run_stage("--stage", "prepare", "--tmp-dir", str(tmp_dir))
        manifest = json.loads((tmp_dir / PREPARE_MANIFEST_NAME).read_text())
        n_batches = manifest["n_batches"]

        batch_paths = []
        for batch_index in range(n_batches):
            batch_out_path = tmp_dir / f"labels_batch_{batch_index}.parquet"
            _run_stage(
                "--stage",
                "batch",
                "--tmp-dir",
                str(tmp_dir),
                "--batch-index",
                str(batch_index),
                "--batch-out-path",
                str(batch_out_path),
            )
            batch_paths.append(batch_out_path)
            logger.info("build_labels_batch_done", batch_index=batch_index, batch_count=n_batches)

        _run_stage(
            "--stage",
            "finalize",
            "--batch-paths",
            *[str(p) for p in batch_paths],
            "--labels-path",
            str(labels_path),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    audit_root = Path(data_config["audit_dir"])
    t0 = time.perf_counter()
    # Only date/split are needed for the dataset version record's
    # date_range/split_counts (and row_count, which doesn't depend on
    # which columns are present) - not the full ~11-column, ~92M-row (one
    # real quarter) label table.
    version_columns = pl.scan_parquet(labels_path).select(["date", "split"]).collect()
    dataset_version_record = build_dataset_version_record(
        features_path=features_path,
        labels=version_columns,
        labels_path=labels_path,
        feature_registry_version=features_config["version"],
        model_config_version=model_config["version"],
    )
    dataset_version_path = write_dataset_version(dataset_version_record, audit_root)
    _log_stage("dataset_version_computed", t0)

    t0 = time.perf_counter()
    # Same reasoning: the imbalance report only ever touches
    # horizon_days/split/label.
    imbalance_columns = (
        pl.scan_parquet(labels_path).select(["horizon_days", "split", "label"]).collect()
    )
    imbalance_report = {
        "class_distribution": class_distribution_report(imbalance_columns),
        "scale_pos_weight_by_horizon": {
            h: compute_scale_pos_weight(imbalance_columns.filter(pl.col("horizon_days") == h))
            for h in model_config["horizons_days"]
        },
    }
    _log_stage("imbalance_report_computed", t0)
    audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "label_imbalance_report.json").write_text(
        json.dumps(imbalance_report, indent=2, default=str)
    )

    row_count = pl.scan_parquet(labels_path).select(pl.len()).collect().item()
    logger.info("gold_labels_written", row_count=row_count, path=str(labels_dir))
    logger.info(
        "label_imbalance_report_written", path=str(audit_dir / "label_imbalance_report.json")
    )
    logger.info(
        "dataset_version_written",
        version_id=dataset_version_record["version_id"],
        path=str(dataset_version_path),
    )


if __name__ == "__main__":
    main()
