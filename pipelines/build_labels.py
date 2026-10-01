"""Entry point for `make build-labels`.

Reads gold features and silver drive metadata, classifies removal event
types, computes leakage-free failure labels for every configured horizon,
applies the chronological (+ optional drive-holdout) split, and writes the
label table, split assignment, and imbalance report.

The label-computation step (`_stage_compute`) runs in its own subprocess,
for the same reason batches are isolated into their own subprocesses in
pipelines/build_gold_features.py: Polars is built on jemalloc, which on
64-bit Linux defaults to retaining freed virtual memory for reuse rather
than returning it to the OS, so RLIMIT_AS (this pipeline's memory cap)
tracks a process's cumulative high-water mark, not what's currently live.
`_stage_compute` runs several sequential operations one after another
(label computation across every configured horizon, three split passes,
Pandera validation) against a label table that is ~3x the drive-day count
(one row per horizon) - for one real quarter, ~92M rows. Doing all of that
in one un-isolated process risks the same "each step's high-water mark
stacks on the last" failure already seen in build_gold_features.py and
train_model.py.

Unlike build_gold_features.py's pivot, this does NOT need drive-based
batching on top of that: every join here is keyed on drive_id alone
against drive_metadata's one-row-per-drive table (so it can never fan
out), and the label table never widens beyond ~11 narrow columns - there
is no wide/expanding transform for a batch boundary to bound. A single
subprocess boundary around the whole compute step is enough.

Once `_stage_compute` writes the label table to disk and exits, the
top-level orchestrator (`main()` with no `--stage`) computes the dataset
version record and imbalance report from narrow (2-3 column) projections
scanned back off that file - never re-reading the full label table into
this process either.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
)
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_labels_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def _require_inputs(features_path: Path, metadata_path: Path) -> None:
    if not features_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing gold features or drive metadata; run `make build-features` first."
        )


def _stage_compute(labels_path: Path) -> None:
    """Subprocess stage: computes leakage-free labels for every configured
    horizon, applies every split, validates the result, and writes it
    directly to `labels_path`. See the module docstring for why this runs
    in its own process."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    _require_inputs(features_path, metadata_path)

    # Only drive_id/date/source_dataset are ever needed from the gold
    # features table here - never read it in full. It's ~196 columns and
    # ~10GB for one month of real data; reading it whole just to select 3
    # narrow columns out of it would (as with canonical_long/drive_day in
    # build_gold_features.py) mean an unused ~10GB reference sitting alive
    # in this function's own locals for the rest of the pipeline.
    t0 = time.perf_counter()
    features_columns = pl.scan_parquet(features_path).collect_schema().names()
    drive_day_columns = [
        c for c in ("drive_id", "date", "source_dataset") if c in features_columns
    ]
    drive_days = pl.scan_parquet(features_path).select(drive_day_columns).collect()
    _log_stage("drive_days_read", t0, row_count=drive_days.height)

    t0 = time.perf_counter()
    drive_metadata = pl.read_parquet(metadata_path)
    _log_stage("drive_metadata_read", t0, row_count=drive_metadata.height)

    as_of_date = drive_days["date"].max()
    drive_metadata = classify_event_types(drive_metadata, as_of_date=as_of_date)

    t0 = time.perf_counter()
    labels = compute_labels(
        drive_days, drive_metadata, horizons_days=model_config["horizons_days"]
    )
    _log_stage("labels_computed", t0, row_count=labels.height)

    splits_cfg = model_config["splits"]
    t0 = time.perf_counter()
    labels = add_chronological_split(
        labels,
        train_end=splits_cfg["train_end"],
        validation_end=splits_cfg["validation_end"],
    )
    labels = apply_drive_level_holdout(labels)
    if "source_dataset" in drive_days.columns:
        labels = labels.join(
            drive_days.select(["drive_id", "date", "source_dataset"]),
            on=["drive_id", "date"],
            how="left",
        )
    labels = apply_vendor_holdout(labels)
    _log_stage("splits_applied", t0, row_count=labels.height)

    t0 = time.perf_counter()
    labels = validate_gold_labels(labels, horizons_days=model_config["horizons_days"])
    _log_stage("labels_validated", t0)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    labels.write_parquet(labels_path, compression="zstd")
    _log_stage("labels_written", t0, row_count=labels.height)


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage - a new
    process, and therefore a new address space, regardless of what the
    calling process's allocator has retained. See the module docstring."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["compute"])
    parser.add_argument("--labels-path", type=Path)
    args = parser.parse_args()

    if args.stage == "compute":
        _stage_compute(args.labels_path)
        return

    # No --stage: the top-level orchestrator. It never itself reads gold
    # features, drive_metadata, or the full label table - only the compute
    # subprocess it spawns does - so its own address space stays flat
    # regardless of fleet size.
    configure_logging()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    _require_inputs(features_path, metadata_path)

    labels_dir = gold_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / "part.parquet"

    _run_stage("--stage", "compute", "--labels-path", str(labels_path))

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
