"""Entry point for `make build-silver`.

Reads every Bronze partition, normalizes identifiers, harmonizes SMART
attributes into the canonical long schema, computes telemetry-gap/staleness
features and drive-level feature maturity, runs data-quality checks, and
writes the Silver layer plus a quality report to data/audit/.

Tonight's crashes across three separate pipelines (this one,
build_gold_features.py, train_model.py) all trace to the same lesson:
treating a multi-file, multi-stage transformation as one long streaming
Polars pipeline does not reliably stay memory-bounded once real data scale
and multiple source files are involved - a wide `.rolling()`/join, a
multi-source `diagonal_relaxed` concat, or simply more row groups than an
earlier run can each break the streaming guarantee in ways that are hard to
predict and easy to reintroduce. The pattern that has held up every time
instead is: partition the work into bounded, independent batches, each
checkpointed to its own file, each processed in its own subprocess so nothing
accumulates across batches (Polars is built on jemalloc, which on 64-bit
Linux retains freed virtual memory for reuse rather than returning it to the
OS - see the longer explanation in build_gold_features.py's module
docstring - so only a process boundary reliably resets a pipeline's high-
water mark).

This pipeline now follows that same shape, partitioned by drive rather than
by month: `normalize_identifiers`, `derive_failure_date`
(`.over("drive_id")`), and `compute_telemetry_gaps` (per-drive shift/rolling
windows) are all per-drive operations with no cross-drive dependency, but
`compute_telemetry_gaps` DOES need a single drive's full history in one
place - a drive observed in both January and February must not have its
day-over-day gap computed against only one month's rows. Partitioning by
month, the more obvious split given Bronze's own year/month layout, would
silently corrupt that calculation for any drive spanning a partition
boundary; partitioning by `hash(drive_id)` keeps a drive's entire history
together regardless of which Bronze files it appears in.
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

from src.logging_config import configure_logging, get_logger
from src.preprocess.failure_events import derive_failure_date
from src.preprocess.feature_maturity import build_drive_metadata
from src.preprocess.identifiers import normalize_identifiers
from src.preprocess.quality_checks import run_all_checks
from src.preprocess.smart_mapping import melt_smart_attributes
from src.preprocess.telemetry_gaps import compute_telemetry_gaps
from src.resource_limits import apply_memory_limit_from_config

CONFIG_PATH = Path("configs/data.yaml")

#: Target rows per drive-day batch. Mirrors
#: resource_limits.feature_batch_target_rows in configs/data.yaml, which
#: governs the analogous batching in build_gold_features.py.
DEFAULT_BATCH_TARGET_ROWS = 2_000_000

#: Manifest/result filenames inside the run's temp directory - how the
#: subprocess stages hand information back to the orchestrator, since they
#: can't return Python values across a process boundary.
PREPARE_MANIFEST_NAME = "prepare_manifest.json"
FINALIZE_RESULT_NAME = "finalize_result.json"

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_silver_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def _require_bronze_files(bronze_root: Path) -> list[Path]:
    bronze_files = sorted(bronze_root.glob("**/*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze Parquet files found under {bronze_root}")
    return bronze_files


def _load_configs() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _stage_prepare(bronze_root: Path, tmp_dir: Path) -> None:
    """Subprocess stage: determines how many drive batches to split into.

    A plain row count, not a scan of any actual column data - Parquet
    stores each row group's row count in its footer metadata, so this
    resolves from metadata alone regardless of how large the files are,
    unlike every other operation that touches these files."""
    configure_logging()
    apply_memory_limit_from_config()
    config = _load_configs()
    bronze_files = _require_bronze_files(bronze_root)
    logger.info("build_silver_bronze_files_found", file_count=len(bronze_files))

    t0 = time.perf_counter()
    total_rows = (
        pl.concat([pl.scan_parquet(p) for p in bronze_files], how="diagonal_relaxed")
        .select(pl.len())
        .collect()
        .item()
    )
    _log_stage("bronze_row_count_determined", t0, row_count=total_rows)

    batch_target_rows = int(
        config.get("resource_limits", {}).get(
            "silver_batch_target_rows", DEFAULT_BATCH_TARGET_ROWS
        )
    )
    n_batches = max(1, math.ceil(total_rows / batch_target_rows))
    (tmp_dir / PREPARE_MANIFEST_NAME).write_text(
        json.dumps({"n_batches": n_batches, "total_rows": total_rows})
    )


def _stage_batch(
    bronze_root: Path,
    batch_index: int,
    n_batches: int,
    drive_day_out_path: Path,
    canonical_out_path: Path,
) -> None:
    """Subprocess stage: processes one drive batch - every Bronze row whose
    *normalized* drive_id hashes to `batch_index` - through normalization,
    failure-date derivation, and telemetry-gap computation, sinking the
    result to `drive_day_out_path`; then melts that batch's own drive_day
    into canonical (long) form and sinks that to `canonical_out_path`.

    The melt happens here, per batch, rather than once in `_stage_finalize`
    against the full recombined drive_day table - which is what this
    function used to do, and which crashed against real Q1-scale data even
    though building the combined drive_day table itself, and the
    (row-count-*shrinking*) drive_metadata aggregate over it, both
    succeeded. The melt is the one step here that *expands* row count
    (~5x, one row per SMART attribute per drive-day) rather than shrinking
    or preserving it, and that expansion is what broke streaming once
    Bronze grew to multiple files/3x the row count - so it gets the same
    per-batch treatment as everything else that touched the full dataset
    and didn't scale. `_stage_finalize` now only ever concatenates already-
    melted, identically-shaped batch outputs, never re-melts anything.

    Normalizing before hashing matters for correctness, not just style: the
    same physical drive can appear with inconsistent casing/whitespace
    across different Bronze files (e.g. different ingest months), and
    hashing the raw, un-normalized drive_id could split what should be a
    single drive's rows across two different batches - which would corrupt
    compute_telemetry_gaps's day-over-day and 30-day-coverage calculations
    for that drive, since each half would see only part of its history.
    Every Bronze file is re-scanned and re-normalized for every batch
    (rather than normalizing once and writing an intermediate), the same
    redundant-rescan shape as build_gold_features.py's `_write_wide_batches`
    - normalization is cheap and purely row-wise, so paying it once per
    batch is far cheaper than the alternative of introducing another
    checkpoint file just to avoid it."""
    configure_logging()
    apply_memory_limit_from_config()
    config = _load_configs()
    bronze_files = _require_bronze_files(bronze_root)

    lf = pl.concat([pl.scan_parquet(p) for p in bronze_files], how="diagonal_relaxed")
    lf = normalize_identifiers(lf)
    lf = lf.filter(pl.col("drive_id").hash(seed=0) % n_batches == batch_index)
    lf = derive_failure_date(lf)
    gap_cfg = config["telemetry_gap"]
    lf = compute_telemetry_gaps(
        lf, short_gap_days=gap_cfg["short_gap_days"], stale_gap_days=gap_cfg["stale_gap_days"]
    )

    t0 = time.perf_counter()
    lf.sink_parquet(drive_day_out_path, compression="zstd")
    drive_day_row_count = pl.scan_parquet(drive_day_out_path).select(pl.len()).collect().item()
    _log_stage(
        "drive_day_batch_written",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=drive_day_row_count,
    )

    t0 = time.perf_counter()
    melt_smart_attributes(
        pl.scan_parquet(drive_day_out_path), id_columns=["drive_id", "date"]
    ).sink_parquet(canonical_out_path, compression="zstd")
    canonical_row_count = pl.scan_parquet(canonical_out_path).select(pl.len()).collect().item()
    _log_stage(
        "canonical_batch_written",
        t0,
        batch_index=batch_index,
        batch_count=n_batches,
        row_count=canonical_row_count,
    )


def _stage_finalize(
    drive_day_batch_paths: list[Path],
    canonical_batch_paths: list[Path],
    drive_day_path: Path,
    canonical_path: Path,
    metadata_path: Path,
    quality_report_path: Path,
    result_path: Path,
) -> None:
    """Subprocess stage: combines the per-batch drive_day outputs into the
    final drive_day table and the per-batch canonical (already-melted)
    outputs into the final canonical_telemetry table, then builds
    drive_metadata and runs quality checks against the combined drive_day.

    Both combining concats are `vertical` (not `diagonal_relaxed`) concats
    of files that all share an identical schema by construction (every
    batch ran through the same pipeline), and every step here reads back
    via `pl.scan_parquet`/sinks rather than holding an eager DataFrame.
    Unlike an earlier version of this function, canonical_telemetry is
    never built by melting the combined drive_day here - see
    `_stage_batch`'s docstring for why that row-expanding operation has to
    happen per batch instead."""
    configure_logging()
    apply_memory_limit_from_config()
    config = _load_configs()

    drive_day_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pl.concat([pl.scan_parquet(p) for p in drive_day_batch_paths], how="vertical").sink_parquet(
        drive_day_path, compression="zstd"
    )
    drive_day_row_count = pl.scan_parquet(drive_day_path).select(pl.len()).collect().item()
    _log_stage("drive_day_combined", t0, row_count=drive_day_row_count)

    t0 = time.perf_counter()
    pl.concat([pl.scan_parquet(p) for p in canonical_batch_paths], how="vertical").sink_parquet(
        canonical_path, compression="zstd"
    )
    canonical_row_count = pl.scan_parquet(canonical_path).select(pl.len()).collect().item()
    _log_stage("canonical_telemetry_combined", t0, row_count=canonical_row_count)

    t0 = time.perf_counter()
    maturity_cfg = config["feature_maturity"]
    drive_metadata = build_drive_metadata(
        pl.scan_parquet(drive_day_path), min_history_days=maturity_cfg["min_history_days"]
    )
    drive_metadata.write_parquet(metadata_path, compression="zstd")
    _log_stage("drive_metadata_built", t0, row_count=drive_metadata.height)

    t0 = time.perf_counter()
    quality_reports = run_all_checks(pl.scan_parquet(drive_day_path))
    quality_report_path.write_text(json.dumps(quality_reports, indent=2, default=str))
    _log_stage("quality_checks_run", t0, check_count=len(quality_reports))

    result_path.write_text(
        json.dumps(
            {
                "drive_day_row_count": drive_day_row_count,
                "canonical_row_count": canonical_row_count,
                "drive_metadata_row_count": drive_metadata.height,
            }
        )
    )


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage - a new
    process, and therefore a new address space, regardless of what the
    calling process's allocator has retained. See the module docstring."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "batch", "finalize"])
    parser.add_argument("--bronze-root", type=Path)
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--n-batches", type=int)
    parser.add_argument("--drive-day-out-path", type=Path)
    parser.add_argument("--canonical-out-path", type=Path)
    parser.add_argument("--drive-day-batch-paths", type=Path, nargs="*")
    parser.add_argument("--canonical-batch-paths", type=Path, nargs="*")
    parser.add_argument("--drive-day-path", type=Path)
    parser.add_argument("--canonical-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--quality-report-path", type=Path)
    parser.add_argument("--result-path", type=Path)
    args = parser.parse_args()

    if args.stage == "prepare":
        _stage_prepare(args.bronze_root, args.tmp_dir)
        return
    if args.stage == "batch":
        _stage_batch(
            args.bronze_root,
            args.batch_index,
            args.n_batches,
            args.drive_day_out_path,
            args.canonical_out_path,
        )
        return
    if args.stage == "finalize":
        _stage_finalize(
            args.drive_day_batch_paths,
            args.canonical_batch_paths,
            args.drive_day_path,
            args.canonical_path,
            args.metadata_path,
            args.quality_report_path,
            args.result_path,
        )
        return

    # No --stage: the top-level orchestrator. It never itself scans Bronze
    # or holds a large Polars frame - only the subprocesses it spawns do -
    # so its own address space stays flat no matter how many batches there
    # are.
    configure_logging()
    config = _load_configs()
    silver_dir = Path(config["silver_dir"])
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"

    bronze_root = Path(config["bronze_dir"])
    drive_day_path = silver_dir / "drive_day" / "part.parquet"
    canonical_path = silver_dir / "canonical_telemetry" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    quality_report_path = audit_dir / "silver_quality_report.json"

    tmp_dir = Path(tempfile.mkdtemp(prefix="build_silver_"))
    try:
        _run_stage(
            "--stage", "prepare", "--bronze-root", str(bronze_root), "--tmp-dir", str(tmp_dir)
        )
        manifest = json.loads((tmp_dir / PREPARE_MANIFEST_NAME).read_text())
        n_batches = manifest["n_batches"]

        drive_day_batch_paths = []
        canonical_batch_paths = []
        for batch_index in range(n_batches):
            drive_day_batch_path = tmp_dir / f"drive_day_batch_{batch_index}.parquet"
            canonical_batch_path = tmp_dir / f"canonical_batch_{batch_index}.parquet"
            _run_stage(
                "--stage",
                "batch",
                "--bronze-root",
                str(bronze_root),
                "--batch-index",
                str(batch_index),
                "--n-batches",
                str(n_batches),
                "--drive-day-out-path",
                str(drive_day_batch_path),
                "--canonical-out-path",
                str(canonical_batch_path),
            )
            drive_day_batch_paths.append(drive_day_batch_path)
            canonical_batch_paths.append(canonical_batch_path)
            logger.info(
                "build_silver_batch_done", batch_index=batch_index, batch_count=n_batches
            )

        result_path = tmp_dir / FINALIZE_RESULT_NAME
        _run_stage(
            "--stage",
            "finalize",
            "--drive-day-batch-paths",
            *[str(p) for p in drive_day_batch_paths],
            "--canonical-batch-paths",
            *[str(p) for p in canonical_batch_paths],
            "--drive-day-path",
            str(drive_day_path),
            "--canonical-path",
            str(canonical_path),
            "--metadata-path",
            str(metadata_path),
            "--quality-report-path",
            str(quality_report_path),
            "--result-path",
            str(result_path),
        )
        result = json.loads(result_path.read_text())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    quality_reports = json.loads(quality_report_path.read_text())
    failed = [r for r in quality_reports if not r["passed"]]
    logger.info(
        "drive_day_written",
        row_count=result["drive_day_row_count"],
        path=str(drive_day_path.parent),
    )
    logger.info(
        "canonical_telemetry_written",
        row_count=result["canonical_row_count"],
        path=str(canonical_path.parent),
    )
    logger.info(
        "drive_metadata_written",
        row_count=result["drive_metadata_row_count"],
        path=str(metadata_path.parent),
    )
    if failed:
        logger.warning("data_quality_checks_failed", failed_count=len(failed), failed=failed)
    else:
        logger.info("data_quality_checks_passed")


if __name__ == "__main__":
    main()
