"""Entry point for `make build-silver`.

Reads every Bronze partition, normalizes identifiers, harmonizes SMART
attributes into the canonical long schema, computes telemetry-gap/staleness
features and drive-level feature maturity, runs data-quality checks, and
writes the Silver layer plus a quality report to data/audit/.
"""

from __future__ import annotations

import json
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

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_silver_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def build_silver(
    bronze_root: Path, config: dict, *, drive_day_path: Path, canonical_path: Path
) -> tuple[int, int, pl.DataFrame, list[dict]]:
    """Builds the Silver layer and sinks both the per-drive-day table
    (`drive_day_path`) and the canonical long telemetry table
    (`canonical_path`) straight to Parquet rather than returning either in
    memory.

    `melt_smart_attributes` is only ever given `id_columns=["drive_id",
    "date"]` here - carrying every other drive-day column (drive_model,
    capacity_gb, telemetry-gap flags, ...) through the melt as well would
    duplicate each one once per SMART attribute (~10x at fleet scale) for
    no reason, since none of them vary by attribute. Those columns are
    instead persisted once, at their natural drive-day grain, in
    `drive_day_path`; `src/features/pivot.py::pivot_badness_wide` joins
    them back in later, after pivoting canonical_long back down to that
    same grain, where the join is cheap.

    Neither `drive_day` nor `canonical_long` is ever a materialized
    `DataFrame` here - both are written via `LazyFrame.sink_parquet`,
    which streams the computation AND the output in row-group-sized
    batches, so neither one's size is bounded by available RAM. Row
    counts and quality checks likewise run as narrow lazy aggregate
    queries against the Parquet files just written, never against an
    in-memory copy. Only `drive_metadata` (one row per drive) is ever a
    real in-memory `DataFrame`, since it's inherently tiny regardless of
    fleet size.
    """
    bronze_files = sorted(bronze_root.glob("**/*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze Parquet files found under {bronze_root}")
    logger.info("build_silver_bronze_files_found", file_count=len(bronze_files))

    # Scan (not read) every file, and fold normalization, failure-date
    # derivation, and telemetry-gap computation into the SAME lazy plan as
    # the concat, sinking once at the end. Collecting an intermediate
    # "wide" frame eagerly and then feeding it into the next eager step
    # would hold both the input and that step's output fully in memory at
    # once; one streaming sink over the whole chain avoids ever
    # materializing drive_day in memory at all.
    drive_day_lazy = pl.concat(
        [pl.scan_parquet(p) for p in bronze_files], how="diagonal_relaxed"
    )
    drive_day_lazy = normalize_identifiers(drive_day_lazy)
    drive_day_lazy = derive_failure_date(drive_day_lazy)
    gap_cfg = config["telemetry_gap"]
    drive_day_lazy = compute_telemetry_gaps(
        drive_day_lazy,
        short_gap_days=gap_cfg["short_gap_days"],
        stale_gap_days=gap_cfg["stale_gap_days"],
    )

    drive_day_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    drive_day_lazy.sink_parquet(drive_day_path, compression="zstd")
    drive_day_row_count = pl.scan_parquet(drive_day_path).select(pl.len()).collect().item()
    _log_stage("bronze_scanned_normalized_and_gaps_computed", t0, row_count=drive_day_row_count)

    # drive_metadata: it's one row per drive, tiny next to the canonical
    # long table below.
    t0 = time.perf_counter()
    maturity_cfg = config["feature_maturity"]
    drive_metadata = build_drive_metadata(
        pl.scan_parquet(drive_day_path), min_history_days=maturity_cfg["min_history_days"]
    )
    _log_stage("drive_metadata_built", t0, row_count=drive_metadata.height)

    t0 = time.perf_counter()
    melt_smart_attributes(
        pl.scan_parquet(drive_day_path), id_columns=["drive_id", "date"]
    ).sink_parquet(canonical_path, compression="zstd")
    canonical_row_count = pl.scan_parquet(canonical_path).select(pl.len()).collect().item()
    _log_stage("smart_attributes_melted_and_written", t0, row_count=canonical_row_count)

    # Quality checks run against drive_day, not canonical_long: capacity_gb
    # and failure_date only exist at the drive-day grain now, and checking
    # for duplicate (drive_id, date) rows there is equivalent to (and
    # cheaper than) checking canonical_long for duplicate
    # (drive_id, date, smart_attribute_name) rows, since a duplicate
    # drive-day row would produce duplicates for every melted attribute.
    t0 = time.perf_counter()
    quality_reports = run_all_checks(pl.scan_parquet(drive_day_path))
    _log_stage("quality_checks_run", t0, check_count=len(quality_reports))

    return drive_day_row_count, canonical_row_count, drive_metadata, quality_reports


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    silver_dir = Path(config["silver_dir"])
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"

    all_bronze_root = Path(config["bronze_dir"])
    drive_day_dir = silver_dir / "drive_day"
    drive_day_path = drive_day_dir / "part.parquet"
    canonical_dir = silver_dir / "canonical_telemetry"
    canonical_path = canonical_dir / "part.parquet"
    drive_day_row_count, canonical_row_count, drive_metadata, quality_reports = build_silver(
        all_bronze_root, config, drive_day_path=drive_day_path, canonical_path=canonical_path
    )

    metadata_dir = silver_dir / "drive_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    drive_metadata.write_parquet(metadata_dir / "part.parquet", compression="zstd")

    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "silver_quality_report.json").write_text(
        json.dumps(quality_reports, indent=2, default=str)
    )

    failed = [r for r in quality_reports if not r["passed"]]
    logger.info(
        "drive_day_written", row_count=drive_day_row_count, path=str(drive_day_dir)
    )
    logger.info(
        "canonical_telemetry_written", row_count=canonical_row_count, path=str(canonical_dir)
    )
    logger.info(
        "drive_metadata_written", row_count=drive_metadata.height, path=str(metadata_dir)
    )
    if failed:
        logger.warning("data_quality_checks_failed", failed_count=len(failed), failed=failed)
    else:
        logger.info("data_quality_checks_passed")


if __name__ == "__main__":
    main()
