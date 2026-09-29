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


def build_silver(bronze_root: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    bronze_files = sorted(bronze_root.glob("**/*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze Parquet files found under {bronze_root}")
    logger.info("build_silver_bronze_files_found", file_count=len(bronze_files))

    # Scan (not read) every file, and fold normalization + failure-date
    # derivation into the SAME lazy plan as the concat, collecting once at
    # the end - collecting the raw concat eagerly and then re-lazying it
    # for a second, non-streaming collect() (the previous shape here) holds
    # the pre- and post-normalization frames fully in memory at the same
    # time, on top of not streaming the second pass at all. One streaming
    # collect over the whole chain avoids that second full-size duplicate.
    t0 = time.perf_counter()
    wide_lazy = pl.concat(
        [pl.scan_parquet(p) for p in bronze_files], how="diagonal_relaxed"
    )
    wide_lazy = normalize_identifiers(wide_lazy)
    wide_lazy = derive_failure_date(wide_lazy)
    wide = wide_lazy.collect(engine="streaming")
    _log_stage("bronze_scanned_normalized_and_concatenated", t0, row_count=wide.height)

    t0 = time.perf_counter()
    gap_cfg = config["telemetry_gap"]
    drive_day = compute_telemetry_gaps(
        wide,
        short_gap_days=gap_cfg["short_gap_days"],
        stale_gap_days=gap_cfg["stale_gap_days"],
    )
    del wide  # superseded by drive_day; drop it before allocating anything else
    _log_stage("telemetry_gaps_computed", t0, row_count=drive_day.height)

    # drive_metadata first: it's one row per drive, tiny next to
    # canonical_long (melt_smart_attributes unpivots to one row per
    # drive-day-*attribute*, ~10x drive_day's row count for the 10
    # priority SMART attributes) - freeing drive_day before that unpivot
    # keeps peak memory to roughly one big frame at a time, not three.
    t0 = time.perf_counter()
    maturity_cfg = config["feature_maturity"]
    drive_metadata = build_drive_metadata(
        drive_day, min_history_days=maturity_cfg["min_history_days"]
    )
    _log_stage("drive_metadata_built", t0, row_count=drive_metadata.height)

    t0 = time.perf_counter()
    canonical_long = melt_smart_attributes(drive_day)
    del drive_day
    _log_stage("smart_attributes_melted", t0, row_count=canonical_long.height)

    t0 = time.perf_counter()
    quality_reports = run_all_checks(canonical_long)
    _log_stage("quality_checks_run", t0, check_count=len(quality_reports))

    return canonical_long, drive_metadata, quality_reports


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    silver_dir = Path(config["silver_dir"])
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"

    all_bronze_root = Path(config["bronze_dir"])
    canonical_long, drive_metadata, quality_reports = build_silver(all_bronze_root, config)

    canonical_dir = silver_dir / "canonical_telemetry"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_long.write_parquet(canonical_dir / "part.parquet", compression="zstd")

    metadata_dir = silver_dir / "drive_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    drive_metadata.write_parquet(metadata_dir / "part.parquet", compression="zstd")

    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "silver_quality_report.json").write_text(
        json.dumps(quality_reports, indent=2, default=str)
    )

    failed = [r for r in quality_reports if not r["passed"]]
    logger.info(
        "canonical_telemetry_written", row_count=canonical_long.height, path=str(canonical_dir)
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
