"""Entry point for `make build-silver`.

Reads every Bronze partition, normalizes identifiers, harmonizes SMART
attributes into the canonical long schema, computes telemetry-gap/staleness
features and drive-level feature maturity, runs data-quality checks, and
writes the Silver layer plus a quality report to data/audit/.
"""

from __future__ import annotations

import json
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

CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def build_silver(bronze_root: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    bronze_files = sorted(bronze_root.glob("**/*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze Parquet files found under {bronze_root}")

    # Scan (not read) every file so none of them is ever fully materialized
    # on its own before the concat - `pl.read_parquet` per file would hold
    # every file's full contents in memory simultaneously, on top of the
    # concatenated result, roughly doubling peak RAM for no reason.
    wide = pl.concat(
        [pl.scan_parquet(p) for p in bronze_files], how="diagonal_relaxed"
    ).collect(engine="streaming")
    wide = normalize_identifiers(wide.lazy()).collect()
    wide = derive_failure_date(wide)

    gap_cfg = config["telemetry_gap"]
    drive_day = compute_telemetry_gaps(
        wide,
        short_gap_days=gap_cfg["short_gap_days"],
        stale_gap_days=gap_cfg["stale_gap_days"],
    )
    del wide  # superseded by drive_day; drop it before allocating anything else

    # drive_metadata first: it's one row per drive, tiny next to
    # canonical_long (melt_smart_attributes unpivots to one row per
    # drive-day-*attribute*, ~10x drive_day's row count for the 10
    # priority SMART attributes) - freeing drive_day before that unpivot
    # keeps peak memory to roughly one big frame at a time, not three.
    maturity_cfg = config["feature_maturity"]
    drive_metadata = build_drive_metadata(
        drive_day, min_history_days=maturity_cfg["min_history_days"]
    )

    canonical_long = melt_smart_attributes(drive_day)
    del drive_day

    quality_reports = run_all_checks(canonical_long)

    return canonical_long, drive_metadata, quality_reports


def main() -> None:
    configure_logging()
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
