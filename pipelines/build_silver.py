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

from src.preprocess.feature_maturity import build_drive_metadata
from src.preprocess.identifiers import normalize_identifiers
from src.preprocess.quality_checks import run_all_checks
from src.preprocess.smart_mapping import melt_smart_attributes
from src.preprocess.telemetry_gaps import compute_telemetry_gaps

CONFIG_PATH = Path("configs/data.yaml")


def build_silver(bronze_root: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    bronze_files = sorted(bronze_root.glob("**/*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze Parquet files found under {bronze_root}")

    wide = pl.concat([pl.read_parquet(p) for p in bronze_files], how="diagonal_relaxed")
    wide = normalize_identifiers(wide.lazy()).collect()

    gap_cfg = config["telemetry_gap"]
    drive_day = compute_telemetry_gaps(
        wide,
        short_gap_days=gap_cfg["short_gap_days"],
        stale_gap_days=gap_cfg["stale_gap_days"],
    )

    canonical_long = melt_smart_attributes(drive_day)

    quality_reports = run_all_checks(canonical_long)

    maturity_cfg = config["feature_maturity"]
    drive_metadata = build_drive_metadata(
        drive_day, min_history_days=maturity_cfg["min_history_days"]
    )

    return canonical_long, drive_metadata, quality_reports


def main() -> None:
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
    print(f"Wrote {canonical_long.height} canonical telemetry rows to {canonical_dir}")
    print(f"Wrote {drive_metadata.height} drive metadata rows to {metadata_dir}")
    if failed:
        print(f"WARNING: {len(failed)} data quality check(s) failed: {failed}")
    else:
        print("All data quality checks passed.")


if __name__ == "__main__":
    main()
