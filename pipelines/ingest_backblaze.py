"""Entry point for `make ingest-backblaze`.

Processes each raw Backblaze CSV one file at a time (never loading the full
dataset into memory), lands it in the Bronze layer, and writes a profiling
report to data/audit/data_quality_reports/.

Which raw files get ingested (and therefore how much data `make
build-silver` has to process) is restricted the same way as downloads -
configurably, with no hardcoded default:

    configs/data.yaml -> sources.backblaze.start_date / end_date

or overridden per-run without touching the config file:

    uv run python pipelines/ingest_backblaze.py --start-date 2026-01-01 --end-date 2026-01-31

Leaving both unset (the default) ingests every CSV matching `raw_glob`,
same as before this option existed.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

from src.ingest.backblaze import filter_files_by_date_range, ingest_backblaze_file
from src.ingest.profiling import profile_bronze_dataset, write_profiling_report
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only ingest raw CSVs dated on/after this day (default: "
        "configs/data.yaml's sources.backblaze.start_date, or unrestricted)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only ingest raw CSVs dated on/before this day (default: "
        "configs/data.yaml's sources.backblaze.end_date, or unrestricted)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    apply_memory_limit_from_config()
    args = parse_args(argv)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    backblaze_cfg = config["sources"]["backblaze"]
    raw_glob = backblaze_cfg["raw_glob"]
    bronze_root = Path(backblaze_cfg["bronze_path"])

    start_date_str = args.start_date or backblaze_cfg.get("start_date")
    end_date_str = args.end_date or backblaze_cfg.get("end_date")
    start_date = dt.date.fromisoformat(start_date_str) if start_date_str else None
    end_date = dt.date.fromisoformat(end_date_str) if end_date_str else None

    raw_files = sorted(Path().glob(raw_glob))
    raw_files = filter_files_by_date_range(raw_files, start_date=start_date, end_date=end_date)
    if not raw_files:
        logger.warning(
            "no_backblaze_raw_files_found",
            raw_glob=raw_glob,
            start_date=start_date_str,
            end_date=end_date_str,
            hint="Download quarterly archives into data/raw/backblaze/ first, "
            "or widen start_date/end_date if they're excluding everything.",
        )
        return

    for csv_path in raw_files:
        written = ingest_backblaze_file(csv_path, bronze_root)
        logger.info("backblaze_file_ingested", source=str(csv_path), partition_count=len(written))

    report = profile_bronze_dataset(bronze_root)
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"
    out_path = write_profiling_report(report, audit_dir, "backblaze")
    logger.info("profiling_report_written", path=str(out_path))


if __name__ == "__main__":
    main()
