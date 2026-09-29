"""Entry point for `make ingest-backblaze`.

Processes each raw Backblaze CSV one file at a time (never loading the full
dataset into memory), lands it in the Bronze layer, and writes a profiling
report to data/audit/data_quality_reports/.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingest.backblaze import ingest_backblaze_file
from src.ingest.profiling import profile_bronze_dataset, write_profiling_report
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    raw_glob = config["sources"]["backblaze"]["raw_glob"]
    bronze_root = Path(config["sources"]["backblaze"]["bronze_path"])

    raw_files = sorted(Path().glob(raw_glob))
    if not raw_files:
        logger.warning(
            "no_backblaze_raw_files_found",
            raw_glob=raw_glob,
            hint="Download quarterly archives into data/raw/backblaze/ first.",
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
