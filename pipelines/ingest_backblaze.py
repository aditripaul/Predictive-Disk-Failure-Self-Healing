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

CONFIG_PATH = Path("configs/data.yaml")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    raw_glob = config["sources"]["backblaze"]["raw_glob"]
    bronze_root = Path(config["sources"]["backblaze"]["bronze_path"])

    raw_files = sorted(Path().glob(raw_glob))
    if not raw_files:
        print(f"No Backblaze raw files found matching '{raw_glob}'.")
        print("Download quarterly archives into data/raw/backblaze/ first.")
        return

    for csv_path in raw_files:
        written = ingest_backblaze_file(csv_path, bronze_root)
        print(f"Ingested {csv_path} -> {len(written)} partition(s)")

    report = profile_bronze_dataset(bronze_root)
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"
    out_path = write_profiling_report(report, audit_dir, "backblaze")
    print(f"Wrote profiling report to {out_path}")


if __name__ == "__main__":
    main()
