"""Entry point for `make ingest-smartz`.

Processes each raw SMART-Z CSV one file at a time, lands it in the Bronze
layer, and writes a profiling report to data/audit/data_quality_reports/.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingest.profiling import profile_bronze_dataset, write_profiling_report
from src.ingest.smartz import ingest_smartz_file

CONFIG_PATH = Path("configs/data.yaml")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    raw_glob = config["sources"]["smartz"]["raw_glob"]
    bronze_root = Path(config["sources"]["smartz"]["bronze_path"])

    raw_files = sorted(Path().glob(raw_glob))
    if not raw_files:
        print(f"No SMART-Z raw files found matching '{raw_glob}'.")
        print("Request/download the SMART-Z dataset into data/raw/smartz/ first.")
        return

    for csv_path in raw_files:
        written = ingest_smartz_file(csv_path, bronze_root)
        print(f"Ingested {csv_path} -> {len(written)} partition(s)")

    report = profile_bronze_dataset(bronze_root)
    audit_dir = Path(config["audit_dir"]) / "data_quality_reports"
    out_path = write_profiling_report(report, audit_dir, "smartz")
    print(f"Wrote profiling report to {out_path}")


if __name__ == "__main__":
    main()
