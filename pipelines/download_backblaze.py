"""Entry point for `make download-backblaze`.

Downloads and extracts the configured Backblaze quarterly archives
(docs/dataset_strategy.md section 3.1) into `data/raw/backblaze/`, ready
for `make ingest-backblaze`. The time period is fully configurable - there
is no hardcoded default quarter - via either:

    configs/data.yaml -> download.backblaze.quarters: ["Q1_2025", "Q2_2025"]
    configs/data.yaml -> download.backblaze.start_quarter / end_quarter

or overridden per-run without touching the config file:

    uv run python pipelines/download_backblaze.py --quarters Q1_2025 Q2_2025
    uv run python pipelines/download_backblaze.py --start-quarter Q1_2025 --end-quarter Q4_2025

`--quarters` (or the config's `quarters` list) takes priority if given;
otherwise the start/end quarter range is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.ingest.download import DEFAULT_BACKBLAZE_BASE_URL, download_backblaze, resolve_quarters
from src.logging_config import configure_logging, get_logger

DATA_CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quarters",
        nargs="+",
        default=None,
        metavar="Qn_YYYY",
        help="Quarters to download, e.g. --quarters Q1_2025 Q2_2025 "
        "(default: configs/data.yaml's download.backblaze.quarters)",
    )
    parser.add_argument(
        "--start-quarter",
        default=None,
        metavar="Qn_YYYY",
        help="Start of an inclusive quarter range (used only if --quarters/"
        "download.backblaze.quarters is empty)",
    )
    parser.add_argument(
        "--end-quarter",
        default=None,
        metavar="Qn_YYYY",
        help="End of an inclusive quarter range (used only if --quarters/"
        "download.backblaze.quarters is empty)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already present"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    download_cfg = data_config.get("download", {}).get("backblaze", {})

    quarters = resolve_quarters(
        cli_quarters=args.quarters,
        cli_start_quarter=args.start_quarter,
        cli_end_quarter=args.end_quarter,
        config_quarters=download_cfg.get("quarters"),
        config_start_quarter=download_cfg.get("start_quarter"),
        config_end_quarter=download_cfg.get("end_quarter"),
    )
    base_url = download_cfg.get("base_url", DEFAULT_BACKBLAZE_BASE_URL)
    raw_dir = Path(data_config["sources"]["backblaze"]["raw_glob"]).parent

    results = download_backblaze(quarters, base_url=base_url, raw_dir=raw_dir, force=args.force)

    for quarter, extracted_files in results.items():
        if extracted_files:
            logger.info(
                "backblaze_quarter_downloaded",
                quarter=quarter,
                file_count=len(extracted_files),
                raw_dir=str(raw_dir),
            )
        else:
            logger.info("backblaze_quarter_already_downloaded", quarter=quarter)


if __name__ == "__main__":
    main()
