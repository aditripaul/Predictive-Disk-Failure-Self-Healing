"""Entry point for `make download-smartz`.

SMART-Z has no public bulk-download API (docs/dataset_strategy.md section
3.2): request access first, then either set `download.smartz.url` in
configs/data.yaml to the archive URL you were given, or place its CSV
files directly into `data/raw/smartz/` yourself and skip this script
entirely. `--url` overrides the config file for a one-off run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.ingest.download import download_smartz
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=None,
        help="Archive URL to download (default: configs/data.yaml's download.smartz.url)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already present"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    apply_memory_limit_from_config()
    args = parse_args(argv)
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    download_cfg = data_config.get("download", {}).get("smartz", {})

    url = args.url or download_cfg.get("url")
    raw_dir = Path(data_config["sources"]["smartz"]["raw_glob"]).parent

    extracted_files = download_smartz(raw_dir, url=url, force=args.force)

    if extracted_files:
        logger.info(
            "smartz_downloaded", file_count=len(extracted_files), raw_dir=str(raw_dir)
        )
    else:
        logger.info("smartz_already_downloaded")


if __name__ == "__main__":
    main()
