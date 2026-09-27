"""Entry point for `make ingest-synthetic-stub`.

Lands the trivial synthetic placeholder dataset into Bronze so downstream
code can be exercised against a `source_dataset=synthetic` partition before
the full chaos generator (Phase 8) exists.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingest.synthetic_stub import ingest_synthetic_stub
from src.logging_config import configure_logging, get_logger

CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    bronze_root = Path(config["sources"]["synthetic"]["bronze_path"])
    written = ingest_synthetic_stub(bronze_root)
    logger.info("synthetic_stub_written", partition_count=len(written))


if __name__ == "__main__":
    main()
