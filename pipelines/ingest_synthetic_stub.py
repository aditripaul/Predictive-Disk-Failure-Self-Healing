"""Entry point for `make ingest-synthetic-stub`.

Lands the trivial synthetic placeholder dataset into Bronze so downstream
code can be exercised against a `source_dataset=synthetic` partition before
the full chaos generator (Phase 8) exists.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.ingest.synthetic_stub import ingest_synthetic_stub

CONFIG_PATH = Path("configs/data.yaml")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    bronze_root = Path(config["sources"]["synthetic"]["bronze_path"])
    written = ingest_synthetic_stub(bronze_root)
    print(f"Wrote synthetic stub -> {len(written)} partition(s)")


if __name__ == "__main__":
    main()
