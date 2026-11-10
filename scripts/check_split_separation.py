"""Check that train, validation and test stay separate under `configs/model.yaml`'s `splits`.

    uv run python scripts/check_split_separation.py                  # primary horizon
    uv run python scripts/check_split_separation.py --horizon-days 60

Runs the config checks, then, if the label table exists, reads the rows each role would use and
checks their date ranges (src/labels/split_checks.py; docs/dataset_strategy.md section 9.5).
Exits 1 when `splits.strict_separation` is on and a check fails. With it off (the drive-holdout
design, which shares dates between train and validation by construction) the violations are
printed for information and the exit code stays 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

from src.labels.split_checks import (
    check_label_table_separation,
    check_split_config,
    strict_separation_enabled,
)

MODEL_CONFIG_PATH = Path("configs/model.yaml")
DATA_CONFIG_PATH = Path("configs/data.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--horizon-days", type=int, default=None, help="default: primary_horizon_days"
    )
    args = parser.parse_args(argv)

    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    horizon = args.horizon_days or model_config["primary_horizon_days"]
    strict = strict_separation_enabled(model_config)

    violations = check_split_config(model_config, horizon)
    labels_path = Path(data_config["gold_dir"]) / "labels" / "part.parquet"
    if labels_path.exists():
        test_split = str(model_config.get("splits", {}).get("test_split") or "test")
        violations += check_label_table_separation(
            pl.scan_parquet(labels_path), model_config, horizon, test_split=test_split
        )
    else:
        print(f"label table not found at {labels_path}: config checks only")

    print(f"horizon {horizon} days, strict_separation {'on' if strict else 'off'}")
    if not violations:
        print("OK: train, validation and test are separate")
        return 0
    for violation in violations:
        print(f"  - {violation}")
    if strict:
        return 1
    print("strict_separation is off, so these are informational")
    return 0


if __name__ == "__main__":
    sys.exit(main())
