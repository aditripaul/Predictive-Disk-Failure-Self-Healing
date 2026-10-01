"""Entry point for `make build-sequences` (optional LSTM branch,
docs/dataset_strategy.md section 10.7 Feature Family G).

Reads gold features, builds `[samples, time_steps, features]` sequence
tensors for the configured attributes, and writes them as a memory-mapped
`.npy` file plus metadata under `data/gold/sequences/`. This is entirely
optional infrastructure for the LSTM comparison branch
(`pipelines/train_lstm.py`) - the primary training pipeline
(`pipelines/train_model.py`) never reads this output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from src.features.sequences import (
    DEFAULT_SEQUENCE_ATTRIBUTES,
    build_sequence_tensors,
    write_sequence_tensors,
)
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())
    sequence_cfg = features_config.get("sequences", {})

    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found; run `make build-features` first.")

    attributes = tuple(sequence_cfg.get("attributes", DEFAULT_SEQUENCE_ATTRIBUTES))
    time_steps = sequence_cfg.get("time_steps", 30)

    # Read only the columns the sequence builder actually touches
    # (`drive_id`, `date`, and the configured attributes - see
    # src/features/sequences.py::build_sequence_tensors), not the whole
    # gold feature table. It is ~196 columns and ~10GB for one month of
    # real data, of which this needs about 11; unlike the join in
    # src/models/features.py, the columns here are known up front, so
    # projection pushdown genuinely applies. Attributes absent from the
    # table are dropped here rather than passed to a scan that would
    # raise on them - `build_sequence_tensors` already treats a missing
    # attribute as "not available" and reports `attributes_used`.
    available = pl.scan_parquet(features_path).collect_schema().names()
    present_attributes = tuple(a for a in attributes if a in available)
    wide_features = (
        pl.scan_parquet(features_path)
        .select(["drive_id", "date", *present_attributes])
        .collect()
    )

    tensor, attributes_used, drive_ids, as_of_dates = build_sequence_tensors(
        wide_features, attributes=attributes, time_steps=time_steps
    )

    out_dir = gold_dir / "sequences"
    tensor_path = write_sequence_tensors(
        tensor,
        attributes_used=attributes_used,
        drive_ids=drive_ids,
        as_of_dates=as_of_dates,
        out_dir=out_dir,
    )

    logger.info(
        "sequence_tensors_written",
        shape=list(tensor.shape),
        attributes_used=attributes_used,
        path=str(tensor_path),
    )


if __name__ == "__main__":
    main()
