"""Entry point for `make build-features`.

Pivots the silver canonical telemetry into wide form and computes every gold
feature family: rolling aggregates, deltas/slopes/acceleration, event/spike
counts, feature confidence, and lifecycle context. Writes the gold feature
table and a versioned feature registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from src.features.confidence import add_feature_confidence
from src.features.cross_vendor import add_attribute_ratios, add_model_family_zscores
from src.features.derivatives import add_acceleration, add_deltas
from src.features.events import add_positive_day_counts, add_spike_counts, add_zero_to_nonzero_flags
from src.features.lifecycle import add_lifecycle_features
from src.features.pivot import pivot_badness_wide
from src.features.registry import build_registry
from src.features.windows import add_rolling_aggregates
from src.logging_config import configure_logging, get_logger

DATA_CONFIG_PATH = Path("configs/data.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def build_gold_features(canonical_long: pl.DataFrame, features_config: dict) -> pl.DataFrame:
    wide = pivot_badness_wide(canonical_long)

    available_attributes = [
        a for a in features_config["priority_smart_attributes"] if a in wide.columns
    ]
    windows_days = tuple(features_config["windows_days"])
    spike_thresholds = {
        a: t for a, t in features_config["spike_thresholds"].items() if a in available_attributes
    }

    gold = add_rolling_aggregates(wide, available_attributes, windows_days=windows_days)
    gold = add_deltas(gold, available_attributes, windows_days=windows_days)
    gold = add_acceleration(
        gold,
        available_attributes,
        short_window_days=windows_days[0],
        long_window_days=windows_days[-1],
    )
    gold = add_positive_day_counts(gold, available_attributes, windows_days=windows_days)
    if spike_thresholds:
        gold = add_spike_counts(gold, spike_thresholds, windows_days=windows_days)
    gold = add_zero_to_nonzero_flags(gold, available_attributes)
    gold = add_lifecycle_features(gold)
    gold = add_attribute_ratios(gold)
    gold = add_model_family_zscores(gold, available_attributes, windows_days=windows_days)
    gold = add_feature_confidence(
        gold,
        available_attributes,
        recency_tau_hours=features_config["feature_confidence"]["recency_tau_hours"],
    )

    return gold, available_attributes, windows_days, spike_thresholds


def main() -> None:
    configure_logging()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    canonical_path = silver_dir / "canonical_telemetry" / "part.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(f"{canonical_path} not found; run `make build-silver` first.")

    canonical_long = pl.read_parquet(canonical_path)
    gold, attributes, windows_days, spike_thresholds = build_gold_features(
        canonical_long, features_config
    )

    gold_dir = Path(data_config["gold_dir"]) / "features"
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "part.parquet"
    gold.write_parquet(out_path, compression="zstd")

    registry = build_registry(attributes, windows_days, spike_thresholds)
    registry_dir = Path(data_config["audit_dir"]) / "feature_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"v{features_config['version']}.json"
    registry_path.write_text(
        json.dumps([e.model_dump() for e in registry], indent=2, default=str)
    )

    logger.info(
        "gold_features_written",
        row_count=gold.height,
        column_count=len(gold.columns),
        path=str(out_path),
    )
    logger.info(
        "feature_registry_written", entry_count=len(registry), path=str(registry_path)
    )


if __name__ == "__main__":
    main()
