"""Entry point for `make build-features`.

Pivots the silver canonical telemetry into wide form and computes every gold
feature family: rolling aggregates, deltas/slopes/acceleration, event/spike
counts, feature confidence, and lifecycle context. Writes the gold feature
table and a versioned feature registry.
"""

from __future__ import annotations

import json
import time
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
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_gold_features_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def build_gold_features(
    canonical_long: pl.DataFrame, features_config: dict, *, drive_day: pl.DataFrame | None = None
) -> pl.DataFrame:
    t0 = time.perf_counter()
    wide = pivot_badness_wide(canonical_long, drive_day=drive_day)
    _log_stage("pivoted_to_wide", t0, row_count=wide.height, column_count=len(wide.columns))

    available_attributes = [
        a for a in features_config["priority_smart_attributes"] if a in wide.columns
    ]
    windows_days = tuple(features_config["windows_days"])
    spike_thresholds = {
        a: t for a, t in features_config["spike_thresholds"].items() if a in available_attributes
    }

    t0 = time.perf_counter()
    gold = add_rolling_aggregates(wide, available_attributes, windows_days=windows_days)
    _log_stage("rolling_aggregates_added", t0)

    t0 = time.perf_counter()
    gold = add_deltas(gold, available_attributes, windows_days=windows_days)
    _log_stage("deltas_added", t0)

    t0 = time.perf_counter()
    gold = add_acceleration(
        gold,
        available_attributes,
        short_window_days=windows_days[0],
        long_window_days=windows_days[-1],
    )
    _log_stage("acceleration_added", t0)

    t0 = time.perf_counter()
    gold = add_positive_day_counts(gold, available_attributes, windows_days=windows_days)
    _log_stage("positive_day_counts_added", t0)

    t0 = time.perf_counter()
    if spike_thresholds:
        gold = add_spike_counts(gold, spike_thresholds, windows_days=windows_days)
    _log_stage("spike_counts_added", t0)

    t0 = time.perf_counter()
    gold = add_zero_to_nonzero_flags(gold, available_attributes)
    _log_stage("zero_to_nonzero_flags_added", t0)

    t0 = time.perf_counter()
    gold = add_lifecycle_features(gold)
    _log_stage("lifecycle_features_added", t0)

    t0 = time.perf_counter()
    gold = add_attribute_ratios(gold)
    _log_stage("attribute_ratios_added", t0)

    t0 = time.perf_counter()
    gold = add_model_family_zscores(gold, available_attributes, windows_days=windows_days)
    _log_stage("model_family_zscores_added", t0)

    t0 = time.perf_counter()
    gold = add_feature_confidence(
        gold,
        available_attributes,
        recency_tau_hours=features_config["feature_confidence"]["recency_tau_hours"],
    )
    _log_stage(
        "feature_confidence_added", t0, row_count=gold.height, column_count=len(gold.columns)
    )

    return gold, available_attributes, windows_days, spike_thresholds


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    canonical_path = silver_dir / "canonical_telemetry" / "part.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(f"{canonical_path} not found; run `make build-silver` first.")
    drive_day_path = silver_dir / "drive_day" / "part.parquet"
    if not drive_day_path.exists():
        raise FileNotFoundError(f"{drive_day_path} not found; run `make build-silver` first.")

    t0 = time.perf_counter()
    canonical_long = pl.read_parquet(canonical_path)
    _log_stage("canonical_telemetry_read", t0, row_count=canonical_long.height)

    t0 = time.perf_counter()
    drive_day = pl.read_parquet(drive_day_path)
    _log_stage("drive_day_read", t0, row_count=drive_day.height)

    gold, attributes, windows_days, spike_thresholds = build_gold_features(
        canonical_long, features_config, drive_day=drive_day
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
