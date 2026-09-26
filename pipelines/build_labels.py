"""Entry point for `make build-labels`.

Reads gold features and silver drive metadata, classifies removal event
types, computes leakage-free failure labels for every configured horizon,
applies the chronological (+ optional drive-holdout) split, and writes the
label table, split assignment, and imbalance report.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from src.labels.event_types import classify_event_types
from src.labels.imbalance import class_distribution_report, compute_scale_pos_weight
from src.labels.labeling import compute_labels
from src.labels.splits import (
    add_chronological_split,
    apply_drive_level_holdout,
    apply_vendor_holdout,
)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")


def main() -> None:
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())

    silver_dir = Path(data_config["silver_dir"])
    gold_dir = Path(data_config["gold_dir"])

    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    if not features_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing gold features or drive metadata; run `make build-features` first."
        )

    features = pl.read_parquet(features_path)
    drive_day_columns = [c for c in ("drive_id", "date", "source_dataset") if c in features.columns]
    drive_days = features.select(drive_day_columns)
    drive_metadata = pl.read_parquet(metadata_path)

    as_of_date = drive_days["date"].max()
    drive_metadata = classify_event_types(drive_metadata, as_of_date=as_of_date)

    labels = compute_labels(
        drive_days, drive_metadata, horizons_days=model_config["horizons_days"]
    )

    splits_cfg = model_config["splits"]
    labels = add_chronological_split(
        labels,
        train_end=splits_cfg["train_end"],
        validation_end=splits_cfg["validation_end"],
    )
    labels = apply_drive_level_holdout(labels)
    if "source_dataset" in drive_days.columns:
        labels = labels.join(
            drive_days.select(["drive_id", "date", "source_dataset"]),
            on=["drive_id", "date"],
            how="left",
        )
    labels = apply_vendor_holdout(labels)

    labels_dir = gold_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(labels_dir / "part.parquet", compression="zstd")

    imbalance_report = {
        "class_distribution": class_distribution_report(labels),
        "scale_pos_weight_by_horizon": {
            h: compute_scale_pos_weight(labels.filter(pl.col("horizon_days") == h))
            for h in model_config["horizons_days"]
        },
    }
    audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "label_imbalance_report.json").write_text(
        json.dumps(imbalance_report, indent=2, default=str)
    )

    print(f"Wrote {labels.height} label rows to {labels_dir}")
    print(f"Wrote imbalance report to {audit_dir / 'label_imbalance_report.json'}")


if __name__ == "__main__":
    main()
