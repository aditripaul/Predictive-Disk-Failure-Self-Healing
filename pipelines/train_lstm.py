"""Entry point for `make train-lstm` (optional comparison branch,
docs/dataset_strategy.md section 16.2 "LSTM / Sequence Branch").

Trains a small LSTM classifier on the sequence tensors `make
build-sequences` writes and reports its AUPRC alongside the primary
(tree-based) model's - "The LSTM branch should be compared against the
tree-based baseline, not assumed to be superior." Requires the `torch`
extra: `uv sync --extra torch`. Never imported by, or required for,
`pipelines/train_model.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.features.sequences import (
    apply_time_decay_weights,
    compute_time_decay_weights,
    load_sequence_tensors,
)
from src.logging_config import configure_logging, get_logger
from src.models.evaluation import evaluate_at_threshold
from src.models.lstm import predict_proba_positive_lstm, train_lstm
from src.models.threshold import tune_threshold_for_precision
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())
    sequence_cfg = features_config.get("sequences", {})

    gold_dir = Path(data_config["gold_dir"])
    sequences_dir = gold_dir / "sequences"
    labels_path = gold_dir / "labels" / "part.parquet"
    if not (sequences_dir / "sequences.npy").exists():
        raise FileNotFoundError(f"{sequences_dir} not found; run `make build-sequences` first.")
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found; run `make build-labels` first.")

    tensor, metadata, attributes_used = load_sequence_tensors(sequences_dir)
    horizon_days = model_config["primary_horizon_days"]
    labels = pl.read_parquet(labels_path).filter(
        (pl.col("horizon_days") == horizon_days) & pl.col("label").is_not_null()
    )

    # Each tensor row is one drive's sequence "as of" its last observed
    # day; join that back to the label computed for exactly that drive-day
    # so x/y/split all refer to the same sample.
    joined = metadata.with_row_index("tensor_row").join(
        labels.rename({"date": "as_of_date"}),
        on=["drive_id", "as_of_date"],
        how="inner",
    )
    if joined.height == 0:
        raise ValueError(
            "No sequence samples matched a non-censored label for the primary horizon; "
            "run `make build-sequences` and `make build-labels` against real data first."
        )

    row_indices = joined["tensor_row"].to_numpy()
    x = np.asarray(tensor)[row_indices]
    y = joined["label"].to_numpy().astype(np.float32)
    splits = joined["split"].to_list()

    time_decay_lambda = sequence_cfg.get("time_decay_lambda", 0.1)
    weights = compute_time_decay_weights(x.shape[1], lambda_=time_decay_lambda)
    x = apply_time_decay_weights(x, weights)

    split_masks = {
        name: np.array([s == name for s in splits]) for name in ("train", "validation", "test")
    }
    empty_splits = [name for name, mask in split_masks.items() if mask.sum() == 0]
    if empty_splits:
        raise ValueError(
            f"Split(s) {empty_splits} have no matching sequence samples for this horizon."
        )

    x_train, y_train = x[split_masks["train"]], y[split_masks["train"]]
    x_val, y_val = x[split_masks["validation"]], y[split_masks["validation"]]
    x_test, y_test = x[split_masks["test"]], y[split_masks["test"]]

    model = train_lstm(x_train, y_train)

    val_scores = predict_proba_positive_lstm(model, x_val)
    threshold_result = tune_threshold_for_precision(
        y_val, val_scores, target_precision=model_config["threshold"]["target_precision"]
    )
    validation_metrics = evaluate_at_threshold(y_val, val_scores, threshold_result["threshold"])

    test_scores = predict_proba_positive_lstm(model, x_test)
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold_result["threshold"])

    report = {
        "horizon_days": horizon_days,
        "attributes_used": attributes_used,
        "time_steps": x.shape[1],
        "train_row_count": int(x_train.shape[0]),
        "threshold": threshold_result,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_path = audit_dir / "lstm_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    primary_report_path = audit_dir / "model_evaluation_report.json"
    primary_auprc = None
    if primary_report_path.exists():
        primary_auprc = json.loads(primary_report_path.read_text()).get("test_metrics", {}).get(
            "auprc"
        )

    logger.info(
        "lstm_trained",
        test_auprc=test_metrics["auprc"],
        primary_model_test_auprc=primary_auprc,
        recommendation=(
            "lstm"
            if primary_auprc is not None and test_metrics["auprc"] > primary_auprc
            else "tree_based_baseline"
        ),
    )
    logger.info("lstm_evaluation_report_written", path=str(report_path))


if __name__ == "__main__":
    main()
