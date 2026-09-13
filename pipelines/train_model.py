"""Entry point for `make train`.

Assembles the training frame for the primary horizon, trains a LightGBM
model with class-imbalance weighting, tunes a precision-first threshold on
the validation split, evaluates on validation and test, and logs metrics
plus the model to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

import mlflow
from src.models.evaluation import evaluate_at_threshold
from src.models.features import assemble_training_frame, select_feature_columns
from src.models.threshold import tune_threshold_for_precision
from src.models.training import predict_proba_positive, train_lightgbm

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")


def main() -> None:
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())

    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    labels_path = gold_dir / "labels" / "part.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing gold features or labels; run `make build-features` and "
            "`make build-labels` first."
        )

    gold_features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path)

    horizon_days = model_config["primary_horizon_days"]
    frame = assemble_training_frame(gold_features, labels, horizon_days=horizon_days)
    if frame.height == 0:
        raise ValueError(
            f"No rows with an observed (non-censored) {horizon_days}-day label. "
            "This is expected against the synthetic stub dataset, which has too "
            "short a history for any row to reach horizon observability - it will "
            "resolve once real Backblaze data (Phase 1) is ingested."
        )
    feature_columns = select_feature_columns(frame)

    splits = {
        split_name: frame.filter(pl.col("split") == split_name)
        for split_name in ("train", "validation", "test")
    }
    empty_splits = [name for name, split_df in splits.items() if split_df.height == 0]
    if empty_splits:
        raise ValueError(
            f"Split(s) {empty_splits} have no observed-label rows for "
            f"horizon={horizon_days}d. Ensure the chronological split boundaries "
            "in configs/model.yaml align with the ingested data's date range."
        )

    x_train = splits["train"].select(feature_columns).fill_null(0.0).to_numpy()
    y_train = splits["train"]["label"].to_numpy()

    mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(model_config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model = train_lightgbm(x_train, y_train, params=model_config["model"]["params"])

        results: dict = {"horizon_days": horizon_days, "feature_columns": feature_columns}

        x_val = splits["validation"].select(feature_columns).fill_null(0.0).to_numpy()
        y_val = splits["validation"]["label"].to_numpy()
        val_scores = predict_proba_positive(model, x_val)
        threshold_result = tune_threshold_for_precision(
            y_val, val_scores, target_precision=model_config["threshold"]["target_precision"]
        )
        results["threshold"] = threshold_result
        results["validation_metrics"] = evaluate_at_threshold(
            y_val, val_scores, threshold_result["threshold"]
        )

        x_test = splits["test"].select(feature_columns).fill_null(0.0).to_numpy()
        y_test = splits["test"]["label"].to_numpy()
        test_scores = predict_proba_positive(model, x_test)
        results["test_metrics"] = evaluate_at_threshold(
            y_test, test_scores, threshold_result["threshold"]
        )

        mlflow.log_params({"horizon_days": horizon_days, **model_config["model"]["params"]})
        mlflow.log_metric("validation_auprc", results["validation_metrics"]["auprc"])
        mlflow.log_metric("validation_precision", results["validation_metrics"]["precision"])
        mlflow.log_metric("validation_recall", results["validation_metrics"]["recall"])
        mlflow.log_metric("test_auprc", results["test_metrics"]["auprc"])
        mlflow.lightgbm.log_model(model, name="model")

        audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
        audit_dir.mkdir(parents=True, exist_ok=True)
        report_path = audit_dir / "model_evaluation_report.json"
        report_path.write_text(json.dumps(results, indent=2, default=str))

        print(f"Trained model for {horizon_days}-day horizon on {x_train.shape[0]} rows.")
        print(f"Threshold: {threshold_result}")
        print(f"Validation metrics: {results['validation_metrics']}")
        print(f"Test metrics: {results['test_metrics']}")
        print(f"Wrote evaluation report to {report_path}")


if __name__ == "__main__":
    main()
