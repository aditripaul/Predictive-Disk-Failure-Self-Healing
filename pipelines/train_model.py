"""Entry point for `make train`.

Assembles the training frame for the primary horizon, trains a LightGBM
model with class-imbalance weighting, tunes a precision-first threshold on
the validation split, evaluates on validation and test, and logs metrics
plus the model to MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import mlflow
from src.labels.dataset_version import latest_dataset_version
from src.labels.event_types import FAILURE_EVENT_TYPES
from src.logging_config import configure_logging, get_logger
from src.models.evaluation import (
    compute_auprc,
    compute_warning_lead_time_days,
    evaluate_at_threshold,
)
from src.models.explainability import (
    build_explainer,
    compute_shap_values,
    global_feature_importance,
)
from src.models.features import assemble_training_frame, select_feature_columns
from src.models.hyperparameter_tuning import tune_lightgbm_hyperparameters
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.model_card import build_model_card, render_model_card_markdown
from src.models.threshold import tune_threshold_for_precision
from src.models.training import predict_proba_positive, train_lightgbm
from src.models.xgboost_training import train_xgboost

SHAP_BACKGROUND_SAMPLE_SIZE = 100

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())

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

    x_val = splits["validation"].select(feature_columns).fill_null(0.0).to_numpy()
    y_val = splits["validation"]["label"].to_numpy()

    model_type = model_config["model"].get("type", "lightgbm")

    with mlflow.start_run():
        results: dict = {
            "horizon_days": horizon_days,
            "feature_columns": feature_columns,
            "model_type": model_type,
        }

        hp_search_cfg = model_config.get("hyperparameter_search", {})
        model_params = model_config["model"]["params"]
        if model_type == "xgboost" and hp_search_cfg.get("enabled", False):
            raise ValueError(
                "hyperparameter_search is only implemented for model.type: lightgbm "
                "(src/models/hyperparameter_tuning.py); disable it or switch model.type."
            )
        if model_type == "lightgbm" and hp_search_cfg.get("enabled", False):
            search_result = tune_lightgbm_hyperparameters(
                x_train,
                y_train,
                x_val,
                y_val,
                base_params=model_config["model"]["params"],
                search_space=hp_search_cfg["search_space"],
                n_trials=hp_search_cfg.get("n_trials", 20),
                subsample_fraction=hp_search_cfg.get("subsample_fraction", 0.3),
                seed=hp_search_cfg.get("seed", 0),
            )
            model_params = search_result["best_params"]
            results["hyperparameter_search"] = {
                "best_params": search_result["best_params"],
                "best_value": search_result["best_value"],
                "n_trials": search_result["n_trials"],
                "subsample_size": search_result["subsample_size"],
            }
            mlflow.log_params({f"tuned_{k}": v for k, v in search_result["best_params"].items()})

        # Retrain on the FULL training partition with the (possibly tuned)
        # params - the search above only ever sees a subsample.
        if model_type == "xgboost":
            model = train_xgboost(x_train, y_train, params=model_params)
        elif model_type == "lightgbm":
            model = train_lightgbm(x_train, y_train, params=model_params)
        else:
            raise ValueError(f"Unknown model.type: {model_type!r} (expected lightgbm or xgboost)")

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

        # Logistic Regression sanity baseline (docs/project_plan.md Phase 6
        # Model Candidates: "Interpretable sanity baseline") - confirms the
        # primary model is actually adding value over a simple linear model,
        # rather than assuming it. Never used for production decisions.
        baseline_model = train_logistic_regression_baseline(x_train, y_train)
        baseline_val_scores = predict_proba_positive(baseline_model, x_val)
        baseline_test_scores = predict_proba_positive(baseline_model, x_test)
        results["logistic_regression_baseline"] = {
            "validation_auprc": compute_auprc(y_val, baseline_val_scores),
            "test_auprc": compute_auprc(y_test, baseline_test_scores),
        }
        mlflow.log_metric(
            "baseline_validation_auprc",
            results["logistic_regression_baseline"]["validation_auprc"],
        )
        mlflow.log_metric(
            "baseline_test_auprc", results["logistic_regression_baseline"]["test_auprc"]
        )

        # Warning lead time (docs/dataset_strategy.md section 15): restrict
        # to drive-days belonging to drives with a genuine failure event
        # before asking "how early did the score cross threshold".
        test_with_scores = splits["test"].with_columns(pl.Series("_p_fail_score", test_scores))
        failing_test_rows = test_with_scores.filter(
            pl.col("event_type").is_in(list(FAILURE_EVENT_TYPES))
        )
        results["test_warning_lead_time"] = compute_warning_lead_time_days(
            failing_test_rows,
            score_column="_p_fail_score",
            threshold=threshold_result["threshold"],
        )

        mlflow.log_params({"horizon_days": horizon_days, **model_params})
        mlflow.log_metric("validation_auprc", results["validation_metrics"]["auprc"])
        mlflow.log_metric("validation_precision", results["validation_metrics"]["precision"])
        mlflow.log_metric("validation_recall", results["validation_metrics"]["recall"])
        mlflow.log_metric("test_auprc", results["test_metrics"]["auprc"])
        mlflow.log_metric(
            "test_expected_calibration_error",
            results["test_metrics"]["calibration"]["expected_calibration_error"],
        )
        mlflow.log_metric("test_brier_score", results["test_metrics"]["calibration"]["brier_score"])
        if results["test_warning_lead_time"]["mean_lead_time_days"] is not None:
            mlflow.log_metric(
                "test_mean_warning_lead_time_days",
                results["test_warning_lead_time"]["mean_lead_time_days"],
            )
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.lightgbm.log_model(model, name="model")

        # SHAP global feature importance (docs/design_goal.md "Explainability
        # by default"): background sample keeps TreeExplainer fast even on a
        # large training set.
        rng = np.random.default_rng(0)
        background_size = min(SHAP_BACKGROUND_SAMPLE_SIZE, x_train.shape[0])
        background = x_train[rng.choice(x_train.shape[0], size=background_size, replace=False)]
        explainer = build_explainer(model, background)
        shap_values = compute_shap_values(explainer, x_val)
        feature_importance = global_feature_importance(shap_values, feature_columns)
        results["shap_global_feature_importance"] = feature_importance[:20]

        audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
        audit_dir.mkdir(parents=True, exist_ok=True)
        report_path = audit_dir / "model_evaluation_report.json"
        report_path.write_text(json.dumps(results, indent=2, default=str))

        shap_report_path = audit_dir / "shap_feature_importance.json"
        shap_report_path.write_text(json.dumps(feature_importance, indent=2, default=str))
        mlflow.log_artifact(str(shap_report_path))

        dataset_version_record = latest_dataset_version(Path(data_config["audit_dir"]))
        model_card = build_model_card(
            horizon_days=horizon_days,
            model_params=model_params,
            model_version=model_config["version"],
            model_type=model_type,
            feature_registry_version=features_config["version"],
            dataset_version=(
                dataset_version_record["version_id"] if dataset_version_record else None
            ),
            feature_columns=feature_columns,
            threshold_result=threshold_result,
            validation_metrics=results["validation_metrics"],
            test_metrics=results["test_metrics"],
            shap_top_features=feature_importance[:20],
            train_row_count=x_train.shape[0],
            test_warning_lead_time=results["test_warning_lead_time"],
            logistic_regression_baseline=results["logistic_regression_baseline"],
        )
        model_cards_dir = Path(data_config["audit_dir"]) / "model_cards"
        model_cards_dir.mkdir(parents=True, exist_ok=True)
        model_card_json_path = model_cards_dir / f"v{model_config['version']}_h{horizon_days}d.json"
        model_card_md_path = model_cards_dir / f"v{model_config['version']}_h{horizon_days}d.md"
        model_card_json_path.write_text(json.dumps(model_card, indent=2, default=str))
        model_card_md_path.write_text(render_model_card_markdown(model_card))
        mlflow.log_artifact(str(model_card_json_path))
        mlflow.log_artifact(str(model_card_md_path))

        logger.info(
            "model_trained", horizon_days=horizon_days, train_row_count=x_train.shape[0]
        )
        logger.info("threshold_tuned", **threshold_result)
        logger.info("validation_metrics", **results["validation_metrics"])
        logger.info("test_metrics", **results["test_metrics"])
        logger.info(
            "logistic_regression_baseline", **results["logistic_regression_baseline"]
        )
        logger.info("test_warning_lead_time", **results["test_warning_lead_time"])
        logger.info("top_shap_features", features=feature_importance[:5])
        logger.info("evaluation_report_written", path=str(report_path))
        logger.info("shap_feature_importance_written", path=str(shap_report_path))
        logger.info("model_card_written", path=str(model_card_md_path))


if __name__ == "__main__":
    main()
