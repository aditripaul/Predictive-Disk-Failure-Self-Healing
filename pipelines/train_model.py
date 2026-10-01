"""Entry point for `make train`.

Assembles the training frame for the primary horizon, trains a LightGBM
model with class-imbalance weighting, tunes a precision-first threshold on
the validation split, evaluates on validation and test, and logs metrics
plus the model to MLflow.
"""

from __future__ import annotations

import gc
import json
import time
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
from src.models.features import assemble_training_frame, feature_matrix, select_feature_columns
from src.models.hyperparameter_tuning import tune_lightgbm_hyperparameters
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.model_card import build_model_card, render_model_card_markdown
from src.models.smote import apply_smote
from src.models.threshold import tune_threshold_for_precision
from src.models.training import predict_proba_positive, train_lightgbm
from src.models.xgboost_training import train_xgboost
from src.resource_limits import apply_memory_limit_from_config

SHAP_BACKGROUND_SAMPLE_SIZE = 100

#: Fallback when `resource_limits.training_join_chunk_rows` is absent from
#: configs/data.yaml.
DEFAULT_TRAINING_JOIN_CHUNK_ROWS = 2_000_000

#: Label-table columns the training pipeline actually uses: the target
#: (`label`), the split assignment, and `event_type`/`days_to_event` for
#: the warning-lead-time metric - plus the join keys. The label table has
#: 12 columns and is ~3x the drive-day count (one row per horizon), so
#: projecting to these before joining avoids carrying ~5.6GB of unused
#: columns through the join at fleet scale.
LABEL_COLUMNS_USED = [
    "drive_id",
    "date",
    "label",
    "split",
    "event_type",
    "days_to_event",
]

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"train_model_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def _subsample_rows(
    x: np.ndarray, y: np.ndarray, *, max_rows: int | None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)` unchanged when they already fit within `max_rows`, else a
    fixed-seed random row sample of exactly `max_rows` rows.

    Returns the inputs themselves (not a copy) in the common case, so a
    dataset under the cap costs nothing - the cap only ever materializes a
    sample at fleet scale. `max_rows=None` disables the cap entirely."""
    if max_rows is None or x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    selected = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[selected], y[selected]


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
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

    # Scanned lazily, not pl.read_parquet'd: gold features is ~196 columns
    # and ~10GB for one month of real data, but assemble_training_frame's
    # join only keeps one horizon's observed-label rows - reading the whole
    # table eagerly first would materialize all ~10GB before the join gets
    # a chance to discard most of it (see src/models/features.py).
    gold_features = pl.scan_parquet(features_path)
    # Scanned lazily and projected to LABEL_COLUMNS_USED inside
    # assemble_training_frame, rather than pl.read_parquet'd: the label
    # table is one row per drive-day per horizon (31.4M rows, ~5.6GB for
    # one month of real data) and only one horizon and 6 of its 12
    # columns are ever used, so reading it whole put ~5GB of dead columns
    # alive alongside the join's own output.
    labels = pl.scan_parquet(labels_path)

    horizon_days = model_config["primary_horizon_days"]
    t0 = time.perf_counter()
    frame = assemble_training_frame(
        gold_features,
        labels,
        horizon_days=horizon_days,
        label_columns=LABEL_COLUMNS_USED,
        chunk_rows=data_config.get("resource_limits", {}).get(
            "training_join_chunk_rows", DEFAULT_TRAINING_JOIN_CHUNK_ROWS
        ),
    )
    _log_stage(
        "training_frame_assembled", t0, row_count=frame.height, column_count=len(frame.columns)
    )
    del labels
    gc.collect()
    if frame.height == 0:
        raise ValueError(
            f"No rows with an observed (non-censored) {horizon_days}-day label. "
            "This is expected against the synthetic stub dataset, which has too "
            "short a history for any row to reach horizon observability - it will "
            "resolve once real Backblaze data (Phase 1) is ingested."
        )
    feature_columns = select_feature_columns(frame)

    t0 = time.perf_counter()
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
    _log_stage(
        "splits_extracted", t0, **{name: split_df.height for name, split_df in splits.items()}
    )
    # `frame` is never used again past this point - only `splits[...]` is.
    # Deleting it here (rather than leaving it referenced by this function's
    # own locals for the rest of training/SHAP/model-card generation, which
    # all still need to run) matters exactly as much as it did for
    # canonical_long/drive_day in build_gold_features.py and gold_features
    # in build_labels.py: it's ~196 columns wide, and a live local variable
    # reference is enough to keep it resident regardless of whether
    # anything downstream still reads it.
    del frame
    gc.collect()

    x_train = feature_matrix(splits["train"], feature_columns)
    y_train = splits["train"]["label"].to_numpy()
    # Same reasoning as `frame` above: once copied into numpy arrays,
    # `splits["train"]` (often the largest split, and still 196 columns
    # wide) is dead weight for the rest of the run - model training, SHAP,
    # and model-card generation never touch the Polars frame again.
    del splits["train"]
    gc.collect()

    mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(model_config["mlflow"]["experiment_name"])

    x_val = feature_matrix(splits["validation"], feature_columns)
    y_val = splits["validation"]["label"].to_numpy()
    del splits["validation"]
    gc.collect()

    model_type = model_config["model"].get("type", "lightgbm")
    diagnostics_cfg = model_config.get("diagnostics", {})

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
        t0 = time.perf_counter()
        if model_type == "xgboost":
            model = train_xgboost(x_train, y_train, params=model_params)
        elif model_type == "lightgbm":
            model = train_lightgbm(x_train, y_train, params=model_params)
        else:
            raise ValueError(f"Unknown model.type: {model_type!r} (expected lightgbm or xgboost)")
        _log_stage("model_trained", t0, model_type=model_type, train_row_count=x_train.shape[0])

        val_scores = predict_proba_positive(model, x_val)
        threshold_result = tune_threshold_for_precision(
            y_val, val_scores, target_precision=model_config["threshold"]["target_precision"]
        )
        results["threshold"] = threshold_result
        results["validation_metrics"] = evaluate_at_threshold(
            y_val, val_scores, threshold_result["threshold"]
        )

        x_test = feature_matrix(splits["test"], feature_columns)
        y_test = splits["test"]["label"].to_numpy()
        test_scores = predict_proba_positive(model, x_test)
        results["test_metrics"] = evaluate_at_threshold(
            y_test, test_scores, threshold_result["threshold"]
        )

        # Logistic Regression sanity baseline (docs/project_plan.md Phase 6
        # Model Candidates: "Interpretable sanity baseline") - confirms the
        # primary model is actually adding value over a simple linear model,
        # rather than assuming it. Never used for production decisions.
        #
        # Fitted on at most `diagnostics.baseline_max_rows` rows: sklearn's
        # StandardScaler and LogisticRegression each force a float64 copy
        # of their input, so an uncapped fit needs ~2x the (already
        # fleet-scale) training matrix on top of the primary model's own.
        # This is a non-production sanity check, so a bounded, fixed-seed
        # sample is the right trade; the cap is high enough that smaller
        # datasets are fitted in full and unaffected.
        t0 = time.perf_counter()
        x_baseline, y_baseline = _subsample_rows(
            x_train, y_train, max_rows=diagnostics_cfg.get("baseline_max_rows", 1_000_000)
        )
        baseline_model = train_logistic_regression_baseline(x_baseline, y_baseline)
        baseline_val_scores = predict_proba_positive(baseline_model, x_val)
        baseline_test_scores = predict_proba_positive(baseline_model, x_test)
        results["logistic_regression_baseline"] = {
            "validation_auprc": compute_auprc(y_val, baseline_val_scores),
            "test_auprc": compute_auprc(y_test, baseline_test_scores),
            "train_row_count": int(x_baseline.shape[0]),
        }
        _log_stage("baseline_trained", t0, baseline_row_count=int(x_baseline.shape[0]))
        del x_baseline, y_baseline
        gc.collect()
        mlflow.log_metric(
            "baseline_validation_auprc",
            results["logistic_regression_baseline"]["validation_auprc"],
        )
        mlflow.log_metric(
            "baseline_test_auprc", results["logistic_regression_baseline"]["test_auprc"]
        )

        # Subsampled SMOTE comparison (docs/dataset_strategy.md section 15
        # Imbalance Mitigations #3) - a SECOND, comparison-only model, never
        # used in place of the primary (class-weighted) one. SMOTE-resampled
        # data is already balanced, so class weighting is turned off for
        # this comparison fit to avoid double-compensating.
        smote_cfg = model_config.get("smote_comparison", {})
        if smote_cfg.get("enabled", False):
            x_smote, y_smote = apply_smote(
                x_train,
                y_train,
                subsample_fraction=smote_cfg.get("subsample_fraction", 1.0),
                seed=smote_cfg.get("seed", 0),
            )
            smote_params = dict(model_params)
            if model_type == "lightgbm":
                smote_params["is_unbalance"] = False
            smote_model = (
                train_xgboost(x_smote, y_smote, params=smote_params)
                if model_type == "xgboost"
                else train_lightgbm(x_smote, y_smote, params=smote_params)
            )
            smote_val_auprc = compute_auprc(y_val, predict_proba_positive(smote_model, x_val))
            class_weighting_val_auprc = results["validation_metrics"]["auprc"]
            results["smote_comparison"] = {
                "smote_validation_auprc": smote_val_auprc,
                "class_weighting_validation_auprc": class_weighting_val_auprc,
                "recommendation": (
                    "smote" if smote_val_auprc > class_weighting_val_auprc else "class_weighting"
                ),
                "smote_resampled_row_count": len(y_smote),
            }
            mlflow.log_metric("smote_validation_auprc", smote_val_auprc)
            logger.info("smote_comparison", **results["smote_comparison"])

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
        del splits["test"], test_with_scores, failing_test_rows
        gc.collect()

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
        # Explained on at most `diagnostics.shap_max_rows` rows: the global
        # importance below is a mean of |shap value| per feature, which
        # converges long before millions of rows, while TreeExplainer
        # materializes a full (rows x features) float64 array to get there.
        t0 = time.perf_counter()
        rng = np.random.default_rng(0)
        background_size = min(SHAP_BACKGROUND_SAMPLE_SIZE, x_train.shape[0])
        background = x_train[rng.choice(x_train.shape[0], size=background_size, replace=False)]
        x_shap, _ = _subsample_rows(
            x_val, y_val, max_rows=diagnostics_cfg.get("shap_max_rows", 200_000)
        )
        explainer = build_explainer(model, background)
        shap_values = compute_shap_values(explainer, x_shap)
        feature_importance = global_feature_importance(shap_values, feature_columns)
        results["shap_global_feature_importance"] = feature_importance[:20]
        _log_stage("shap_computed", t0, explained_row_count=int(x_shap.shape[0]))
        del x_shap, shap_values
        gc.collect()

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
            smote_comparison=results.get("smote_comparison"),
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
