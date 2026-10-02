"""Entry point for `make train`.

Assembles the training frame for the primary horizon, trains a LightGBM
model with class-imbalance weighting, tunes a precision-first threshold on
the validation split, evaluates on validation and test, and logs metrics
plus the model to MLflow.

Assembling the training frame (joining the ~196-column, ~10GB-for-one-
month gold feature table to the label table) runs in its own subprocess
(`--stage assemble`), for the same reason batches are isolated into their
own subprocesses in pipelines/build_gold_features.py: Polars is built on
jemalloc, which on 64-bit Linux defaults to retaining freed virtual memory
for reuse rather than returning it to the OS. RLIMIT_AS (this pipeline's
memory cap, src/resource_limits.py) constrains mapped address space, not
resident memory, so within one process it tracks the high-water mark of
everything that process has EVER allocated, not what's currently live.
Chunking the join (src/models/features.py::assemble_training_frame) keeps
that join itself under the cap, but every subsequent step in the same
process - splits, model training, SHAP - would otherwise inherit the
join's high-water mark on top of its own needs, and on real data that was
enough to fail on allocations of a few hundred KB right after the join
finished successfully. Exiting a process unconditionally unmaps its
entire address space regardless of what the allocator was retaining, so
the assemble subprocess writes the joined frame to a temp Parquet file
and exits; the top-level orchestrator (this function with no `--stage`)
never touches gold_features or the label table itself, so its own address
space stays clean for everything that runs after.

Once the join itself stopped crashing, the next failure on real data
moved one step downstream: `main()` used to collect all three splits
(train/validation/test) as full eager, full-width DataFrames in one go,
then build a second, similarly sized numpy copy of each while the
DataFrame was still alive - several times the size of even the largest
single split, at a scale (tens of millions of rows) where a single clean
copy of the train split alone was already multiple GB. `main()` now
collects each split ONE AT A TIME, each projected at scan time to only
the columns it actually needs (see the split-extraction block below), and
additionally caps the train split's row count (DEFAULT_MAX_TRAIN_ROWS /
`model.max_train_rows`, _subsampled_train_lazy) - keeping every failure
(positive-label) row and randomly subsampling negatives down to the cap -
since even a single copy of a real quarter's full train split no longer
reliably fits regardless of how many redundant copies are removed.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import tempfile
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

#: Fallback when `model.max_train_rows` is absent from configs/model.yaml.
#: At ~185 float32 feature columns, 5,000,000 rows is ~3.7GB as a single
#: matrix - comfortable under the 20GB default memory cap even with the
#: transient second copy `feature_matrix`'s to_numpy() briefly needs plus
#: LightGBM's own histogram-building overhead. A real quarter's train
#: split is tens of millions of rows, well above this - see
#: _subsampled_train_lazy for how the cap is applied. Set
#: `model.max_train_rows: null` explicitly in configs/model.yaml to train
#: on every row instead (only advisable with a much higher memory cap or
#: a smaller dataset).
DEFAULT_MAX_TRAIN_ROWS = 5_000_000

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


def _require_gold_inputs(data_config: dict) -> tuple[Path, Path]:
    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    labels_path = gold_dir / "labels" / "part.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing gold features or labels; run `make build-features` and "
            "`make build-labels` first."
        )
    return features_path, labels_path


def _stage_assemble(frame_path: Path) -> None:
    """Subprocess stage: joins gold features to the label table for the
    primary horizon and writes the result to `frame_path`. See the module
    docstring for why this runs in its own process rather than as the
    first step of `main()`."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_path, labels_path = _require_gold_inputs(data_config)

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
    # gold_features_path lets assemble_training_frame join by gold
    # features' own existing row groups (one per batch, already written
    # by build_gold_features.py) instead of re-deriving batches with a
    # fresh hash filter, which - even after being fixed to collect each
    # side of the join separately - still meant rescanning the entire
    # ~196-column, ~10GB (one real quarter) file once per batch just to
    # evaluate an unprunable filter. See src/models/features.py's
    # join_by_native_row_groups for the full explanation.
    #
    # out_path=frame_path: the joined training frame can itself be tens
    # of millions of rows at fleet scale, and this function was just
    # going to write it to frame_path immediately after anyway - passing
    # the destination straight through means it's written incrementally,
    # one batch at a time, and never exists as a single eager DataFrame
    # in this process at all. Against real Q1 data, this was the actual
    # remaining crash after every earlier fix in this join's history:
    # every batch's own join succeeded, and reassembling them all into
    # one eager DataFrame right before writing it straight back out
    # (which frame.write_parquet(...) below used to do) is what failed.
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    assemble_training_frame(
        gold_features,
        labels,
        horizon_days=horizon_days,
        label_columns=LABEL_COLUMNS_USED,
        chunk_rows=data_config.get("resource_limits", {}).get(
            "training_join_chunk_rows", DEFAULT_TRAINING_JOIN_CHUNK_ROWS
        ),
        gold_features_path=features_path,
        out_path=frame_path,
    )
    # row_count/column_count read back from the written file - cheap,
    # metadata-only for the count - rather than from the DataFrame this
    # function no longer returns.
    row_count = pl.scan_parquet(frame_path).select(pl.len()).collect().item()
    column_count = len(pl.scan_parquet(frame_path).collect_schema().names())
    _log_stage("training_frame_assembled", t0, row_count=row_count, column_count=column_count)


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage - a new
    process, and therefore a new address space, regardless of what the
    calling process's allocator has retained. See the module docstring."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def _subsampled_train_lazy(
    train_lazy: pl.LazyFrame, *, max_rows: int | None, seed: int = 0
) -> pl.LazyFrame:
    """Caps the primary model's training row count for fleet-scale data,
    where a single copy of the full train split's feature matrix - let
    alone the second, similarly sized copy `feature_matrix`'s to_numpy()
    transiently needs - no longer fits in the memory cap regardless of how
    efficiently the rest of the pipeline avoids redundant copies. See
    DEFAULT_MAX_TRAIN_ROWS.

    Keeps every positive (failure) row - disk failure is the rare class,
    and subsampling can't afford to lose it - and randomly keeps a
    `hash(drive_id, date)`-derived fraction of the negative rows to reach
    `max_rows`. This is the same deterministic, reproducible,
    engine-native sampling idiom used for drive-batching elsewhere in this
    codebase (see src/models/features.py's `hash(drive_id) % n_batches`),
    rather than reading the label column into Python to sample row
    indices, which would itself require materializing a column across the
    whole (uncapped) train split just to decide how to cap it."""
    if max_rows is None:
        return train_lazy
    label_counts = train_lazy.group_by("label").agg(pl.len().alias("count")).collect()
    counts = dict(
        zip(label_counts["label"].to_list(), label_counts["count"].to_list(), strict=True)
    )
    n_positive = counts.get(1, 0)
    n_negative = counts.get(0, 0)
    if n_positive + n_negative <= max_rows:
        return train_lazy
    target_negative = max(0, max_rows - n_positive)
    keep_fraction = min(1.0, target_negative / n_negative) if n_negative else 1.0
    hash_bucket = 1_000_000
    row_hash = (pl.col("drive_id").hash(seed=seed) ^ pl.col("date").hash(seed=seed)) % hash_bucket
    return train_lazy.filter((pl.col("label") == 1) | (row_hash < int(keep_fraction * hash_bucket)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["assemble"])
    parser.add_argument("--frame-path", type=Path)
    args = parser.parse_args()

    if args.stage == "assemble":
        _stage_assemble(args.frame_path)
        return

    # No --stage: the top-level orchestrator. It never itself scans
    # gold_features or the label table - only the assemble subprocess it
    # spawns does - so its own address space stays clean for splits, model
    # training, and SHAP, no matter how large the join was.
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())
    _require_gold_inputs(data_config)

    horizon_days = model_config["primary_horizon_days"]
    frame_tmp_dir = Path(tempfile.mkdtemp(prefix="train_model_frame_"))
    frame_path = frame_tmp_dir / "frame.parquet"
    try:
        _run_stage("--stage", "assemble", "--frame-path", str(frame_path))

        # Split row counts read via a projection to just `split` - one
        # narrow column, cheap regardless of the frame's ~196-column
        # width - rather than collecting any split to check it's
        # non-empty.
        split_counts_df = (
            pl.scan_parquet(frame_path).group_by("split").agg(pl.len().alias("count")).collect()
        )
        if split_counts_df.height == 0:
            raise ValueError(
                f"No rows with an observed (non-censored) {horizon_days}-day label. "
                "This is expected against the synthetic stub dataset, which has too "
                "short a history for any row to reach horizon observability - it will "
                "resolve once real Backblaze data (Phase 1) is ingested."
            )
        split_counts = dict(
            zip(
                split_counts_df["split"].to_list(),
                split_counts_df["count"].to_list(),
                strict=True,
            )
        )
        empty_splits = [
            name for name in ("train", "validation", "test") if split_counts.get(name, 0) == 0
        ]
        if empty_splits:
            raise ValueError(
                f"Split(s) {empty_splits} have no observed-label rows for "
                f"horizon={horizon_days}d. Ensure the chronological split boundaries "
                "in configs/model.yaml align with the ingested data's date range."
            )

        # A schema-only read (no rows) is enough for select_feature_columns,
        # which only inspects `.columns`/`.dtypes` - the assembled frame
        # itself is never read into this process as one eager object.
        feature_columns = select_feature_columns(pl.scan_parquet(frame_path).limit(0).collect())

        # Splits are collected ONE AT A TIME below, each projected down to
        # only the columns it actually needs, rather than all three
        # collected together as full ~196-column frames (the previous
        # approach). At fleet scale (tens of millions of rows for a real
        # quarter) that meant peak memory was several times the size of
        # even the largest single split: all three full-width splits alive
        # simultaneously, then a second, similarly sized numpy copy per
        # split built while the source Polars frame was STILL alive. Doing
        # one split at a time, narrowed to just the columns it needs,
        # bounds peak memory to roughly one split's worth instead. The
        # train split is additionally capped - see DEFAULT_MAX_TRAIN_ROWS
        # and _subsampled_train_lazy - because even a single clean copy of
        # every real quarter's full train split no longer fits in the
        # memory cap, regardless of how many redundant copies are removed.
        t0 = time.perf_counter()
        max_train_rows = model_config["model"].get("max_train_rows", DEFAULT_MAX_TRAIN_ROWS)
        train_lazy = _subsampled_train_lazy(
            pl.scan_parquet(frame_path).filter(pl.col("split") == "train"),
            max_rows=max_train_rows,
        )
        train_df = train_lazy.select([*feature_columns, "label"]).collect()
        x_train = feature_matrix(train_df, feature_columns)
        y_train = train_df["label"].to_numpy()
        train_row_count = train_df.height
        # Same reasoning as `frame` above: once copied into numpy arrays,
        # `train_df` (often the largest split, and still every feature
        # column wide) is dead weight for the rest of the run - model
        # training, SHAP, and model-card generation never touch it again.
        del train_df
        gc.collect()

        mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(model_config["mlflow"]["experiment_name"])

        validation_df = (
            pl.scan_parquet(frame_path)
            .filter(pl.col("split") == "validation")
            .select([*feature_columns, "label"])
            .collect()
        )
        x_val = feature_matrix(validation_df, feature_columns)
        y_val = validation_df["label"].to_numpy()
        del validation_df
        gc.collect()
        _log_stage(
            "splits_extracted",
            t0,
            train=train_row_count,
            train_before_cap=split_counts["train"],
            validation=split_counts["validation"],
            test=split_counts["test"],
        )

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
                mlflow.log_params(
                    {f"tuned_{k}": v for k, v in search_result["best_params"].items()}
                )

            # Retrain on the FULL training partition with the (possibly tuned)
            # params - the search above only ever sees a subsample.
            t0 = time.perf_counter()
            if model_type == "xgboost":
                model = train_xgboost(x_train, y_train, params=model_params)
            elif model_type == "lightgbm":
                model = train_lightgbm(x_train, y_train, params=model_params)
            else:
                raise ValueError(
                    f"Unknown model.type: {model_type!r} (expected lightgbm or xgboost)"
                )
            _log_stage(
                "model_trained", t0, model_type=model_type, train_row_count=x_train.shape[0]
            )

            val_scores = predict_proba_positive(model, x_val)
            threshold_result = tune_threshold_for_precision(
                y_val, val_scores, target_precision=model_config["threshold"]["target_precision"]
            )
            results["threshold"] = threshold_result
            results["validation_metrics"] = evaluate_at_threshold(
                y_val, val_scores, threshold_result["threshold"]
            )

            # Collected here, right before its first use, rather than
            # up front with train/validation: test is only needed from
            # this point on, so collecting it earlier would just extend
            # how long it overlaps with x_train/train_df in memory for no
            # benefit. Projected to feature_columns plus the handful of
            # extra columns compute_warning_lead_time_days needs below
            # (drive_id, event_type, days_to_event), not every column in
            # the assembled frame.
            test_df = (
                pl.scan_parquet(frame_path)
                .filter(pl.col("split") == "test")
                .select([*feature_columns, "label", "drive_id", "event_type", "days_to_event"])
                .collect()
            )
            x_test = feature_matrix(test_df, feature_columns)
            y_test = test_df["label"].to_numpy()
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
                        "smote"
                        if smote_val_auprc > class_weighting_val_auprc
                        else "class_weighting"
                    ),
                    "smote_resampled_row_count": len(y_smote),
                }
                mlflow.log_metric("smote_validation_auprc", smote_val_auprc)
                logger.info("smote_comparison", **results["smote_comparison"])

            # Warning lead time (docs/dataset_strategy.md section 15): restrict
            # to drive-days belonging to drives with a genuine failure event
            # before asking "how early did the score cross threshold".
            test_with_scores = test_df.with_columns(pl.Series("_p_fail_score", test_scores))
            failing_test_rows = test_with_scores.filter(
                pl.col("event_type").is_in(list(FAILURE_EVENT_TYPES))
            )
            results["test_warning_lead_time"] = compute_warning_lead_time_days(
                failing_test_rows,
                score_column="_p_fail_score",
                threshold=threshold_result["threshold"],
            )
            del test_df, test_with_scores, failing_test_rows
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
            mlflow.log_metric(
                "test_brier_score", results["test_metrics"]["calibration"]["brier_score"]
            )
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
            model_card_stem = f"v{model_config['version']}_h{horizon_days}d"
            model_card_json_path = model_cards_dir / f"{model_card_stem}.json"
            model_card_md_path = model_cards_dir / f"{model_card_stem}.md"
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
    finally:
        shutil.rmtree(frame_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
