"""Dev tool (`make experiment-model`): compares LightGBM variants on the
arrays a previous `make train TRAIN_ARGS=--keep-work-dir` left behind, so
a modelling idea is judged in minutes instead of re-running the whole
join. Not part of the production pipeline; it never writes models, MLflow
runs, reports or model cards.

Every variant is trained on the (capped) train split, early-stopped on a
validation subsample, and scored at ROW and DRIVE level on validation and
test. Thresholds are always chosen on validation and applied to test.
Excluded features are zeroed in place (constant columns are never split
on) and restored afterwards, so a variant costs no extra copy of the
multi-GB matrices.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
import yaml
from sklearn.metrics import precision_recall_curve

from src.logging_config import configure_logging, get_logger
from src.models.alerting import SMOOTHERS, analyze_false_alarms, smooth_scores
from src.models.evaluation import (
    compute_auprc,
    drive_level_metrics,
    drive_level_table,
    precision_at_recall_table,  # noqa: I001
)
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.threshold import tune_drive_level_threshold
from src.models.training import (
    PREDICT_CHUNK_ROWS,
    early_stopping_subset,
    predict_proba_positive,
)
from src.resource_limits import apply_memory_limit_from_config

logger = get_logger(__name__)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")

#: Columns that identify a drive or the calendar (monotone per drive) rather
#: than describe its health - candidates for memorization.
IDENTITY_COLUMNS = {
    "drive_age_days",
    "drive_age_days_squared",
    "power_on_hours",
    "capacity_gb",
    "capacity_bytes",
}
_WINDOWED = re.compile(r"_(\d+)d_")

REGULARIZED = {
    "min_child_samples": 200,
    "reg_lambda": 10.0,
    "max_delta_step": 1.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "learning_rate": 0.05,
    "n_estimators": 1000,
}

#: XGBoost counterpart of REGULARIZED (hist tree method, same regularization
#: strength in XGBoost's names). "_engine" routes run_variant to XGBoost.
XGBOOST_PARAMS = {
    "_engine": "xgboost",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "min_child_weight": 20,
    "reg_lambda": 10.0,
    "max_delta_step": 1.0,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "tree_method": "hist",
}

#: name -> (extra LightGBM params, features to exclude, weighting)
#: exclude: none | identity | non_windowed | new_features | secondary | lifetime
#: weighting: is_unbalance | spw[:power] | drive
VARIANTS: dict[str, tuple[dict[str, Any], str, str]] = {
    "current": ({}, "none", "is_unbalance"),
    "regularized": (REGULARIZED, "none", "is_unbalance"),
    "reg_spw": (REGULARIZED, "none", "spw"),
    "reg_spw_no_identity": (REGULARIZED, "identity", "spw"),
    "reg_spw_windowed_only": (REGULARIZED, "non_windowed", "spw"),
    "reg_driveweight_no_identity": (REGULARIZED, "identity", "drive"),
    # Tuning around the configuration that works (reg_spw = what configs/model.yaml
    # ships): positive weight power, tree size, regularization strength.
    "spw_power_0.25": (REGULARIZED, "none", "spw:0.25"),
    "spw_power_0.75": (REGULARIZED, "none", "spw:0.75"),
    "leaves_15": ({**REGULARIZED, "num_leaves": 15}, "none", "spw"),
    "leaves_63": ({**REGULARIZED, "num_leaves": 63}, "none", "spw"),
    "min_child_1000": ({**REGULARIZED, "min_child_samples": 1000}, "none", "spw"),
    # Feature-group ablations on the same data build: the earlier feature set,
    # and the new set without its age or context columns.
    "reg_spw_old_features": (REGULARIZED, "new_features", "spw"),
    "reg_spw_no_lifetime": (REGULARIZED, "lifetime", "spw"),
    "reg_spw_no_secondary": (REGULARIZED, "secondary", "spw"),
    # Same data, weights and early stopping (average precision) with XGBoost,
    # to check the precision ceiling is not specific to LightGBM.
    "xgboost": (XGBOOST_PARAMS, "none", "spw"),
    "xgboost_deep": ({**XGBOOST_PARAMS, "max_depth": 8, "min_child_weight": 5}, "none", "spw"),
    "strong_reg": (
        {**REGULARIZED, "min_child_samples": 500, "reg_lambda": 50.0},
        "none",
        "spw",
    ),
}


def _find_work_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    configured = data_config.get("resource_limits", {}).get("scratch_dir")
    base = Path(configured) if configured else Path(data_config["gold_dir"]).parent / "tmp"
    candidates = sorted(base.glob("train_model_frame_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No kept work dir under {base}; run `make train TRAIN_ARGS=--keep-work-dir`."
        )
    return candidates[-1]


#: Context attributes added on 2026-10-06 (configs/features.yaml
#: secondary_smart_attributes), by column-name prefix.
SECONDARY_PREFIXES = (
    "seek_error_rate",
    "power_on_hours",
    "temperature_celsius",
    "udma_crc_error_count",
)
#: Columns that encode how old the drive is.
LIFETIME_PREFIXES = ("power_on_hours", "power_on_days", "drive_age_days")
LIFETIME_COLUMNS = {"uncorrectable_per_power_on_hour"}


def _is_secondary(column: str) -> bool:
    return column.startswith(SECONDARY_PREFIXES) or column in {
        "temperature_spike",
        "power_on_days",
        "uncorrectable_per_power_on_hour",
    }


def _is_new_feature(column: str) -> bool:
    """Every column added with the 2026-10-06 feature extension, so a model
    without them reproduces the earlier feature set on the same data build."""
    return (
        _is_secondary(column)
        or column.startswith("active_defect_")
        or column.endswith("_days_since_last_increase")
    )


def _excluded_indices(feature_columns: list[str], mode: str) -> list[int]:
    if mode == "identity":
        return [i for i, c in enumerate(feature_columns) if c in IDENTITY_COLUMNS]
    if mode == "non_windowed":
        return [
            i
            for i, c in enumerate(feature_columns)
            if not _WINDOWED.search(c) and not c.endswith(("_zscore", "_slope"))
        ]
    if mode == "new_features":
        return [i for i, c in enumerate(feature_columns) if _is_new_feature(c)]
    if mode == "secondary":
        return [i for i, c in enumerate(feature_columns) if _is_secondary(c)]
    if mode == "lifetime":
        return [
            i
            for i, c in enumerate(feature_columns)
            if c.startswith(LIFETIME_PREFIXES) or c in LIFETIME_COLUMNS
        ]
    if mode != "none":
        raise ValueError(f"unknown exclusion mode: {mode!r}")
    return []


def _predict_excluding(model: Any, x: np.ndarray, idx: list[int]) -> np.ndarray:
    """Scores `x` (possibly a read-only memory map) in chunks, with the
    excluded feature columns zeroed in each chunk's copy."""
    if not idx:
        return predict_proba_positive(model, x)
    parts = []
    for start in range(0, len(x), PREDICT_CHUNK_ROWS):
        chunk = np.array(x[start : start + PREDICT_CHUNK_ROWS])
        chunk[:, idx] = 0.0
        parts.append(predict_proba_positive(model, chunk))
    return np.concatenate(parts) if parts else np.empty((0,))


class _Zeroed:
    """Zeroes columns in place for the duration of a `with` block."""

    def __init__(self, arrays: list[np.ndarray], idx: list[int]) -> None:
        self.arrays, self.idx = arrays, idx
        self.saved: list[np.ndarray] = []

    def __enter__(self) -> None:
        if not self.idx:
            return
        self.saved = [a[:, self.idx].copy() for a in self.arrays]
        for a in self.arrays:
            a[:, self.idx] = 0.0

    def __exit__(self, *exc: object) -> None:
        if not self.idx:
            return
        for a, saved in zip(self.arrays, self.saved, strict=True):
            a[:, self.idx] = saved


def _sample_weights(y: np.ndarray, drive_ids: np.ndarray, weighting: str) -> np.ndarray | None:
    if weighting.startswith("spw"):
        power = float(weighting.split(":")[1]) if ":" in weighting else 0.5
        n_pos = max(int((y == 1).sum()), 1)
        weights = np.ones(len(y), dtype=np.float32)
        weights[y == 1] = ((y == 0).sum() / n_pos) ** power
        return weights
    if weighting == "drive":
        # Each failing drive's positive rows together weigh as much as ONE
        # positive row would under sqrt-ratio weighting, times the mean
        # rows-per-drive - so total positive weight stays comparable.
        df = pl.DataFrame({"d": drive_ids, "y": y})
        pos_rows = df.filter(pl.col("y") == 1).group_by("d").agg(pl.len().alias("n"))
        n_by_drive = dict(zip(pos_rows["d"].to_list(), pos_rows["n"].to_list(), strict=True))
        mean_rows = float(np.mean(list(n_by_drive.values()))) if n_by_drive else 1.0
        base = np.sqrt((y == 0).sum() / max(int((y == 1).sum()), 1))
        weights = np.ones(len(y), dtype=np.float32)
        pos_idx = np.flatnonzero(y == 1)
        weights[pos_idx] = [base * mean_rows / n_by_drive[d] for d in drive_ids[pos_idx].tolist()]
        return weights
    return None


def _max_abs_leaf(model: lgb.LGBMClassifier) -> float:
    leaves: list[float] = []

    def walk(node: dict) -> None:
        if "leaf_value" in node:
            leaves.append(abs(node["leaf_value"]))
        else:
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in model.booster_.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return max(leaves) if leaves else 0.0


def threshold_for_precision(labels: np.ndarray, scores: np.ndarray, target: float) -> float | None:
    """Highest-recall threshold whose precision is >= target, else None."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ok = np.flatnonzero(precision[:-1] >= target)
    if len(ok) == 0:
        return None
    return float(thresholds[ok[np.argmax(recall[:-1][ok])]])


def run_variant(name: str, data: dict[str, Any]) -> dict[str, Any]:
    extra, exclude, weighting = VARIANTS[name]
    base = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]["params"]
    params = {**base, **{k: v for k, v in extra.items() if k != "_engine"}, "verbosity": -1}
    if weighting != "is_unbalance":
        params.pop("is_unbalance", None)
    idx = _excluded_indices(data["feature_columns"], exclude)
    t0 = time.perf_counter()
    # Only the train matrix is in RAM and zeroed in place. Validation and test
    # are read-only memory maps: their excluded columns are zeroed per chunk
    # while scoring (`_predict_excluding`), and in the early-stopping copy.
    with _Zeroed([data["x_train"]], idx):
        weights = _sample_weights(data["y_train"], data["train_drives"], weighting)
        sub = data["es_idx"]
        x_eval = np.array(data["x_val"][sub])
        if idx:
            x_eval[:, idx] = 0.0
        if extra.get("_engine") == "xgboost":
            xgb_params = {k: v for k, v in extra.items() if k != "_engine"}
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",  # average precision, like the LightGBM variants
                early_stopping_rounds=50,
                n_jobs=-1,
                verbosity=0,
                **xgb_params,
            )
            model.fit(
                data["x_train"],
                data["y_train"],
                sample_weight=weights,
                eval_set=[(x_eval, data["y_val"][sub])],
                verbose=False,
            )
        else:
            # Monitor ONLY average precision: the default also tracks unweighted
            # binary_logloss, which class weighting worsens, and stopped
            # weighted/is_unbalance variants after a handful of trees (see
            # src/models/training.py::train_lightgbm).
            model = lgb.LGBMClassifier(**params, metric="average_precision")
            model.fit(
                data["x_train"],
                data["y_train"],
                sample_weight=weights,
                eval_X=x_eval,
                eval_y=data["y_val"][sub],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
            )
        del x_eval
        val_scores = _predict_excluding(model, data["x_val"], idx)
        test_scores = _predict_excluding(model, data["x_test"], idx)
    data.setdefault("_scores", {})[name] = (val_scores, test_scores)
    if isinstance(model, xgb.XGBClassifier):
        gains = model.feature_importances_  # normalized gain
    else:
        gains = model.booster_.feature_importance("gain")
    importances = sorted(zip(data["feature_columns"], gains, strict=True), key=lambda p: -p[1])
    total_gain = float(sum(g for _, g in importances)) or 1.0

    val_labels, val_drive_scores = drive_level_table(data["val_drives"], data["y_val"], val_scores)
    result: dict[str, Any] = {
        "variant": name,
        "excluded_features": len(idx),
        "trees": (
            int(model.best_iteration) + 1
            if isinstance(model, xgb.XGBClassifier)
            else int(model.best_iteration_ or model.n_estimators)
        ),
        "max_abs_leaf": (
            None if isinstance(model, xgb.XGBClassifier) else round(_max_abs_leaf(model), 2)
        ),
        "val_row_auprc": compute_auprc(data["y_val"], val_scores),
        "test_row_auprc": compute_auprc(data["y_test"], test_scores),
        "operating_points": {},
        "top_features": [(f, round(g / total_gain, 3)) for f, g in importances[:6]],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for target in (0.95, 0.5, 0.2):
        threshold = threshold_for_precision(val_labels, val_drive_scores, target)
        if threshold is None:
            result["operating_points"][str(target)] = None
            continue
        result["operating_points"][str(target)] = {
            "threshold": threshold,
            "val_drive": drive_level_metrics(
                data["val_drives"], data["y_val"], val_scores, threshold
            ),
            "test_drive": drive_level_metrics(
                data["test_drives"], data["y_test"], test_scores, threshold
            ),
        }
    result["val_drive_auprc"] = drive_level_metrics(
        data["val_drives"], data["y_val"], val_scores, 0.5
    )["auprc"]
    result["test_drive_auprc"] = drive_level_metrics(
        data["test_drives"], data["y_test"], test_scores, 0.5
    )["auprc"]
    return result


def _operating_points(
    data: dict[str, Any], val_scores: np.ndarray, test_scores: np.ndarray
) -> dict:
    """Thresholds chosen on validation, applied to test, at the drive level:
    best precision at recall >= 35% (the goal's recall floor) and the
    highest-recall point with precision >= 50%."""
    floor = tune_drive_level_threshold(
        data["val_drives"], data["y_val"], val_scores, target_precision=0.95
    )
    labels, drive_scores = drive_level_table(data["val_drives"], data["y_val"], val_scores)
    half = threshold_for_precision(labels, drive_scores, 0.5)
    out: dict[str, dict[str, Any] | None] = {}
    for name, threshold in (("recall>=35%", floor["threshold"]), ("precision>=50%", half)):
        if threshold is None:
            out[name] = None
            continue
        out[name] = {
            "threshold": threshold,
            "val": drive_level_metrics(data["val_drives"], data["y_val"], val_scores, threshold),
            "test": drive_level_metrics(
                data["test_drives"], data["y_test"], test_scores, threshold
            ),
        }
    return out


def deep_dive(variant: str, data: dict[str, Any]) -> dict[str, Any]:
    """(1) What are the false alarms? (2) Does requiring persistence cut them?"""
    if variant not in data.get("_scores", {}):
        run_variant(variant, data)
    val_scores, test_scores = data["_scores"][variant]
    report: dict[str, Any] = {"variant": variant, "persistence": {}, "false_alarms": {}}

    raw_points = _operating_points(data, val_scores, test_scores)
    raw_floor = raw_points["recall>=35%"]
    val_ids = data["val_ids"]
    for split, ids, y, scores in (
        ("val", val_ids, data["y_val"], val_scores),
        ("test", data["test_ids"], data["y_test"], test_scores),
    ):
        report["false_alarms"][split] = analyze_false_alarms(
            ids["drive_id"].to_numpy(),
            ids["date"].to_numpy(),
            y,
            scores,
            raw_floor["threshold"],
            ids["event_type"].to_numpy(),
            ids["days_to_event"].to_numpy().astype(float),
        )

    for smoother in SMOOTHERS:
        sv = smooth_scores(data["val_drives"], val_ids["date"].to_numpy(), val_scores, smoother)
        st = smooth_scores(
            data["test_drives"], data["test_ids"]["date"].to_numpy(), test_scores, smoother
        )
        report["persistence"][smoother] = _operating_points(data, sv, st)
    return report


def _print_deep_dive(report: dict[str, Any]) -> None:
    print(f"\n#### FALSE ALARMS ({report['variant']}, raw score, threshold for recall>=35%)")
    for split, fa in report["false_alarms"].items():
        print(
            f"  {split}: {fa['false_alarm_drives']} of {fa['healthy_drives']} 'healthy' drives "
            f"alerted (thr={fa['threshold']:.4f})"
        )
        for event_type, count in fa["alerted_by_event_type"].items():
            base = fa["base_rate_by_event_type"].get(event_type, 0)
            lift = fa["lift_by_event_type"].get(event_type)
            print(
                f"     {event_type:34s} alerted={count:6d}  of all healthy={base:7d}  "
                f"lift={'n/a' if lift is None else f'{lift:.2f}x'}"
            )
        if fa["failed_later_days_after_alert"]:
            print(
                f"     failed later, days after first alert: {fa['failed_later_days_after_alert']}"
            )
    print(
        f"\n#### PERSISTENCE ({report['variant']}): TEST drive metrics, thresholds from validation"
    )
    print(f"  {'smoother':9s} | {'@ recall>=35% (val)':36s} | @ precision>=50% (val)")
    for smoother, points in report["persistence"].items():
        cells = []
        for key in ("recall>=35%", "precision>=50%"):
            point = points[key]
            if point is None:
                cells.append("unreachable")
                continue
            t = point["test"]
            cells.append(
                f"P={t['precision']:.3f} R={t['recall']:.3f} "
                f"({t['caught_drive_count']}/{t['failing_drive_count']}, "
                f"FA={t['false_alarm_drive_count']})"
            )
        print(f"  {smoother:9s} | {cells[0]:36s} | {cells[1]}")


def drive_folds(drive_ids: np.ndarray, n_folds: int) -> np.ndarray:
    """A fold number per row, the same for every row of a drive (hash of the
    drive id), so out-of-fold scores never come from a model that saw that
    drive."""
    return (pl.Series("d", drive_ids).hash(seed=0) % n_folds).to_numpy().astype(np.int64)


def _fit_lgbm(
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    drives: np.ndarray,
    *,
    eval_x: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> lgb.LGBMClassifier:
    """LightGBM with sqrt positive weights; early-stopped on average precision
    when an evaluation set is given, otherwise trained for `n_estimators`."""
    weights = _sample_weights(y, drives, "spw")
    if eval_x is None or eval_y is None:
        return lgb.LGBMClassifier(**params).fit(x, y, sample_weight=weights)
    model = lgb.LGBMClassifier(**params, metric="average_precision")
    model.fit(
        x,
        y,
        sample_weight=weights,
        eval_X=eval_x,
        eval_y=eval_y,
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
    )
    return model


def two_stage_experiment(
    data: dict[str, Any], *, n_folds: int = 3, candidate_recall: float = 0.5
) -> dict[str, Any]:
    """Does a second model trained only on the hard cases beat the pooled model?

    Stage 1 is the pooled model. Its candidates are the rows scoring at or above
    the validation threshold that still catches `candidate_recall` of failing
    drives. Stage 2 is trained on the TRAIN rows that are candidates, using
    out-of-fold Stage 1 scores for them (each drive scored by a model that never
    saw it; in-sample scores would be far more confident than anything Stage 2
    meets later). The final score is Stage 2's for candidates and 0 otherwise.

    A cascade cannot raise precision by filtering alone; it helps only if
    Stage 2 separates the look-alike healthy drives better than Stage 1 did.
    The report compares both at the same recall levels, on the same test
    drives, with thresholds chosen on validation."""
    base = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]["params"]
    params = {**base, **REGULARIZED, "verbosity": -1}
    sub = data["es_idx"]
    y_train, train_drives = data["y_train"], data["train_drives"]

    stage1 = _fit_lgbm(
        params,
        data["x_train"],
        y_train,
        train_drives,
        eval_x=np.array(data["x_val"][sub]),
        eval_y=data["y_val"][sub],
    )
    n_trees = int(stage1.best_iteration_ or stage1.n_estimators)
    val_s1 = predict_proba_positive(stage1, data["x_val"])
    test_s1 = predict_proba_positive(stage1, data["x_test"])

    # Out-of-fold Stage 1 scores for the train rows, with the same tree count.
    folds = drive_folds(train_drives, n_folds)
    fold_params = {**params, "n_estimators": n_trees}
    train_s1 = np.zeros(len(y_train), dtype=np.float64)
    for fold in range(n_folds):
        held_out = folds == fold
        model = _fit_lgbm(
            fold_params, data["x_train"][~held_out], y_train[~held_out], train_drives[~held_out]
        )
        train_s1[held_out] = predict_proba_positive(model, data["x_train"][held_out])
        del model

    # Candidate threshold: the highest-precision validation threshold that
    # still catches `candidate_recall` of the failing drives.
    candidate_row = precision_at_recall_table(
        data["val_drives"],
        data["y_val"],
        val_s1,
        data["test_drives"],
        data["y_test"],
        test_s1,
        recalls=(candidate_recall,),
    )[0]
    report: dict[str, Any] = {
        "stage1_trees": n_trees,
        "candidate_recall": candidate_recall,
        "candidate_threshold": candidate_row["threshold"],
    }
    if candidate_row["threshold"] is None:
        report["status"] = "candidate recall not reachable on validation"
        return report
    threshold = candidate_row["threshold"]
    cand_train, cand_val, cand_test = (
        train_s1 >= threshold,
        val_s1 >= threshold,
        test_s1 >= threshold,
    )
    report["candidates"] = {
        "train_rows": int(cand_train.sum()),
        "train_positive_rows": int(y_train[cand_train].sum()),
        "validation_rows": int(cand_val.sum()),
        "test_rows": int(cand_test.sum()),
    }
    if report["candidates"]["train_positive_rows"] < 20 or data["y_val"][cand_val].sum() < 5:
        report["status"] = "too few candidate failures to train a second stage"
        return report

    def with_score(x: np.ndarray, mask: np.ndarray, score: np.ndarray) -> np.ndarray:
        """Candidate rows' features plus the Stage 1 score as a last column."""
        return np.column_stack([np.asarray(x[np.flatnonzero(mask)]), score[mask]])

    stage2 = _fit_lgbm(
        {**params, "min_child_samples": 50},
        with_score(data["x_train"], cand_train, train_s1),
        y_train[cand_train],
        train_drives[cand_train],
        eval_x=with_score(data["x_val"], cand_val, val_s1),
        eval_y=data["y_val"][cand_val],
    )
    val_final = np.zeros(len(val_s1))
    test_final = np.zeros(len(test_s1))
    val_final[cand_val] = predict_proba_positive(
        stage2, with_score(data["x_val"], cand_val, val_s1)
    )
    test_final[cand_test] = predict_proba_positive(
        stage2, with_score(data["x_test"], cand_test, test_s1)
    )
    report["stage2_trees"] = int(stage2.best_iteration_ or stage2.n_estimators)
    recalls = tuple(r for r in (0.05, 0.10, 0.20, 0.35) if r <= candidate_recall)
    for name, val_scores, test_scores in (
        ("stage1", val_s1, test_s1),
        ("two_stage", val_final, test_final),
    ):
        report[name] = precision_at_recall_table(
            data["val_drives"],
            data["y_val"],
            val_scores,
            data["test_drives"],
            data["y_test"],
            test_scores,
            recalls=recalls,
        )
    report["status"] = "ok"
    return report


def _print_two_stage(report: dict[str, Any]) -> None:
    print("\n#### TWO STAGE (second model on Stage 1's hard cases) - TEST, drive level")
    print(f"  stage 1 trees={report['stage1_trees']}  candidates={report.get('candidates')}")
    if report["status"] != "ok":
        print(f"  not run: {report['status']}")
        return
    print(f"  stage 2 trees={report['stage2_trees']}")
    for stage1_row, two_row in zip(report["stage1"], report["two_stage"], strict=True):
        cells = []
        for row in (stage1_row, two_row):
            if row.get("threshold") is None:
                cells.append("unreachable")
                continue
            t = row["test"]
            cells.append(
                f"P={t['precision']:.3f} R={t['recall']:.3f} "
                f"({t['caught_drive_count']}/{t['failing_drive_count']}, "
                f"FA={t['false_alarm_drive_count']})"
            )
        print(
            f"  recall>={stage1_row['target_recall']:.0%}: stage 1 alone {cells[0]}"
            f"  |  two stage {cells[1]}"
        )


def per_model_experiment(data: dict[str, Any], *, top_n: int) -> list[dict[str, Any]]:
    """Trains one LightGBM per drive family (the `top_n` families with the most
    failing validation drives) and compares it with the POOLED model on exactly
    the same test drives. Each family's threshold is chosen on that family's
    own validation drives (drive-level precision >= 50%, then the highest
    recall), so the comparison is what a per-family deployment would get.
    Families with too few failures are reported but not scored: with a
    handful of failing drives, any precision figure is noise."""
    pooled_val, pooled_test = data["_scores"]["reg_spw"]  # (validation, test) scores
    val_labels = data["y_val"]
    failing_by_model = (
        pl.DataFrame({"m": data["val_models"], "d": data["val_drives"], "y": val_labels})
        .filter(pl.col("y") == 1)
        .group_by("m")
        .agg(pl.col("d").n_unique().alias("failing"))
        .sort("failing", descending=True)
    )
    families = failing_by_model["m"].to_list()[:top_n]
    base = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]["params"]
    params = {**base, **{k: v for k, v in REGULARIZED.items()}, "verbosity": -1}
    results = []
    for family in families:
        tr = data["train_models"] == family
        va = data["val_models"] == family
        te = data["test_models"] == family
        n_fail_val = int(failing_by_model.filter(pl.col("m") == family)["failing"][0])
        # Drives, not rows: a failing drive has many positive rows.
        n_fail_test = int(np.unique(data["test_drives"][te & (data["y_test"] == 1)]).size)
        row: dict[str, Any] = {
            "family": family,
            "train_rows": int(tr.sum()),
            "failing_val_drives": n_fail_val,
            "failing_test_drives": n_fail_test,
        }
        if int((data["y_train"][tr] == 1).sum()) < 20 or n_fail_val < 10:
            row["scored"] = False
            results.append(row)
            continue
        row["scored"] = True
        weights = _sample_weights(data["y_train"][tr], data["train_drives"][tr], "spw")
        model = lgb.LGBMClassifier(**params, metric="average_precision")
        xv = data["x_val"][va]
        yv = data["y_val"][va]
        model.fit(
            data["x_train"][tr],
            data["y_train"][tr],
            sample_weight=weights,
            eval_X=xv,
            eval_y=yv,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
        )
        fam_val_scores = predict_proba_positive(model, xv)
        fam_test_scores = predict_proba_positive(model, data["x_test"][te])
        labels, drive_scores = drive_level_table(data["val_drives"][va], yv, fam_val_scores)
        threshold = threshold_for_precision(labels, drive_scores, 0.5)
        row["trees"] = int(model.best_iteration_ or model.n_estimators)
        row["val_drive_auprc"] = compute_auprc(labels, drive_scores) if labels.sum() else None
        if threshold is None:
            row["threshold"] = None
            results.append(row)
            continue
        row["threshold"] = threshold
        own = drive_level_metrics(
            data["test_drives"][te], data["y_test"][te], fam_test_scores, threshold
        )
        # Same test drives, pooled model at the pooled threshold chosen on the
        # pooled validation set at the same 50% precision target.
        pooled_thr = threshold_for_precision(
            *drive_level_table(data["val_drives"], data["y_val"], pooled_val), 0.5
        )
        if pooled_thr is None:
            row["pooled_on_same_drives"] = None
            results.append(row)
            continue
        pooled = drive_level_metrics(
            data["test_drives"][te], data["y_test"][te], pooled_test[te], pooled_thr
        )
        row["own_test"] = {
            k: own[k]
            for k in (
                "precision",
                "recall",
                "caught_drive_count",
                "failing_drive_count",
                "false_alarm_drive_count",
                "auprc",
            )
        }
        row["pooled_on_same_drives"] = {
            k: pooled[k]
            for k in (
                "precision",
                "recall",
                "caught_drive_count",
                "failing_drive_count",
                "false_alarm_drive_count",
                "auprc",
            )
        }
        results.append(row)
    return results


def _print_per_model(results: list[dict[str, Any]]) -> None:
    print("\n#### PER DRIVE FAMILY (test drives; own model vs pooled model on the same drives)")
    for r in results:
        head = (
            f"  {r['family']}: train rows={r['train_rows']} failing val={r['failing_val_drives']} "
            f"test={r['failing_test_drives']}"
        )
        if not r["scored"]:
            print(head + "  -> too few failures to score")
            continue
        if r.get("threshold") is None:
            print(head + f"  trees={r['trees']}  -> 50% drive precision unreachable on validation")
            continue
        own, pooled = r["own_test"], r["pooled_on_same_drives"]
        if pooled is None:
            pooled = {k: float("nan") for k in own}
        print(
            head + f"  trees={r['trees']}\n"
            f"      own model:    P={own['precision']:.3f} R={own['recall']:.3f} "
            f"caught={own['caught_drive_count']}/{own['failing_drive_count']} "
            f"false_alarms={own['false_alarm_drive_count']} AUPRC={own['auprc']:.3f}\n"
            f"      pooled model: P={pooled['precision']:.3f} R={pooled['recall']:.3f} "
            f"caught={pooled['caught_drive_count']}/{pooled['failing_drive_count']} "
            f"false_alarms={pooled['false_alarm_drive_count']} AUPRC={pooled['auprc']:.3f}"
        )


def _print(result: dict[str, Any]) -> None:
    print(
        f"\n== {result['variant']}: trees={result['trees']} excluded={result['excluded_features']} "
        f"max|leaf|={result['max_abs_leaf']} ({result['seconds']}s)"
    )
    print(
        f"   row AUPRC val={result['val_row_auprc']:.4f} test={result['test_row_auprc']:.4f} | "
        f"drive AUPRC val={result['val_drive_auprc']:.4f} test={result['test_drive_auprc']:.4f}"
    )
    for target, point in result["operating_points"].items():
        if point is None:
            print(f"   precision>={target}: unreachable on validation (drive level)")
            continue
        t = point["test_drive"]
        print(
            f"   precision>={target} (val-chosen thr={point['threshold']:.4f}): TEST drive "
            f"precision={t['precision']:.3f} recall={t['recall']:.3f} "
            f"caught={t['caught_drive_count']}/{t['failing_drive_count']} "
            f"false_alarm_drives={t['false_alarm_drive_count']}"
        )
    print("   top gain:", ", ".join(f"{f} {g:.1%}" for f, g in result["top_features"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Train a second model on the rows the pooled model flags (out-of-fold "
        "scores on train) and compare it with the pooled model alone at fixed recall.",
    )
    parser.add_argument(
        "--per-model",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Also train one model per drive family for the N families with the most "
        "failing validation drives (default 3), compared with the pooled model on the "
        "same test drives. Runs after the variants; needs reg_spw to have been run.",
    )
    parser.add_argument(
        "--deep-dive",
        nargs="?",
        const="reg_spw",
        choices=list(VARIANTS),
        help="After the variants, analyze this variant's false alarms and test persistence "
        "rules (rolling mean/min/median over each drive's last n days). Default reg_spw.",
    )
    args = parser.parse_args()

    configure_logging()
    apply_memory_limit_from_config()
    work_dir = _find_work_dir(args.work_dir)
    logger.info("experiment_work_dir", path=str(work_dir))
    feature_columns = json.loads((work_dir / "frame_meta.json").read_text())["feature_columns"]

    val_ids = pl.read_parquet(work_dir / "validation_ids.parquet")
    test_ids = pl.read_parquet(work_dir / "test_ids.parquet")
    train_ids = pl.read_parquet(work_dir / "train_ids.parquet")
    data: dict[str, Any] = {
        "feature_columns": feature_columns,
        # Train in RAM (LightGBM needs it there, and exclusions zero it in place);
        # validation and test as read-only memory maps, scored in chunks.
        "x_train": np.load(work_dir / "x_train.npy"),
        "y_train": np.load(work_dir / "y_train.npy"),
        "train_drives": train_ids["drive_id"].to_numpy(),
        "train_models": train_ids["drive_model"].to_numpy(),
        "x_val": np.load(work_dir / "x_val.npy", mmap_mode="r"),
        "y_val": np.load(work_dir / "y_val.npy"),
        "val_ids": val_ids,
        "val_drives": val_ids["drive_id"].to_numpy(),
        "val_models": val_ids["drive_model"].to_numpy(),
        "x_test": np.load(work_dir / "x_test.npy", mmap_mode="r"),
        "y_test": np.load(work_dir / "y_test.npy"),
        "test_ids": test_ids,
        "test_drives": test_ids["drive_id"].to_numpy(),
        "test_models": test_ids["drive_model"].to_numpy(),
    }
    data["es_idx"] = early_stopping_subset(data["y_val"])

    results: list[dict[str, Any]] = []
    if not args.skip_baseline:
        t0 = time.perf_counter()
        rng = np.random.default_rng(0)
        pick = rng.choice(
            len(data["y_train"]), size=min(500_000, len(data["y_train"])), replace=False
        )
        baseline = train_logistic_regression_baseline(data["x_train"][pick], data["y_train"][pick])
        base_val = predict_proba_positive(baseline, data["x_val"])
        base_test = predict_proba_positive(baseline, data["x_test"])
        val_drive_auprc = drive_level_metrics(data["val_drives"], data["y_val"], base_val, 0.5)
        test_drive_auprc = drive_level_metrics(data["test_drives"], data["y_test"], base_test, 0.5)
        print(
            f"\n== logistic baseline (reference, {time.perf_counter() - t0:.0f}s): "
            f"row AUPRC val={compute_auprc(data['y_val'], base_val):.4f} "
            f"test={compute_auprc(data['y_test'], base_test):.4f} | drive AUPRC "
            f"val={val_drive_auprc['auprc']:.4f} test={test_drive_auprc['auprc']:.4f}"
        )
    for name in args.variants:
        result = run_variant(name, data)
        results.append(result)
        _print(result)
        (work_dir / "experiment_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
    if args.per_model:
        if "reg_spw" not in data.get("_scores", {}):
            run_variant("reg_spw", data)
        per_model = per_model_experiment(data, top_n=args.per_model)
        _print_per_model(per_model)
        (work_dir / "experiment_per_model.json").write_text(
            json.dumps(per_model, indent=2, default=str)
        )
    if args.two_stage:
        two_stage_report = two_stage_experiment(data)
        _print_two_stage(two_stage_report)
        (work_dir / "experiment_two_stage.json").write_text(
            json.dumps(two_stage_report, indent=2, default=str)
        )
    if args.deep_dive:
        report = deep_dive(args.deep_dive, data)
        _print_deep_dive(report)
        (work_dir / "experiment_deep_dive.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
    print(f"\nResults saved to {work_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
