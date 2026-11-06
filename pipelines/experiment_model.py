"""Dev tool (`make experiment-model`): compares LightGBM variants on the
arrays a previous `make train TRAIN_ARGS=--keep-work-dir` left behind, so
a modelling idea is judged in minutes instead of re-running the whole
join. Not part of the production pipeline; it never writes models, reports
or model cards. The `--two-stage` comparison logs one MLflow run (tag
`run_kind=experiment`, no model), which `latest_run_id_for_horizon` skips.

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
import subprocess
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

from src.logging_config import configure_logging, get_logger
from src.models.access_log import _git_sha, record_test_access
from src.models.alerting import SMOOTHERS, analyze_false_alarms, smooth_scores
from src.models.evaluation import (
    compute_auprc,
    drive_level_metrics,
    drive_level_table,
    precision_at_recall_table,  # noqa: I001
)
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.survival import AftModel, survival_bounds, train_aft
from src.models.threshold import tune_drive_level_threshold
from src.models.training import (
    PREDICT_CHUNK_ROWS,
    early_stopping_subset,
    predict_proba_positive,
)
from src.models.two_stage import (  # noqa: F401
    drive_folds,
    fit_second_stage,
    lightgbm_fitter,
    out_of_fold_scores,
    threshold_at_drive_recall,
    tree_count,
)
from src.models.uncertainty import paired_flag_difference_ci
from src.outputs import refuse_overwrite
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

#: Survival variants (src/models/survival.py): trained on time to failure
#: instead of the binary label. "_engine" routes run_variant to the AFT trainer.
AFT_ENGINE = {"_engine": "xgboost_aft"}

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
    # Survival: unweighted, and with the sqrt weight on rows whose failure
    # is observed. Scored on the same binary label as every other variant.
    "xgboost_aft": (AFT_ENGINE, "none", "none"),
    "xgboost_aft_spw": (AFT_ENGINE, "none", "spw"),
    "strong_reg": (
        {**REGULARIZED, "min_child_samples": 500, "reg_lambda": 50.0},
        "none",
        "spw",
    ),
}


def _tracked_diff() -> str:
    """`git diff HEAD` of tracked code, configs and docs, not data/ ("" when
    clean or git is unavailable)."""
    try:
        return subprocess.run(
            ["git", "diff", "HEAD", "--", ".", ":(exclude)data"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


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


def _train_survival_bounds(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Time-to-failure bounds for the train rows, using outcomes only up to
    the last train date plus the label horizon (what the binary label uses)."""
    train_ids = data["train_ids"]
    missing = {"event_type", "days_to_event"} - set(train_ids.columns)
    if missing:
        raise SystemExit(
            f"train_ids.parquet has no {sorted(missing)}: this work dir predates the survival "
            "variants. Re-run `make train TRAIN_ARGS=--keep-work-dir`."
        )
    dates = train_ids["date"].to_numpy().astype("datetime64[D]")
    return survival_bounds(
        dates,
        train_ids["event_type"].to_numpy(),
        train_ids["days_to_event"].to_numpy().astype(np.float64),
        cutoff_date=dates.max(),
        horizon_days=data["horizon_days"],
    )


def _lightgbm_params(base: dict[str, Any], extra: dict[str, Any], weighting: str) -> dict[str, Any]:
    """LightGBM params for one variant. weighting "is_unbalance" must turn LightGBM's own
    is_unbalance ON: configs/model.yaml no longer carries it (commit 536458a moved the shipped
    model to positive_weight_power), so merely removing the key for other weightings left the
    is_unbalance variants unweighted."""
    params = {**base, **{k: v for k, v in extra.items() if k != "_engine"}, "verbosity": -1}
    if weighting == "is_unbalance":
        params["is_unbalance"] = True
    else:
        params.pop("is_unbalance", None)
    return params


def run_variant(name: str, data: dict[str, Any]) -> dict[str, Any]:
    record_test_access(variant=name, purpose="experiment_model run_variant (test scoring)")
    extra, exclude, weighting = VARIANTS[name]
    base = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]["params"]
    params = _lightgbm_params(base, extra, weighting)
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
        if extra.get("_engine") == "xgboost_aft":
            lower, upper = _train_survival_bounds(data)
            observed = np.isfinite(upper).astype(np.int8)
            model = train_aft(
                data["x_train"],
                lower,
                upper,
                eval_x=x_eval,
                eval_y=data["y_val"][sub],
                horizon_days=data["horizon_days"],
                sample_weight=_sample_weights(observed, data["train_drives"], weighting),
            )
        elif extra.get("_engine") == "xgboost":
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
    if isinstance(model, AftModel):
        gains = model.gain_by_feature(len(data["feature_columns"]))
    elif isinstance(model, xgb.XGBClassifier):
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
            model.tree_count
            if isinstance(model, AftModel)
            else int(model.best_iteration) + 1
            if isinstance(model, xgb.XGBClassifier)
            else int(model.best_iteration_ or model.n_estimators)
        ),
        "max_abs_leaf": (
            round(_max_abs_leaf(model), 2) if isinstance(model, lgb.LGBMClassifier) else None
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


#: Recall levels of the two-stage comparison when neither the command line nor
#: `model.two_stage.experiment` names them.
TWO_STAGE_RECALL_LEVELS = (0.05, 0.10, 0.20, 0.35)

#: (drive labels, drive scores, drive scores counting only rows at least
#: `lead_min_days` before the failure, or None), one entry per drive, ordered
#: by drive_id.
DriveView = tuple[np.ndarray, np.ndarray, np.ndarray | None]


def drive_view(
    drive_ids: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    days_to_event: np.ndarray | None = None,
    lead_min_days: int = 1,
) -> DriveView:
    """The drive table `drive_level_table` builds (same labels, scores and
    order), plus a lead-aware score: for a failing drive the highest score
    over its positive rows at least `lead_min_days` before the failure
    (-inf when it has none), for a healthy drive its ordinary score. A drive
    flagged only on the day it fails is caught by the first and missed by the
    second. Built once per score vector, so many thresholds cost no further
    pass over the rows."""
    df = pl.DataFrame({"drive_id": drive_ids, "y": y, "score": scores})
    positive = pl.col("y") == 1
    aggs = [
        positive.any().alias("failing"),
        pl.col("score").filter(positive).max().alias("positive_max"),
        pl.col("score").max().alias("all_max"),
    ]
    if days_to_event is not None:
        df = df.with_columns(early=pl.Series(days_to_event).fill_null(-1) >= lead_min_days)
        aggs.append(pl.col("score").filter(positive & pl.col("early")).max().alias("lead_max"))
    table = df.group_by("drive_id").agg(aggs).sort("drive_id")
    failing = table["failing"].to_numpy()
    all_max = table["all_max"].to_numpy().astype(float)
    drive_scores = np.where(failing, table["positive_max"].fill_null(-np.inf).to_numpy(), all_max)
    lead = None
    if days_to_event is not None:
        lead = np.where(failing, table["lead_max"].fill_null(-np.inf).to_numpy(), all_max)
    return failing.astype(int), drive_scores.astype(float), lead


def view_metrics(view: DriveView, threshold: float, cost_ratio: float) -> dict[str, Any]:
    """Drive-level counts at `threshold` (as `drive_level_metrics`), plus
    recall with lead and an illustrative cost: missed failures x
    `cost_ratio` + false alarms."""
    labels, scores, lead = view
    failing = labels == 1
    flagged = scores >= threshold
    caught = int((flagged & failing).sum())
    false_alarms = int((flagged & ~failing).sum())
    failing_count, healthy_count = int(failing.sum()), int((~failing).sum())
    out: dict[str, Any] = {
        "failing_drive_count": failing_count,
        "healthy_drive_count": healthy_count,
        "caught_drive_count": caught,
        "false_alarm_drive_count": false_alarms,
        "precision": caught / (caught + false_alarms) if caught + false_alarms else 0.0,
        "recall": caught / failing_count if failing_count else 0.0,
        "false_alarm_rate": false_alarms / healthy_count if healthy_count else 0.0,
        "cost": (failing_count - caught) * cost_ratio + false_alarms,
    }
    if lead is not None:
        caught_lead = int(((lead >= threshold) & failing).sum())
        out["caught_with_lead_count"] = caught_lead
        out["recall_with_lead"] = caught_lead / failing_count if failing_count else 0.0
    return out


def level_row(
    target_recall: float, val_view: DriveView, test_view: DriveView, cost_ratio: float
) -> dict[str, Any]:
    """One recall level: the threshold chosen on validation (the most precise
    one catching at least `target_recall` of failing drives, as
    `precision_at_recall_table`), applied to test."""
    threshold = threshold_at_drive_recall(val_view[0], val_view[1], target_recall)
    if threshold is None:
        return {"target_recall": target_recall, "threshold": None}
    return {
        "target_recall": target_recall,
        "threshold": threshold,
        "validation": view_metrics(val_view, threshold, cost_ratio),
        "test": view_metrics(test_view, threshold, cost_ratio),
    }


def two_stage_experiment(
    data: dict[str, Any],
    *,
    candidate_recalls: tuple[float, ...] = (0.5,),
    recall_levels: tuple[float, ...] = TWO_STAGE_RECALL_LEVELS,
    n_folds: int = 3,
    seed: int = 0,
    lead_min_days: int = 1,
    cost_ratio: float = 20.0,
    n_boot: int = 1000,
    stage2_params: dict[str, Any] | None = None,
    early_stopping_max_negatives: int | None = None,
) -> list[dict[str, Any]]:
    """Does a second model trained only on the hard cases beat the pooled model?

    Uses the same code `make train TRAIN_ARGS=--two-stage` runs
    (src/models/two_stage.py): Stage 1 is the pooled model; Stage 2 is trained
    on the train rows Stage 1 scores above its candidate threshold, with
    out-of-fold Stage 1 scores. One report per net in `candidate_recalls`
    (the validation recall the candidate threshold keeps), each comparing
    both models at every level of `recall_levels` up to the net, on the same
    test drives, with thresholds chosen on validation. Stage 1, its scores and
    its out-of-fold train scores are computed once and shared by every net.
    `seed` sets the fold assignment, both models' sampling and the bootstrap
    resamples, so repeated seeds show how much of a difference is chance.

    A level equal to the net is a no-op by construction (its threshold is the
    candidate threshold, and the cascade flags what Stage 1 flags); it is
    kept as a check."""
    record_test_access(
        variant="two_stage_experiment",
        purpose="experiment_model two-stage test scoring",
        extra={"seed": seed, "candidate_recalls": list(candidate_recalls)},
    )
    model_cfg = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]
    fit = lightgbm_fitter(
        {**model_cfg["params"], **REGULARIZED},
        positive_weight_power=0.5,
        early_stopping_rounds=50,
        seed=seed,
    )
    t0 = time.perf_counter()
    sub = data["es_idx"]
    stage1 = fit(
        data["x_train"],
        data["y_train"],
        eval_x=np.array(data["x_val"][sub]),
        eval_y=data["y_val"][sub],
    )
    val_s1 = predict_proba_positive(stage1, data["x_val"])
    test_s1 = predict_proba_positive(stage1, data["x_test"])
    train_oof = out_of_fold_scores(
        fit,
        data["x_train"],
        data["y_train"],
        data["train_drives"],
        n_estimators=tree_count(stage1),
        n_folds=n_folds,
        seed=seed,
    )
    logger.info("two_stage_stage1_done", seed=seed, seconds=round(time.perf_counter() - t0))

    def view(split: str, scores: np.ndarray) -> DriveView:
        return drive_view(
            data[f"{split}_drives"],
            data[f"y_{split}"],
            scores,
            data.get(f"{split}_days_to_event"),
            lead_min_days,
        )

    s1_val, s1_test = view("val", val_s1), view("test", test_s1)
    stage1_rows = {r: level_row(r, s1_val, s1_test, cost_ratio) for r in recall_levels}

    reports: list[dict[str, Any]] = []
    for net in candidate_recalls:
        t1 = time.perf_counter()
        levels = [r for r in recall_levels if r <= net + 1e-9]
        model, report = fit_second_stage(
            fit,
            stage1,
            x_train=data["x_train"],
            y_train=data["y_train"],
            train_drives=data["train_drives"],
            x_val=data["x_val"],
            y_val=data["y_val"],
            val_drives=data["val_drives"],
            val_stage1_scores=val_s1,
            candidate_recall=net,
            n_folds=n_folds,
            seed=seed,
            train_stage1_scores=train_oof,
            stage2_params=stage2_params,
            early_stopping_max_negatives=early_stopping_max_negatives,
        )
        report["stage1"] = [stage1_rows[r] for r in levels]
        if model is None:
            reports.append(report)
            continue
        threshold = model.candidate_threshold
        test_candidates = test_s1 >= threshold
        report["candidates"]["test_rows"] = int(test_candidates.sum())
        report["candidates"]["test_positive_rows"] = int(data["y_test"][test_candidates].sum())
        # The leaky bucket: no threshold on the combined score can catch a
        # failing drive Stage 1 did not pass, so Stage 1's recall at the
        # candidate threshold is the most the cascade can reach. The share of
        # drives passing Stage 1 is what a drift monitor would watch.
        at_net = view_metrics(s1_test, threshold, cost_ratio)
        report["ceiling"] = {
            "test_recall": at_net["recall"],
            "test_recall_with_lead": at_net.get("recall_with_lead"),
            "validation_drive_share": float((s1_val[1] >= threshold).mean()),
            "test_drive_share": float((s1_test[1] >= threshold).mean()),
        }
        c_val = view("val", model.final_scores(data["x_val"], val_s1))
        c_test = view("test", model.final_scores(data["x_test"], test_s1))
        report["two_stage"] = []
        for r in levels:
            row = level_row(r, c_val, c_test, cost_ratio)
            first = stage1_rows[r]
            if row["threshold"] is not None and first["threshold"] is not None:
                # Same seed, same drive count: every net and level of this seed
                # sees the same resampled drives.
                row["paired_test"] = paired_flag_difference_ci(
                    s1_test[0],
                    s1_test[1] >= first["threshold"],
                    c_test[1] >= row["threshold"],
                    n_boot=n_boot,
                    seed=seed,
                )
            report["two_stage"].append(row)
        report["seconds"] = round(time.perf_counter() - t1)
        logger.info("two_stage_net_done", seed=seed, net=net, seconds=report["seconds"])
        reports.append(report)
    return reports


def _print_two_stage(report: dict[str, Any]) -> None:
    print(
        f"\n#### TWO STAGE (net {report['candidate_recall']:.0%}) - TEST, drive level"
        f" - seed {report['seed']}"
    )
    print(f"  stage 1 trees={report['stage1_trees']}  candidates={report.get('candidates')}")
    if report["status"] != "ok":
        print(f"  not run: {report['status']}")
        return
    print(f"  stage 2 trees={report['stage2_trees']}  ceiling={report.get('ceiling')}")
    for stage1_row, two_row in zip(report["stage1"], report["two_stage"], strict=True):
        paired = two_row.get("paired_test")
        gap = (
            f"  gap {paired['difference']:+.3f} [{paired['difference_ci'][0]:+.3f}, "
            f"{paired['difference_ci'][1]:+.3f}]"
            if paired
            else ""
        )
        print(
            f"  recall>={stage1_row['target_recall']:.0%}: stage 1 alone "
            f"{_recall_cell(stage1_row)}  |  two stage {_recall_cell(two_row)}{gap}"
        )


def summarize_two_stage_seeds(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per recall level, across the seeds that trained a second stage (all
    with the same net): mean test precision and recall of each model, and in
    how many seeds the two-stage model was at least as good on both (and
    better on one). With paired bootstraps, also the mean precision gap and in
    how many seeds its interval lies above (below) 0; with lead-aware recall
    and cost, their means."""
    usable = [r for r in reports if r.get("status") == "ok"]
    summary: list[dict[str, Any]] = []
    if not usable:
        return summary
    for i, first in enumerate(usable[0]["stage1"]):
        rows = [
            (r["stage1"][i], r["two_stage"][i])
            for r in usable
            if r["stage1"][i].get("threshold") is not None
            and r["two_stage"][i].get("threshold") is not None
        ]
        if not rows:
            continue
        pairs = [(one["test"], two["test"]) for one, two in rows]

        def mean(side: int, key: str, pairs: list = pairs) -> float | None:
            values = [pair[side][key] for pair in pairs if key in pair[side]]
            return float(np.mean(values)) if values else None

        entry: dict[str, Any] = {
            "target_recall": first["target_recall"],
            "seeds": len(pairs),
            "stage1_precision": mean(0, "precision"),
            "stage1_recall": mean(0, "recall"),
            "two_stage_precision": mean(1, "precision"),
            "two_stage_recall": mean(1, "recall"),
            "precision_gain_min": float(
                min(two["precision"] - one["precision"] for one, two in pairs)
            ),
            "precision_gain_max": float(
                max(two["precision"] - one["precision"] for one, two in pairs)
            ),
            "two_stage_better_seeds": sum(
                two["precision"] >= one["precision"]
                and two["recall"] >= one["recall"]
                and (two["precision"] > one["precision"] or two["recall"] > one["recall"])
                for one, two in pairs
            ),
        }
        for key in ("recall_with_lead", "cost", "false_alarm_drive_count", "caught_drive_count"):
            if mean(0, key) is not None:
                entry[f"stage1_{key}"] = mean(0, key)
                entry[f"two_stage_{key}"] = mean(1, key)
        paired = [two["paired_test"] for _, two in rows if "paired_test" in two]
        if paired:
            entry["bootstrap_gap_mean"] = float(np.mean([p["difference"] for p in paired]))
            entry["ci_above_zero_seeds"] = sum(p["difference_ci"][0] > 0 for p in paired)
            entry["ci_below_zero_seeds"] = sum(p["difference_ci"][1] < 0 for p in paired)
        summary.append(entry)
    return summary


def summarize_two_stage_nets(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`summarize_two_stage_seeds` per net, plus each net's mean candidate
    pool and ceiling over the seeds that trained it."""
    nets = sorted({r["candidate_recall"] for r in reports})
    out = []
    for net in nets:
        group = [r for r in reports if r["candidate_recall"] == net]
        usable = [r for r in group if r.get("status") == "ok"]

        def mean(path: tuple[str, str], usable: list = usable) -> float | None:
            values = [r[path[0]][path[1]] for r in usable if r[path[0]].get(path[1]) is not None]
            return float(np.mean(values)) if values else None

        out.append(
            {
                "candidate_recall": net,
                "seeds_run": len(group),
                "seeds_trained": len(usable),
                "statuses": sorted({r["status"] for r in group}),
                "train_rows": mean(("candidates", "train_rows")),
                "train_positive_share": mean(("candidates", "train_positive_share")),
                "validation_rows": mean(("candidates", "validation_rows")),
                "test_rows": mean(("candidates", "test_rows")),
                "test_ceiling": mean(("ceiling", "test_recall")),
                "test_ceiling_with_lead": mean(("ceiling", "test_recall_with_lead")),
                "test_drive_share": mean(("ceiling", "test_drive_share")),
                "levels": summarize_two_stage_seeds(group),
            }
        )
    return out


def _print_two_stage_summary(summary: list[dict[str, Any]]) -> None:
    print("\n#### TWO STAGE ACROSS SEEDS - TEST, drive level (means)")
    for row in summary:
        gap = (
            f"; bootstrap gap {row['bootstrap_gap_mean']:+.3f}, interval above 0 in "
            f"{row['ci_above_zero_seeds']}/{row['seeds']}"
            if "bootstrap_gap_mean" in row
            else ""
        )
        print(
            f"  recall>={row['target_recall']:.0%} ({row['seeds']} seeds): stage 1 alone "
            f"P={row['stage1_precision']:.3f} R={row['stage1_recall']:.3f}  |  two stage "
            f"P={row['two_stage_precision']:.3f} R={row['two_stage_recall']:.3f}  |  "
            f"precision gain {row['precision_gain_min']:+.3f} to "
            f"{row['precision_gain_max']:+.3f}; two stage better on precision and recall "
            f"in {row['two_stage_better_seeds']}/{row['seeds']}{gap}"
        )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def two_stage_tables(nets: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    """Markdown: one candidate-pool table, then one comparison table per net."""
    lines = [
        "# Two-stage cascade, wide Stage 1 nets (test, drive level)",
        "",
        f"Work dir `{settings['work_dir']}`, horizon {settings['horizon_days']} days, "
        f"git `{settings['git_commit']}`. Seeds {settings['seeds']}; thresholds chosen on "
        f"validation. Recall with lead counts a failing drive only when a row at least "
        f"{settings['lead_min_days']} day(s) before its failure clears the threshold. Cost = "
        f"missed failures x {settings['cost_ratio']:g} + false alarms (an illustration, not a "
        "measured cost). Gap = cascade minus Stage 1 test precision, paired drive bootstrap "
        f"({settings['n_boot']} resamples, 95%).",
        "",
        "## Candidate pools (means over seeds)",
        "",
        "| Net | Seeds trained | Train candidate rows | Positive share | Validation candidate "
        "rows | Test candidate rows | Ceiling (test recall at T1) | Ceiling with lead | "
        "Test drives passing Stage 1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for net in nets:
        rows = [net[k] for k in ("train_rows", "validation_rows", "test_rows")]
        counts = ["-" if v is None else f"{v:,.0f}" for v in rows]
        lines.append(
            f"| {net['candidate_recall']:.0%} | {net['seeds_trained']}/{net['seeds_run']} | "
            f"{counts[0]} | {_pct(net['train_positive_share'])} | {counts[1]} | {counts[2]} | "
            f"{_pct(net['test_ceiling'])} | {_pct(net['test_ceiling_with_lead'])} | "
            f"{_pct(net['test_drive_share'])} |"
        )
    for net in nets:
        lines += [
            "",
            f"## Net {net['candidate_recall']:.0%}",
            "",
            "| Level | Stage 1 P / R | Cascade P / R | Gain range | Better (P and R) | Gap mean "
            "| Interval above 0 | Recall with lead S1 / cascade | False alarms S1 / cascade | "
            "Cost S1 / cascade |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        if not net["levels"]:
            lines.append(f"| - | not trained: {', '.join(net['statuses'])} |||||||||")
        for row in net["levels"]:
            seeds = row["seeds"]
            gap = row.get("bootstrap_gap_mean")
            lines.append(
                f"| {row['target_recall']:.0%} | {_pct(row['stage1_precision'])} / "
                f"{_pct(row['stage1_recall'])} | {_pct(row['two_stage_precision'])} / "
                f"{_pct(row['two_stage_recall'])} | {row['precision_gain_min']:+.1%} to "
                f"{row['precision_gain_max']:+.1%} | {row['two_stage_better_seeds']}/{seeds} | "
                f"{'-' if gap is None else f'{gap:+.1%}'} | "
                f"{row.get('ci_above_zero_seeds', '-')}/{seeds} | "
                f"{_pct(row.get('stage1_recall_with_lead'))} / "
                f"{_pct(row.get('two_stage_recall_with_lead'))} | "
                f"{row.get('stage1_false_alarm_drive_count', 0):,.0f} / "
                f"{row.get('two_stage_false_alarm_drive_count', 0):,.0f} | "
                f"{row.get('stage1_cost', 0):,.0f} / {row.get('two_stage_cost', 0):,.0f} |"
            )
    return "\n".join(lines) + "\n"


#: Written beside the results once they are logged; a second log is refused.
MLFLOW_RUN_FILE = "mlflow_run.json"


def two_stage_mlflow_metrics(results: dict[str, Any]) -> list[tuple[str, float, int]]:
    """(name, value, step) for an `experiment_two_stage.json`: the per-seed
    test numbers (step = seed) and the across-seed summary (step 0), named
    `net<N>_r<L>_...` with N and L in percent."""
    out: list[tuple[str, float, int]] = []

    def tag(net: float, level: float | None = None) -> str:
        name = f"net{round(net * 100)}"
        return name if level is None else f"{name}_r{round(level * 100)}"

    for report in results["runs"]:
        if report.get("status") != "ok":
            continue
        seed, net = int(report["seed"]), report["candidate_recall"]
        for one, two in zip(report["stage1"], report["two_stage"], strict=True):
            if one.get("threshold") is None or two.get("threshold") is None:
                continue
            prefix = tag(net, one["target_recall"])
            for side, row in (("stage1", one), ("cascade", two)):
                for key in ("precision", "recall", "recall_with_lead"):
                    if key in row["test"]:
                        out.append((f"{prefix}_test_{key}_{side}", row["test"][key], seed))
            if "paired_test" in two:
                gap = two["paired_test"]
                out += [
                    (f"{prefix}_gap", gap["difference"], seed),
                    (f"{prefix}_gap_ci_low", gap["difference_ci"][0], seed),
                    (f"{prefix}_gap_ci_high", gap["difference_ci"][1], seed),
                ]
    for net in results["summary"]:
        for key in (
            "seeds_trained",
            "train_rows",
            "train_positive_share",
            "test_ceiling",
            "test_ceiling_with_lead",
            "test_drive_share",
        ):
            if net.get(key) is not None:
                out.append((f"{tag(net['candidate_recall'])}_{key}", float(net[key]), 0))
        for row in net["levels"]:
            prefix = f"{tag(net['candidate_recall'], row['target_recall'])}_mean"
            for key in (
                "stage1_precision",
                "two_stage_precision",
                "stage1_recall",
                "two_stage_recall",
                "bootstrap_gap_mean",
                "ci_above_zero_seeds",
                "two_stage_better_seeds",
            ):
                if row.get(key) is not None:
                    out.append((f"{prefix}_{key}", float(row[key]), 0))
    return out


def log_two_stage_to_mlflow(
    out_dir: Path,
    *,
    extra_artifacts: tuple[Path, ...] = (),
    overwrite: bool = False,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Logs a finished two-stage comparison (`out_dir` holds its
    `experiment_two_stage.json`) as one MLflow run in the pipeline's
    experiment: settings as params, provenance as tags, metrics per seed and
    summarised, and every file in `out_dir` (plus `extra_artifacts`, e.g. the
    console log) as artifacts. Records the run id in `mlflow_run.json` and
    refuses to log the same results twice."""
    import mlflow

    refuse_overwrite([out_dir / MLFLOW_RUN_FILE], overwrite=overwrite)
    results = json.loads((out_dir / "experiment_two_stage.json").read_text())
    settings = results["settings"]
    mlflow_cfg = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["mlflow"]
    experiment_name = experiment_name or mlflow_cfg["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri or mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(experiment_name)
    provenance = ("git_commit", "git_dirty", "work_dir")
    with mlflow.start_run(run_name=f"cascade_wide_net_h{settings['horizon_days']}") as run:
        mlflow.set_tags(
            {
                "run_kind": "experiment",
                "experiment": "cascade_wide_net",
                **{key: str(settings.get(key)) for key in provenance},
            }
        )
        mlflow.log_params({k: v for k, v in settings.items() if k not in provenance})
        for name, value, step in two_stage_mlflow_metrics(results):
            mlflow.log_metric(name, value, step=step)
        for path in sorted(out_dir.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))
        for path in extra_artifacts:
            mlflow.log_artifact(str(path))
        run_id = run.info.run_id
    (out_dir / MLFLOW_RUN_FILE).write_text(
        json.dumps({"run_id": run_id, "experiment": experiment_name}, indent=2)
    )
    return run_id


def _anomaly_scores(forest: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Isolation-forest anomaly score per row (higher = more unusual), scored
    in chunks so `x` may be a memory map. Missing values become 0, which the
    forest (unlike LightGBM) cannot take as they are."""
    parts = []
    for start in range(0, len(x), PREDICT_CHUNK_ROWS):
        chunk = np.nan_to_num(np.asarray(x[start : start + PREDICT_CHUNK_ROWS], dtype=np.float32))
        parts.append(-forest.score_samples(chunk))
    return np.concatenate(parts) if parts else np.empty((0,))


def _with_column(x: np.ndarray, column: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """`x` (optionally only `rows`) with `column` appended as a last feature."""
    if rows is None:
        return np.column_stack([np.asarray(x), column])
    return np.column_stack([np.asarray(x[rows]), column[rows]])


def _predict_with_column(model: Any, x: np.ndarray, column: np.ndarray) -> np.ndarray:
    parts = []
    for start in range(0, len(x), PREDICT_CHUNK_ROWS):
        stop = start + PREDICT_CHUNK_ROWS
        parts.append(predict_proba_positive(model, _with_column(x[start:stop], column[start:stop])))
    return np.concatenate(parts) if parts else np.empty((0,))


def anomaly_stage_experiment(
    data: dict[str, Any],
    *,
    keep_fractions: tuple[float, ...] = (0.05, 0.01),
    fit_rows: int = 500_000,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """Does "learn what healthy looks like, then classify only the unusual
    drives" beat the pooled model?

    An isolation forest is fitted on healthy train rows only. Three things are
    then measured on the same test drives as every other variant, thresholds
    chosen on validation:

    1. the anomaly score on its own, as the alert score;
    2. the cascade: keep only the most anomalous `keep_fraction` of rows (the
       cut is that quantile of healthy train rows), train LightGBM on the
       train rows that pass, score the test rows that pass, 0 for the rest.
       `failing_drives_kept` is the recall ceiling the filter imposes;
    3. the pooled model with the anomaly score added as one more feature.

    The pooled `reg_spw` model is the reference in each table."""
    record_test_access(
        variant="anomaly_stage_experiment", purpose="experiment_model anomaly test scoring"
    )
    if "reg_spw" not in data.get("_scores", {}):
        run_variant("reg_spw", data)
    pooled_val, pooled_test = data["_scores"]["reg_spw"]
    base = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["model"]["params"]
    params = {**base, **REGULARIZED, "verbosity": -1}
    y_train, train_drives = data["y_train"], data["train_drives"]
    sub = data["es_idx"]

    healthy = np.flatnonzero(y_train == 0)
    rng = np.random.default_rng(0)
    fit_idx = np.sort(rng.choice(healthy, size=min(fit_rows, len(healthy)), replace=False))
    forest = IsolationForest(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    forest.fit(np.nan_to_num(data["x_train"][fit_idx]))
    train_a = _anomaly_scores(forest, data["x_train"])
    val_a = _anomaly_scores(forest, data["x_val"])
    test_a = _anomaly_scores(forest, data["x_test"])

    def table(val_scores: np.ndarray, test_scores: np.ndarray) -> list[dict[str, Any]]:
        return precision_at_recall_table(
            data["val_drives"],
            data["y_val"],
            val_scores,
            data["test_drives"],
            data["y_test"],
            test_scores,
        )

    report: dict[str, Any] = {
        "fit_rows": len(fit_idx),
        "pooled": table(pooled_val, pooled_test),
        "anomaly_score_alone": table(val_a, test_a),
        "cascade": [],
    }

    failing_test = set(data["test_drives"][data["y_test"] == 1].tolist())
    for keep in keep_fractions:
        cut = float(np.quantile(train_a[healthy], 1.0 - keep))
        cand_train, cand_val, cand_test = train_a >= cut, val_a >= cut, test_a >= cut
        kept_failing = set(data["test_drives"][cand_test & (data["y_test"] == 1)].tolist())
        entry: dict[str, Any] = {
            "keep_fraction": keep,
            "train_rows": int(cand_train.sum()),
            "train_positive_rows": int(y_train[cand_train].sum()),
            "test_rows": int(cand_test.sum()),
            "test_rows_total": len(test_a),
            "failing_test_drives": len(failing_test),
            "failing_drives_kept": len(kept_failing),
        }
        if entry["train_positive_rows"] < 20 or data["y_val"][cand_val].sum() < 5:
            entry["status"] = "too few candidate failures to train a second stage"
            report["cascade"].append(entry)
            continue
        rows_train, rows_val, rows_test = (
            np.flatnonzero(cand_train),
            np.flatnonzero(cand_val),
            np.flatnonzero(cand_test),
        )
        stage2 = _fit_lgbm(
            params,
            _with_column(data["x_train"], train_a, rows_train),
            y_train[rows_train],
            train_drives[rows_train],
            eval_x=_with_column(data["x_val"], val_a, rows_val),
            eval_y=data["y_val"][rows_val],
        )
        val_final, test_final = np.zeros(len(val_a)), np.zeros(len(test_a))
        val_final[rows_val] = predict_proba_positive(
            stage2, _with_column(data["x_val"], val_a, rows_val)
        )
        test_final[rows_test] = predict_proba_positive(
            stage2, _with_column(data["x_test"], test_a, rows_test)
        )
        entry["stage2_trees"] = int(stage2.best_iteration_ or stage2.n_estimators)
        entry["table"] = table(val_final, test_final)
        entry["status"] = "ok"
        report["cascade"].append(entry)
        del stage2

    # Pooled model with the anomaly score as an extra feature.
    x_plus = _with_column(data["x_train"], train_a)
    with_feature = _fit_lgbm(
        params,
        x_plus,
        y_train,
        train_drives,
        eval_x=_with_column(data["x_val"], val_a, sub),
        eval_y=data["y_val"][sub],
    )
    del x_plus
    gains = with_feature.booster_.feature_importance("gain")
    report["anomaly_feature_gain_share"] = float(gains[-1] / (gains.sum() or 1.0))
    report["pooled_plus_anomaly_feature"] = table(
        _predict_with_column(with_feature, data["x_val"], val_a),
        _predict_with_column(with_feature, data["x_test"], test_a),
    )
    return report


def _recall_cell(row: dict[str, Any]) -> str:
    if row.get("threshold") is None:
        return "unreachable"
    t = row["test"]
    # P@15% = the same alerts on a test set where 15% of drives fail.
    at_15 = t.get("precision_at_failure_rate", {}).get("0.15")
    reference = f" P@15%={at_15:.3f}" if at_15 is not None else ""
    return (
        f"P={t['precision']:.3f} R={t['recall']:.3f} "
        f"({t['caught_drive_count']}/{t['failing_drive_count']}, "
        f"FA={t['false_alarm_drive_count']}){reference}"
    )


def _print_anomaly_stage(report: dict[str, Any]) -> None:
    print("\n#### ANOMALY DETECTION (isolation forest on healthy rows) - TEST, drive level")
    print(f"  forest fitted on {report['fit_rows']} healthy train rows")
    columns: list[tuple[str, list[dict[str, Any]]]] = [
        ("pooled reg_spw", report["pooled"]),
        ("anomaly score alone", report["anomaly_score_alone"]),
        ("pooled + anomaly feature", report["pooled_plus_anomaly_feature"]),
    ]
    for entry in report["cascade"]:
        label = f"cascade keep {entry['keep_fraction']:.0%}"
        print(
            f"  {label}: test rows kept {entry['test_rows']}/{entry['test_rows_total']}, "
            f"failing drives kept {entry['failing_drives_kept']}/{entry['failing_test_drives']} "
            f"(recall ceiling), stage-2 train rows {entry['train_rows']} "
            f"({entry['train_positive_rows']} positive)"
        )
        if entry["status"] == "ok":
            columns.append((label, entry["table"]))
        else:
            print(f"     not trained: {entry['status']}")
    print(f"  anomaly feature share of gain: {report['anomaly_feature_gain_share']:.1%}")
    for i, pooled_row in enumerate(report["pooled"]):
        print(f"  recall>={pooled_row['target_recall']:.0%}:")
        for name, rows in columns:
            print(f"     {name:<26} {_recall_cell(rows[i])}")


def per_model_experiment(data: dict[str, Any], *, top_n: int) -> list[dict[str, Any]]:
    """Trains one LightGBM per drive family (the `top_n` families with the most
    failing validation drives) and compares it with the POOLED model on exactly
    the same test drives. Each family's threshold is chosen on that family's
    own validation drives (drive-level precision >= 50%, then the highest
    recall), so the comparison is what a per-family deployment would get.
    Families with too few failures are reported but not scored: with a
    handful of failing drives, any precision figure is noise."""
    record_test_access(
        variant="per_model_experiment", purpose="experiment_model per-family test scoring"
    )
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
        "--two-stage-seeds",
        type=int,
        default=None,
        metavar="N",
        help="With --two-stage: repeat it with N different seeds (fold assignment and "
        "model sampling) and print the mean and spread, to tell a real gain from chance. "
        "Default model.two_stage.experiment.seeds (1 without that block).",
    )
    parser.add_argument(
        "--candidate-recalls",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="With --two-stage: the Stage 1 nets to compare (validation recall the candidate "
        "threshold keeps). Default model.two_stage.experiment.candidate_recalls.",
    )
    parser.add_argument(
        "--recall-levels",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="With --two-stage: recall levels to compare the models at (each net uses the "
        "levels up to itself). Default model.two_stage.experiment.recall_levels.",
    )
    parser.add_argument("--lead-min-days", type=int, default=None)
    parser.add_argument("--cost-ratio", type=float, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument(
        "--two-stage-out",
        type=Path,
        default=None,
        help="With --two-stage: directory for experiment_two_stage.json and tables.md "
        "(default data/audit/cascade_wide_net_h<horizon>). Refuses to overwrite.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="With --two-stage: do not log the results as an MLflow run (they still go to "
        "--two-stage-out; `scripts/log_two_stage_experiment.py` can log them later).",
    )
    parser.add_argument(
        "--anomaly-stage",
        action="store_true",
        help="Fit an isolation forest on healthy train rows and compare, against the "
        "pooled model: the anomaly score alone, an anomaly-filter-then-classify cascade, "
        "and the pooled model with the anomaly score as a feature.",
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
    frame_meta = json.loads((work_dir / "frame_meta.json").read_text())
    feature_columns = frame_meta["feature_columns"]
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    horizon_days = int(frame_meta.get("horizon_days") or model_config["primary_horizon_days"])
    logger.info("experiment_horizon", horizon_days=horizon_days)

    if args.two_stage:
        two_stage_cfg = model_config["model"].get("two_stage", {})
        experiment_cfg = two_stage_cfg.get("experiment", {})

        def pick(cli: Any, key: str, fallback: Any) -> Any:
            return cli if cli is not None else experiment_cfg.get(key, fallback)

        two_stage_settings: dict[str, Any] = {
            "work_dir": work_dir.name,
            "horizon_days": horizon_days,
            "git_commit": _git_sha(),
            # Tracked changes not yet committed: saved as code.patch beside the
            # results, so the code that produced them can be rebuilt exactly.
            "git_dirty": bool(_tracked_diff()),
            "seeds": max(1, pick(args.two_stage_seeds, "seeds", 1)),
            "candidate_recalls": pick(args.candidate_recalls, "candidate_recalls", [0.5]),
            "recall_levels": pick(
                args.recall_levels, "recall_levels", list(TWO_STAGE_RECALL_LEVELS)
            ),
            "lead_min_days": pick(args.lead_min_days, "lead_min_days", 1),
            "cost_ratio": float(pick(args.cost_ratio, "cost_ratio", 20.0)),
            "n_boot": pick(args.n_boot, "n_boot", 1000),
            "n_folds": two_stage_cfg.get("n_folds", 3),
            "stage2_params": two_stage_cfg.get("stage2_params"),
            "early_stopping_max_negatives": two_stage_cfg.get("early_stopping_max_negatives"),
        }
        two_stage_out = args.two_stage_out or Path(f"data/audit/cascade_wide_net_h{horizon_days}")
        refuse_overwrite(
            [two_stage_out / "experiment_two_stage.json", two_stage_out / "tables.md"],
            overwrite=args.overwrite,
        )
        logger.info("two_stage_settings", **two_stage_settings)
        if two_stage_settings["git_dirty"]:
            two_stage_out.mkdir(parents=True, exist_ok=True)
            (two_stage_out / "code.patch").write_text(_tracked_diff())

    val_ids = pl.read_parquet(work_dir / "validation_ids.parquet")
    test_ids = pl.read_parquet(work_dir / "test_ids.parquet")
    train_ids = pl.read_parquet(work_dir / "train_ids.parquet")
    data: dict[str, Any] = {
        "feature_columns": feature_columns,
        "horizon_days": horizon_days,
        "train_ids": train_ids,
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
    for split, ids in (("val", val_ids), ("test", test_ids)):
        if "days_to_event" in ids.columns:
            data[f"{split}_days_to_event"] = ids["days_to_event"].to_numpy()

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
        two_stage_reports: list[dict[str, Any]] = []
        for seed in range(two_stage_settings["seeds"]):
            for report in two_stage_experiment(
                data,
                seed=seed,
                candidate_recalls=tuple(two_stage_settings["candidate_recalls"]),
                recall_levels=tuple(two_stage_settings["recall_levels"]),
                n_folds=two_stage_settings["n_folds"],
                lead_min_days=two_stage_settings["lead_min_days"],
                cost_ratio=two_stage_settings["cost_ratio"],
                n_boot=two_stage_settings["n_boot"],
                stage2_params=two_stage_settings["stage2_params"],
                early_stopping_max_negatives=two_stage_settings["early_stopping_max_negatives"],
            ):
                _print_two_stage(report)
                two_stage_reports.append(report)
            nets = summarize_two_stage_nets(two_stage_reports)
            # Rewritten after every seed, so a crash keeps the seeds done.
            two_stage_out.mkdir(parents=True, exist_ok=True)
            (two_stage_out / "experiment_two_stage.json").write_text(
                json.dumps(
                    {"settings": two_stage_settings, "runs": two_stage_reports, "summary": nets},
                    indent=2,
                    default=str,
                )
            )
            (two_stage_out / "tables.md").write_text(two_stage_tables(nets, two_stage_settings))
        for net in summarize_two_stage_nets(two_stage_reports):
            print(f"\n== net {net['candidate_recall']:.0%}")
            _print_two_stage_summary(net["levels"])
        print(f"\nTwo-stage results saved to {two_stage_out}")
        if not args.no_mlflow:
            run_id = log_two_stage_to_mlflow(two_stage_out, overwrite=args.overwrite)
            print(f"Logged to MLflow run {run_id}")
    if args.anomaly_stage:
        anomaly_report = anomaly_stage_experiment(data)
        _print_anomaly_stage(anomaly_report)
        (work_dir / "experiment_anomaly_stage.json").write_text(
            json.dumps(anomaly_report, indent=2, default=str)
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
