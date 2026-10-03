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
from src.models.evaluation import compute_auprc, drive_level_metrics, drive_level_table
from src.models.features import feature_matrix
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.threshold import tune_drive_level_threshold
from src.models.training import early_stopping_subset, predict_proba_positive
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
#: exclude: none | identity | non_windowed; weighting: is_unbalance | spw[:power] | drive
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


def _excluded_indices(feature_columns: list[str], mode: str) -> list[int]:
    if mode == "identity":
        return [i for i, c in enumerate(feature_columns) if c in IDENTITY_COLUMNS]
    if mode == "non_windowed":
        return [
            i
            for i, c in enumerate(feature_columns)
            if not _WINDOWED.search(c) and not c.endswith(("_zscore", "_slope"))
        ]
    return []


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
    arrays = [data["x_train"], data["x_val"], data["x_test"]]
    with _Zeroed(arrays, idx):
        weights = _sample_weights(data["y_train"], data["train_drives"], weighting)
        sub = data["es_idx"]
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
                eval_set=[(data["x_val"][sub], data["y_val"][sub])],
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
                eval_X=data["x_val"][sub],
                eval_y=data["y_val"][sub],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
            )
        val_scores = predict_proba_positive(model, data["x_val"])
        test_scores = predict_proba_positive(model, data["x_test"])
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

    test_df = pl.read_parquet(work_dir / "test_df.parquet")
    data: dict[str, Any] = {
        "feature_columns": feature_columns,
        "x_train": np.load(work_dir / "x_train.npy"),
        "y_train": np.load(work_dir / "y_train.npy"),
        "train_drives": pl.read_parquet(work_dir / "train_ids.parquet")["drive_id"].to_numpy(),
        "val_ids": pl.read_parquet(work_dir / "validation_ids.parquet"),
        "x_val": np.load(work_dir / "x_val.npy"),
        "y_val": np.load(work_dir / "y_val.npy"),
        "val_drives": pl.read_parquet(work_dir / "validation_ids.parquet")["drive_id"].to_numpy(),
        "x_test": feature_matrix(test_df, feature_columns),
        "y_test": test_df["label"].to_numpy(),
        "test_drives": test_df["drive_id"].to_numpy(),
        "test_ids": test_df.select("drive_id", "date", "event_type", "days_to_event"),
    }
    del test_df
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
    if args.deep_dive:
        report = deep_dive(args.deep_dive, data)
        _print_deep_dive(report)
        (work_dir / "experiment_deep_dive.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
    print(f"\nResults saved to {work_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
