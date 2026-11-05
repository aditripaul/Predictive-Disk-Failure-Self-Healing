"""Entry point for `make train-merged`: the merged approach.

aryan_v2's XGBoost AFT and Cox survival models and its temporal CNN, run as
stages of THIS pipeline - its gold frame, splits, labels and drive-level
evaluation - side by side with the LightGBM (two-stage) and LSTM models, plus
validation-rank blends of them (`src/models/ensemble.py`).

Inputs (all produced by existing targets, in this order):

    make train TRAIN_ARGS="--keep-work-dir --save-scores"
        -> <scratch>/train_model_frame_*/ (x/y/ids per split)
        -> <scratch>/lightgbm_scores_h30.parquet (validation + test scores)
    make train-sequence                              -> sequence_model/scores_h30.npz
    make train-sequence SEQUENCE_ARGS="--arch tcn"   -> sequence_model/scores_h30_tcn.npz
    make train-merged

AFT and Cox train here on the kept work dir's train rows. Their targets use
outcomes only up to `splits.train_end` (the last train row is train_end -
horizon; `survival_bounds` censors at that row + horizon), asserted below.

Every model is compared on the SAME rows: the validation/test rows all of them
scored. Thresholds are chosen on validation only, then applied to test.

Writes (`[_TAG]` is empty unless `--tag TAG` is given, so two runs with different
tags never overwrite each other; `.smoke` is added by `--smoke`):
    data/audit/data_quality_reports/merged_evaluation_report[.smoke][_TAG].json
    <scratch>/merged[.smoke]/merged_scores_h30[_TAG].parquet
        (split, drive_id, date, label, one column per model)
    <scratch>/merged[.smoke]/agent_thresholds_h30[_TAG].json
        (action tiers of `merged_models.agent_ensemble`)
    <scratch>/merged[.smoke]/{aft,cox}_h30[_TAG].json   (the trained boosters)

The run refuses to start if any of these exist (`--overwrite` replaces them).
Unless `--smoke`, the run is also logged to MLflow (the experiment named in
configs/model.yaml): the merged_models config, per-model and per-ensemble
drive-level metrics, and the report, thresholds and boosters as artifacts. The
MLflow run id is written into the report as `mlflow_run_id`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

import mlflow
from src.logging_config import configure_logging, get_logger
from src.models.access_log import record_test_access
from src.models.ensemble import (
    assert_no_row_overlap,
    assert_split_dates,
    assert_survival_censored,
    rank_blend,
)
from src.models.evaluation import (
    compute_auprc,
    compute_auroc,
    drive_level_metrics,
    drive_level_table,
    precision_at_recall_table,
    recall_at_false_alarm_table,
)
from src.models.survival import survival_bounds, train_aft, train_cox
from src.models.threshold import (
    precision_targets_from_lift,
    tune_action_tiers,
    tune_drive_level_threshold,
)
from src.models.training import early_stopping_subset
from src.models.uncertainty import drive_level_bootstrap_ci
from src.outputs import refuse_overwrite
from src.resource_limits import apply_memory_limit_from_config

logger = get_logger(__name__)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
SPLITS = ("validation", "test")
BASE_MODELS = ("lightgbm", "aft", "cox", "lstm", "tcn")
PREDICT_CHUNK_ROWS = 1_000_000


TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DRIVE_METRICS = ("precision", "recall", "auprc", "auroc")


def tag_suffix(tag: str | None) -> str:
    """`_TAG` for a run tag, "" for none. Rejects anything that is not safe in a file name."""
    if not tag:
        return ""
    if not TAG_PATTERN.fullmatch(tag):
        raise SystemExit(f"--tag {tag!r} must match {TAG_PATTERN.pattern}")
    return f"_{tag}"


def output_paths(
    audit_dir: Path, out_dir: Path, *, horizon_days: int, smoke_suffix: str, tag: str | None
) -> dict[str, Path]:
    """Every file `main` writes, named so that a different tag gives a different file."""
    t = tag_suffix(tag)
    return {
        "report": audit_dir / f"merged_evaluation_report{smoke_suffix}{t}.json",
        "scores": out_dir / f"merged_scores_h{horizon_days}{t}.parquet",
        "thresholds": out_dir / f"agent_thresholds_h{horizon_days}{t}.json",
        "aft": out_dir / f"aft_h{horizon_days}{t}.json",
        "cox": out_dir / f"cox_h{horizon_days}{t}.json",
    }


def mlflow_metrics(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Flat finite drive-level metrics: `<model>__<split>_drive_<metric>`."""
    flat: dict[str, float] = {}
    for model, result in results.items():
        for split in SPLITS:
            drive_level = result.get(f"{split}_drive_level") or {}
            for metric in DRIVE_METRICS:
                value = drive_level.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    flat[f"{model}__{split}_drive_{metric}"] = float(value)
    return flat


def scratch_base(data_config: dict) -> Path:
    configured = data_config.get("resource_limits", {}).get("scratch_dir")
    return Path(configured) if configured else Path(data_config["gold_dir"]).parent / "tmp"


def find_work_dir(base: Path) -> Path:
    candidates = sorted(base.glob("train_model_frame_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f'No kept work dir under {base}; run `make train TRAIN_ARGS="--keep-work-dir '
            '--save-scores"` first.'
        )
    return candidates[-1]


def split_windows(model_config: dict, horizon_days: int) -> dict[str, dt.date]:
    splits = model_config["splits"]
    train_end = dt.date.fromisoformat(str(splits["train_end"]))
    validation_end = dt.date.fromisoformat(str(splits["validation_end"]))
    test_end = dt.date.fromisoformat(str(splits["test_end"]))
    if not splits.get("purge_label_window", True) or not splits.get(
        "validation_after_train_only", True
    ):
        raise SystemExit(
            "train_merged_models assumes splits.purge_label_window and "
            "splits.validation_after_train_only are on (configs/model.yaml)."
        )
    return {
        "train_end": train_end,
        "train_last_row": train_end - dt.timedelta(days=horizon_days),
        "validation_start": train_end + dt.timedelta(days=1),
        "validation_end": validation_end,
        "test_start": validation_end + dt.timedelta(days=1),
        "test_end": test_end,
    }


def predict_in_chunks(model: Any, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), PREDICT_CHUNK_ROWS):
        stop = min(start + PREDICT_CHUNK_ROWS, len(x))
        out[start:stop] = model.predict_proba(np.ascontiguousarray(x[start:stop]))[:, 1]
    return out


def weights_for(observed: np.ndarray, weighting: str, power: float) -> np.ndarray | None:
    """`none`, or `spw`: observed failures up-weighted by (n_censored / n_observed) ** power,
    as experiment_model's `xgboost_aft_spw`."""
    if weighting == "none":
        return None
    if weighting != "spw":
        raise SystemExit(f"unknown survival weighting {weighting!r}; expected none or spw")
    n_pos = max(int(observed.sum()), 1)
    w = np.ones(len(observed), dtype=np.float32)
    w[observed == 1] = ((observed == 0).sum() / n_pos) ** power
    return w


SEQUENCE_KEYS = (
    "drive_ids",
    "start_date",
    *(
        f"{split}_{field}"
        for split in SPLITS
        for field in ("drive_idx", "day_idx", "label", "score")
    ),
)
LIGHTGBM_COLUMNS = ("split", "drive_id", "date", "label", "score")


def check_inputs(lgbm_path: Path, seq_paths: dict[str, Path]) -> None:
    """Raises SystemExit listing every unusable score input, before any model trains.

    The survival stages take about ten minutes, so an input that is missing or
    in the wrong format must fail here and not after them. Reads only the
    LightGBM parquet's schema and the npz member names, never the scores."""
    problems: list[str] = []
    if not lgbm_path.exists():
        problems.append(f"{lgbm_path}: not found (`make train` with --save-scores)")
    else:
        have = set(pl.scan_parquet(lgbm_path).collect_schema().names())
        if missing := [c for c in LIGHTGBM_COLUMNS if c not in have]:
            problems.append(f"{lgbm_path}: no column(s) {missing}")
    for name, path in seq_paths.items():
        if not path.exists():
            problems.append(f"{path}: not found (`make train-sequence`, {name})")
            continue
        with np.load(path, allow_pickle=False) as npz:
            if missing := [k for k in SEQUENCE_KEYS if k not in npz.files]:
                problems.append(
                    f"{path}: no {missing}; it predates the merged approach, re-run "
                    f"`make train-sequence` ({name}) on this branch"
                )
    if problems:
        listed = "\n  ".join(problems)
        raise SystemExit(f"unusable inputs:\n  {listed}")


def load_sequence_scores(path: Path, name: str) -> dict[str, pl.DataFrame]:
    """`{split: (drive_id, date, label, <name>)}` from a train_sequence_model npz."""
    npz = np.load(path, allow_pickle=False)
    for key in ("validation_day_idx", "test_day_idx", "start_date"):
        if key not in npz.files:
            raise SystemExit(
                f"{path} has no {key!r}: it predates the merged approach. Re-run "
                "`make train-sequence` on this branch."
            )
    start = np.datetime64(str(npz["start_date"]), "D")
    drive_ids = npz["drive_ids"]
    out = {}
    for split in SPLITS:
        out[split] = pl.DataFrame(
            {
                "drive_id": drive_ids[npz[f"{split}_drive_idx"]],
                "date": (start + npz[f"{split}_day_idx"].astype("timedelta64[D]")),
                f"label_{name}": npz[f"{split}_label"].astype(np.int8),
                name: npz[f"{split}_score"].astype(np.float64),
            }
        ).with_columns(pl.col("date").cast(pl.Date))
    return out


def evaluate_model(
    name: str,
    val: pl.DataFrame,
    test: pl.DataFrame,
    model_config: dict,
) -> dict[str, Any]:
    """This pipeline's evaluation protocol (as train_sequence_model's report):
    the drive-level threshold tuned on validation and applied to test."""
    vd, vy, vs = val["drive_id"].to_numpy(), val["label"].to_numpy(), val[name].to_numpy()
    td, ty, ts = test["drive_id"].to_numpy(), test["label"].to_numpy(), test[name].to_numpy()
    threshold_cfg = model_config["threshold"]
    threshold_result = tune_drive_level_threshold(
        vd,
        vy,
        vs,
        target_precision=threshold_cfg["target_precision"],
        target_recall_range=tuple(threshold_cfg["target_recall_range"]),
    )
    threshold = threshold_result["threshold"]
    val_drive_labels, _ = drive_level_table(vd, vy, vs)
    tier_targets = precision_targets_from_lift(
        threshold_cfg["action_tier_lift"], float(val_drive_labels.mean())
    )
    return {
        "threshold": threshold_result,
        "action_tiers": tune_action_tiers(vd, vy, vs, precision_targets=tier_targets),
        "validation_drive_level": drive_level_metrics(vd, vy, vs, threshold),
        "test_drive_level": drive_level_metrics(td, ty, ts, threshold),
        "test_drive_ci": drive_level_bootstrap_ci(td, ty, ts, threshold),
        "precision_at_recall": precision_at_recall_table(vd, vy, vs, td, ty, ts),
        "test_recall_at_false_alarm": recall_at_false_alarm_table(td, ty, ts),
        "validation_row_auroc": compute_auroc(vy, vs),
        "validation_row_auprc": compute_auprc(vy, vs),
        "test_row_auroc": compute_auroc(ty, ts),
        "test_row_auprc": compute_auprc(ty, ts),
    }


def label_summary(ids: pl.DataFrame, y: np.ndarray) -> dict[str, Any]:
    drive_labels = (
        pl.DataFrame({"drive_id": ids["drive_id"], "y": y})
        .group_by("drive_id")
        .agg(pl.col("y").max())
    )
    return {
        "rows": int(len(y)),
        "positive_rows": int((y == 1).sum()),
        "positive_row_rate": float((y == 1).mean()) if len(y) else None,
        "drives": int(drive_labels.height),
        "failing_drives": int(drive_labels["y"].sum()),
        "min_date": str(ids["date"].min()),
        "max_date": str(ids["date"].max()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work-dir", type=Path, help="default: newest kept train_model work dir")
    parser.add_argument("--lightgbm-scores", type=Path)
    parser.add_argument("--lstm-scores", type=Path)
    parser.add_argument("--tcn-scores", type=Path)
    parser.add_argument(
        "--tag",
        help="Label for this run; appended to every output file name so runs do not overwrite "
        "each other (pass the tagged parquet to `agent-fleet --scores-path`).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this tag's existing outputs. Without it the run refuses to start if any "
        "output file already exists.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small-subset run: writes .smoke outputs, records no test access. Not a result.",
    )
    args = parser.parse_args(argv)

    tag_suffix(args.tag)  # fail on a bad tag before any work
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    merged_cfg = model_config["merged_models"]
    seed = int(merged_cfg.get("seed", 0))
    np.random.seed(seed)
    scratch = scratch_base(data_config)
    work_dir = args.work_dir or find_work_dir(scratch)
    meta = json.loads((work_dir / "frame_meta.json").read_text())
    horizon_days = int(meta.get("horizon_days") or model_config["primary_horizon_days"])
    windows = split_windows(model_config, horizon_days)
    suffix = ".smoke" if args.smoke else ""
    audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    out_dir = scratch / f"merged{suffix}"
    paths = output_paths(
        audit_dir, out_dir, horizon_days=horizon_days, smoke_suffix=suffix, tag=args.tag
    )
    refuse_overwrite(paths.values(), overwrite=args.overwrite)
    lgbm_path = args.lightgbm_scores or scratch / f"lightgbm_scores_h{horizon_days}.parquet"
    seq_dir = scratch / f"sequence_model{suffix}"
    seq_paths = {
        "lstm": args.lstm_scores or seq_dir / f"scores_h{horizon_days}.npz",
        "tcn": args.tcn_scores or seq_dir / f"scores_h{horizon_days}_tcn.npz",
    }
    check_inputs(lgbm_path, seq_paths)
    t0 = time.perf_counter()

    ids = {
        "train": pl.read_parquet(work_dir / "train_ids.parquet"),
        "validation": pl.read_parquet(work_dir / "validation_ids.parquet"),
        "test": pl.read_parquet(work_dir / "test_ids.parquet"),
    }
    y = {
        "train": np.load(work_dir / "y_train.npy"),
        "validation": np.load(work_dir / "y_val.npy"),
        "test": np.load(work_dir / "y_test.npy"),
    }
    for split in ids:
        if ids[split].height != len(y[split]):
            raise AssertionError(f"{split}: {ids[split].height} ids vs {len(y[split])} labels")

    # --- leak and split assertions -------------------------------------------------
    split_dates = assert_split_dates(
        {k: v["date"] for k, v in ids.items()},
        train_last_row=windows["train_last_row"],
        validation_start=windows["validation_start"],
        validation_end=windows["validation_end"],
        test_start=windows["test_start"],
        test_end=windows["test_end"],
    )
    assert_no_row_overlap(ids)
    labels = {split: label_summary(ids[split], y[split]) for split in ids}
    logger.info(
        "merged_split_labels", **{f"{k}_{m}": v for k, d in labels.items() for m, v in d.items()}
    )
    print(json.dumps({"split_dates": split_dates, "labels": labels}, indent=2))

    # --- survival stages (AFT, Cox) on the train rows ----------------------------
    train_dates = ids["train"]["date"].to_numpy().astype("datetime64[D]")
    lower, upper = survival_bounds(
        train_dates,
        ids["train"]["event_type"].to_numpy(),
        ids["train"]["days_to_event"].to_numpy().astype(np.float64),
        cutoff_date=np.datetime64(windows["train_last_row"], "D"),
        horizon_days=horizon_days,
    )
    assert_survival_censored(train_dates, lower, upper, outcome_cutoff=windows["train_end"])
    observed = np.isfinite(upper).astype(np.int8)
    survival_summary = {
        "outcome_cutoff": str(windows["train_end"]),
        "observed_failures": int(observed.sum()),
        "censored_rows": int((observed == 0).sum()),
    }

    x_train = np.load(work_dir / "x_train.npy")
    x_val = np.load(work_dir / "x_val.npy", mmap_mode="r")
    x_test = np.load(work_dir / "x_test.npy", mmap_mode="r")
    es_idx = early_stopping_subset(
        y["validation"], max_negatives=merged_cfg.get("early_stopping_max_negatives"), seed=seed
    )
    eval_x = np.ascontiguousarray(x_val[es_idx])
    eval_y = y["validation"][es_idx]
    power = float(model_config["model"].get("positive_weight_power", 0.5))
    common_train = {
        "eval_x": eval_x,
        "eval_y": eval_y,
        "num_boost_round": int(merged_cfg.get("num_boost_round", 1000)),
        "early_stopping_rounds": int(merged_cfg.get("early_stopping_rounds", 50)),
    }
    survival_scores: dict[str, dict[str, np.ndarray]] = {}
    trees: dict[str, int] = {}
    boosters: dict[str, Any] = {}
    for name, trainer, extra in (
        ("aft", train_aft, {"horizon_days": horizon_days}),
        ("cox", train_cox, {}),
    ):
        t = time.perf_counter()
        model = trainer(
            x_train,
            lower,
            upper,
            sample_weight=weights_for(observed, merged_cfg[name]["weighting"], power),
            params={"seed": seed},
            **common_train,
            **extra,
        )
        trees[name] = model.tree_count
        boosters[name] = model.booster
        survival_scores[name] = {
            "validation": predict_in_chunks(model, x_val),
            "test": predict_in_chunks(model, x_test),
        }
        logger.info(
            "merged_survival_trained",
            model=name,
            trees=model.tree_count,
            seconds=round(time.perf_counter() - t, 1),
        )
    del x_train, eval_x

    # --- every model's scores, joined on (drive_id, date) ------------------------
    tables: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        tables[split] = (
            ids[split]
            .select("drive_id", "date")
            .with_columns(
                pl.Series("label", y[split].astype(np.int8)),
                pl.Series("aft", survival_scores["aft"][split]),
                pl.Series("cox", survival_scores["cox"][split]),
            )
        )

    lgbm = pl.read_parquet(lgbm_path)
    member_frames: dict[str, dict[str, pl.DataFrame]] = {
        "lightgbm": {
            split: lgbm.filter(pl.col("split") == split).select(
                "drive_id",
                pl.col("date").cast(pl.Date),
                pl.col("label").cast(pl.Int8).alias("label_lightgbm"),
                pl.col("score").cast(pl.Float64).alias("lightgbm"),
            )
            for split in SPLITS
        },
        **{name: load_sequence_scores(path, name) for name, path in seq_paths.items()},
    }
    coverage: dict[str, dict[str, float]] = {}
    min_cov = float(merged_cfg.get("min_join_coverage", 0.95))
    for split in SPLITS:
        base_rows = tables[split].height
        coverage[split] = {}
        for name, frames in member_frames.items():
            frame = frames[split]
            if frame.select("drive_id", "date").is_duplicated().any():
                raise AssertionError(f"{name} {split} scores repeat a (drive_id, date) row")
            tables[split] = tables[split].join(frame, on=["drive_id", "date"], how="left")
            joined = tables[split].filter(pl.col(name).is_not_null())
            mismatched = joined.filter(pl.col("label") != pl.col(f"label_{name}")).height
            if mismatched:
                raise AssertionError(f"{name} {split}: {mismatched} rows disagree on the label")
            tables[split] = tables[split].drop(f"label_{name}")
            coverage[split][name] = joined.height / base_rows
            if coverage[split][name] < min_cov:
                raise SystemExit(
                    f"{name} scored {coverage[split][name]:.4f} of the {split} rows, below "
                    f"merged_models.min_join_coverage {min_cov}: the models were not run on the "
                    "same frame."
                )
        before = tables[split].height
        tables[split] = tables[split].drop_nulls(list(BASE_MODELS)).sort(["drive_id", "date"])
        coverage[split]["common_rows"] = tables[split].height
        coverage[split]["frame_rows"] = before
    print(json.dumps({"coverage": coverage}, indent=2))

    # --- validation-rank blends ---------------------------------------------------
    ensembles: dict[str, list[str]] = {k: list(v) for k, v in merged_cfg["ensembles"].items()}
    val_ref = {m: tables["validation"][m].to_numpy() for m in BASE_MODELS}
    for ens_name, members in ensembles.items():
        unknown = set(members) - set(BASE_MODELS)
        if unknown:
            raise SystemExit(f"ensemble {ens_name!r} has unknown members {sorted(unknown)}")
        for split in SPLITS:
            target = {m: tables[split][m].to_numpy() for m in members}
            tables[split] = tables[split].with_columns(
                pl.Series(ens_name, rank_blend(val_ref, target, members))
            )

    # --- evaluation, this pipeline's protocol -------------------------------------
    if not args.smoke:
        record_test_access(variant="merged_models", purpose="train_merged_models comparison")
    results = {
        name: evaluate_model(name, tables["validation"], tables["test"], model_config)
        for name in (*BASE_MODELS, *ensembles)
    }

    report = {
        "smoke": args.smoke,
        "horizon_days": horizon_days,
        "seed": seed,
        "work_dir": str(work_dir),
        "split_windows": {k: str(v) for k, v in windows.items()},
        "split_dates": split_dates,
        "labels": labels,
        "survival": survival_summary,
        "survival_trees": trees,
        "coverage": coverage,
        "common_rows_labels": {
            split: label_summary(tables[split], tables[split]["label"].to_numpy())
            for split in SPLITS
        },
        "ensembles": ensembles,
        "models": results,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report["tag"] = args.tag

    # A smoke run is not a result, so it is not tracked (as it records no test access).
    if args.smoke:
        tracking: Any = contextlib.nullcontext()
    else:
        mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(model_config["mlflow"]["experiment_name"])
        tracking = mlflow.start_run(run_name=f"merged{tag_suffix(args.tag)}_h{horizon_days}")

    with tracking as run:
        report["mlflow_run_id"] = run.info.run_id if run is not None else None
        paths["report"].write_text(json.dumps(report, indent=2, default=str))

        pl.concat(
            [tables[s].with_columns(pl.lit(s).alias("split")) for s in SPLITS], how="vertical"
        ).write_parquet(paths["scores"], compression="zstd")
        agent_name = merged_cfg["agent_ensemble"]
        paths["thresholds"].write_text(
            json.dumps(
                {
                    "score_column": agent_name,
                    "action_tiers": results[agent_name]["action_tiers"],
                    "threshold": results[agent_name]["threshold"]["threshold"],
                },
                indent=2,
                default=str,
            )
        )
        for name, booster in boosters.items():
            booster.save_model(str(paths[name]))

        if run is not None:
            mlflow.log_params(
                {
                    "horizon_days": horizon_days,
                    "seed": seed,
                    "tag": args.tag or "",
                    "agent_ensemble": agent_name,
                    "ensembles": json.dumps(ensembles),
                    "aft_weighting": merged_cfg["aft"]["weighting"],
                    "cox_weighting": merged_cfg["cox"]["weighting"],
                    "aft_trees": trees["aft"],
                    "cox_trees": trees["cox"],
                    "work_dir": str(work_dir),
                }
            )
            mlflow.log_metrics(mlflow_metrics(results))
            for path in paths.values():
                mlflow.log_artifact(str(path))
    logger.info(
        "merged_report_written",
        path=str(paths["report"]),
        out_dir=str(out_dir),
        mlflow_run_id=report["mlflow_run_id"],
    )


if __name__ == "__main__":
    main()
