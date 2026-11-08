"""Freeze the stacked Stage 2 (LightGBM + Cox) as a scorer, from validation only.

Phase S of docs/agent_knowledge_base.md. The stack runs of runbook 1.1.4a saved no
models and no per-drive scores, so the knowledge base and the agent cannot use them.
This script refits exactly what `make experiment-model ... --two-stage --stage2-kinds
lightgbm cox stack` fits for one seed and one net (Stage 1, its out-of-fold train
scores, the LightGBM and Cox Stage 2, the stack weights), using the same functions and
settings, then:

1. checks the refit against the frozen run's saved report (`--reference`): candidate
   threshold, candidate counts, tree counts, stack weights and validation average
   precision must match, or nothing is written;
2. saves the stack to `--out` (`src/models/stacking.py::save_stacked_two_stage`);
3. writes the validation scores of three arms (stack, LightGBM Stage 2, Cox Stage 2) in
   the score-table layout (`split, drive_id, date, label, score, score_lightgbm,
   score_cox`) to `--scores-out`;
4. derives, per arm, the lift tiers (`tune_action_tiers`, the repository's rule), the
   cost tiers (`tune_cost_tiers`, configs/model.yaml threshold.action_cost_ratio) and the
   primary threshold, and writes one thresholds file per arm in the layout
   `src/agent/fleet.py::load_precomputed_scores` reads;
5. writes a cost-sensitivity table on validation for the three arms and for the frozen
   LightGBM cascade (`--cascade-scores`, its validation rows), a README and checksums.

The test matrix (`x_test.npy`) is never opened, so this records no test access.

    uv run python scripts/freeze_stack_scorer.py --work-dir data/tmp/train_model_frame_xjxdieoe
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.experiment_model import (  # noqa: E402
    MODEL_CONFIG_PATH,
    REGULARIZED,
    _train_survival_bounds,
)
from src.logging_config import configure_logging, get_logger  # noqa: E402
from src.models.evaluation import drive_level_table  # noqa: E402
from src.models.stacking import fit_stack, save_stacked_two_stage  # noqa: E402
from src.models.threshold import (  # noqa: E402
    ACTION_TIERS,
    precision_targets_from_lift,
    tune_action_tiers,
    tune_cost_tiers,
    tune_drive_level_threshold,
)
from src.models.training import early_stopping_subset, predict_proba_positive  # noqa: E402
from src.models.two_stage import (  # noqa: E402
    TwoStageModel,
    fit_second_stage,
    lightgbm_fitter,
    out_of_fold_scores,
    tree_count,
)
from src.resource_limits import apply_memory_limit_from_config  # noqa: E402

logger = get_logger(__name__)

ARMS = {"stack": "score", "lightgbm": "score_lightgbm", "cox": "score_cox"}
COST_GRID = (2, 5, 10, 20, 50, 100)


def load_validation_data(work_dir: Path) -> dict[str, Any]:
    """The experiment's `data` dict without any test array."""
    meta = json.loads((work_dir / "frame_meta.json").read_text())
    train_ids = pl.read_parquet(work_dir / "train_ids.parquet")
    val_ids = pl.read_parquet(work_dir / "validation_ids.parquet")
    y_val = np.load(work_dir / "y_val.npy")
    return {
        "horizon_days": int(meta["horizon_days"]),
        "train_ids": train_ids,
        "x_train": np.load(work_dir / "x_train.npy"),
        "y_train": np.load(work_dir / "y_train.npy"),
        "train_drives": train_ids["drive_id"].to_numpy(),
        "x_val": np.load(work_dir / "x_val.npy", mmap_mode="r"),
        "y_val": y_val,
        "val_ids": val_ids,
        "val_drives": val_ids["drive_id"].to_numpy(),
        "es_idx": early_stopping_subset(y_val),
    }


def fit_arms(data: dict[str, Any], *, seed: int, net: float) -> dict[str, Any]:
    """Stage 1, the LightGBM and Cox Stage 2 and their stack, as `two_stage_experiment`
    fits them for this seed and net."""
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    model_cfg = model_config["model"]
    two_stage_cfg = model_cfg.get("two_stage", {})
    experiment_cfg = two_stage_cfg.get("experiment", {})
    n_folds = int(two_stage_cfg.get("n_folds", 3))
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
    train_oof = out_of_fold_scores(
        fit,
        data["x_train"],
        data["y_train"],
        data["train_drives"],
        n_estimators=tree_count(stage1),
        n_folds=n_folds,
        seed=seed,
    )
    logger.info("freeze_stage1_done", seconds=round(time.perf_counter() - t0))
    bounds = _train_survival_bounds(data)
    members: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}
    for kind in ("lightgbm", "cox"):
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
            stage2_params=two_stage_cfg.get("stage2_params"),
            early_stopping_max_negatives=two_stage_cfg.get("early_stopping_max_negatives"),
            stage2_kind=kind,
            train_survival_bounds=bounds,
            horizon_days=data["horizon_days"],
            survival_params=experiment_cfg.get("survival_params"),
        )
        if model is None:
            raise SystemExit(f"{kind} Stage 2 not trained: {report['status']}")
        members[kind] = model.stage2
        reports[kind] = report
        logger.info("freeze_stage2_done", kind=kind, trees=report["stage2_trees"])
    threshold = reports["lightgbm"]["candidate_threshold"]
    stacked, stack_report = fit_stack(
        stage1,
        members,
        x_val=data["x_val"],
        y_val=data["y_val"],
        val_drives=data["val_drives"],
        val_stage1_scores=val_s1,
        candidate_threshold=threshold,
    )
    if stacked is None:
        raise SystemExit(f"stack not built: {stack_report['status']}")
    return {
        "stage1": stage1,
        "val_s1": val_s1,
        "models": {
            "stack": stacked,
            "lightgbm": TwoStageModel(stage1, members["lightgbm"], threshold),
            "cox": TwoStageModel(stage1, members["cox"], threshold),
        },
        "reports": {**reports, "stack": stack_report},
        "settings": {
            "n_folds": n_folds,
            "stage2_params": two_stage_cfg.get("stage2_params"),
            "early_stopping_max_negatives": two_stage_cfg.get("early_stopping_max_negatives"),
            "survival_params": experiment_cfg.get("survival_params"),
        },
    }


def reproduction_check(
    fitted: dict[str, Any], reference_path: Path, *, seed: int, net: float
) -> dict:
    """Compares the refit with the frozen run's report for the same seed and net."""
    runs = json.loads(reference_path.read_text())["runs"]
    ref = {
        r["stage2_kind"]: r
        for r in runs
        if r["seed"] == seed and abs(r["candidate_recall"] - net) < 1e-9
    }
    missing = {"lightgbm", "cox", "stack"} - set(ref)
    if missing:
        raise SystemExit(
            f"{reference_path} has no seed {seed}, net {net} report for {sorted(missing)}"
        )
    got = fitted["reports"]
    checks: list[dict[str, Any]] = []

    def add(name: str, expected: Any, actual: Any, tol: float = 0.0) -> None:
        if isinstance(expected, float) or isinstance(actual, float):
            ok = abs(float(expected) - float(actual)) <= tol
        else:
            ok = expected == actual
        checks.append({"check": name, "expected": expected, "actual": actual, "ok": bool(ok)})

    add(
        "candidate_threshold",
        ref["lightgbm"]["candidate_threshold"],
        got["lightgbm"]["candidate_threshold"],
        1e-12,
    )
    add("stage1_trees", ref["lightgbm"]["stage1_trees"], got["lightgbm"]["stage1_trees"])
    for key in ("train_rows", "train_positive_rows", "validation_rows", "validation_positive_rows"):
        add(
            f"candidates.{key}",
            ref["lightgbm"]["candidates"][key],
            got["lightgbm"]["candidates"][key],
        )
    for kind in ("lightgbm", "cox"):
        add(f"{kind}.stage2_trees", ref[kind]["stage2_trees"], got[kind]["stage2_trees"])
    for member, weight in ref["stack"]["stack"]["weights"].items():
        add(f"stack.weights.{member}", weight, got["stack"]["stack"]["weights"][member], 1e-12)
    for name, value in ref["stack"]["stack"]["validation_drive_ap"].items():
        add(
            f"stack.validation_drive_ap.{name}",
            value,
            got["stack"]["stack"]["validation_drive_ap"][name],
            1e-9,
        )
    return {
        "reference": str(reference_path),
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


def drive_point(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, Any]:
    """Drive-level precision, recall and counts at a threshold (None = never fires)."""
    if threshold is None:
        return {"threshold": None, "flagged": 0, "caught": 0, "precision": None, "recall": 0.0}
    flagged = scores >= threshold
    caught = int((flagged & (labels == 1)).sum())
    n_flagged = int(flagged.sum())
    return {
        "threshold": float(threshold),
        "flagged": n_flagged,
        "flagged_share": n_flagged / len(labels),
        "caught": caught,
        "precision": caught / n_flagged if n_flagged else None,
        "recall": caught / int(labels.sum()),
    }


def arm_tiers(drive_ids: np.ndarray, y: np.ndarray, scores: np.ndarray, model_config: dict) -> dict:
    """Lift tiers, cost tiers, the primary threshold and their validation points."""
    labels, drive_scores = drive_level_table(drive_ids, y, scores)
    threshold_cfg = model_config["threshold"]
    rate = float(labels.mean())
    lift = tune_action_tiers(
        drive_ids,
        y,
        scores,
        precision_targets=precision_targets_from_lift(threshold_cfg["action_tier_lift"], rate),
    )
    cost = tune_cost_tiers(drive_ids, y, scores, cost_ratios=threshold_cfg["action_cost_ratio"])
    primary = tune_drive_level_threshold(
        drive_ids,
        y,
        scores,
        target_precision=threshold_cfg["target_precision"],
        target_recall_range=tuple(threshold_cfg["target_recall_range"]),
    )
    return {
        "failure_rate": rate,
        "drives": int(len(labels)),
        "failing_drives": int(labels.sum()),
        "action_tiers": lift,
        "action_tiers_cost": cost,
        "threshold": float(primary["threshold"]),
        "validation": {
            "lift": {t: drive_point(labels, drive_scores, lift[t]) for t in ACTION_TIERS},
            "cost": {t: drive_point(labels, drive_scores, cost[t]) for t in ACTION_TIERS},
            "primary": drive_point(labels, drive_scores, float(primary["threshold"])),
        },
        "cost_ratios": dict(threshold_cfg["action_cost_ratio"]),
        "lift_targets": dict(threshold_cfg["action_tier_lift"]),
    }


def cost_sensitivity(drive_ids: np.ndarray, y: np.ndarray, scores: np.ndarray) -> list[dict]:
    """For each ratio r of COST_GRID, the cost-optimal single threshold and its point."""
    labels, drive_scores = drive_level_table(drive_ids, y, scores)
    rows = []
    for r in COST_GRID:
        thr = tune_cost_tiers(drive_ids, y, scores, cost_ratios={t: r for t in ACTION_TIERS})[
            "warn"
        ]
        rows.append({"ratio": r, **drive_point(labels, drive_scores, thr)})
    return rows


def _fmt(x: Any, pct: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x:.1%}" if pct else f"{x:,}"


def sensitivity_markdown(tables: dict[str, list[dict]], rates: dict[str, float]) -> str:
    lines = [
        "# Cost-optimal operating points on validation (drive level)",
        "",
        "For each ratio r (how many unnecessary actions one prevented failure is worth), the",
        "threshold that maximises `r * caught - false_alarms` on the validation drive table, and",
        "its precision, recall and share of the fleet flagged. Written by",
        "`scripts/freeze_stack_scorer.py`; validation only. The optimum depends on the failure",
        "rate: on a period with half the failure rate (Q3 has about half of validation's), the",
        "same threshold flags more than is optimal there.",
        "",
    ]
    for arm, rows in tables.items():
        lines += [
            f"## {arm} (validation failure rate {rates[arm]:.3%})",
            "",
            "| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            thr = "-" if row["threshold"] is None else f"{row['threshold']:.4f}"
            lines.append(
                f"| {row['ratio']} | {thr} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
                f"{row['caught']:,} | {row['flagged']:,} | {_fmt(row.get('flagged_share'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def write_checksums(folder: Path) -> None:
    lines = []
    for path in sorted(
        p for p in folder.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256"
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(folder).as_posix()}")
    (folder / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--net", type=float, default=0.8)
    parser.add_argument(
        "--reference", type=Path, default=Path("data/audit/stack_q3_h60/experiment_two_stage.json")
    )
    parser.add_argument("--out", type=Path, default=Path("frozen_runs/stack_h60_seed0"))
    parser.add_argument(
        "--scores-out", type=Path, default=Path("data/tmp/stack_scores/scores_h60_seed0.parquet")
    )
    parser.add_argument(
        "--cascade-scores",
        type=Path,
        default=Path("data/tmp/lightgbm_scores_h60_h60_q1q2_q3.parquet"),
    )
    parser.add_argument("--name", default="stack_h60_seed0", help="suffix of the thresholds files")
    args = parser.parse_args(argv)

    configure_logging()
    apply_memory_limit_from_config()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a frozen scorer is never overwritten")
    started = dt.datetime.now(dt.UTC)
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    data = load_validation_data(args.work_dir)
    fitted = fit_arms(data, seed=args.seed, net=args.net)

    check = reproduction_check(fitted, args.reference, seed=args.seed, net=args.net)
    for c in check["checks"]:
        logger.info("freeze_check", **c)
    if not check["passed"]:
        failed = [c["check"] for c in check["checks"] if not c["ok"]]
        raise SystemExit(f"refit does not reproduce {args.reference}: {failed}; nothing written")

    # Validation scores of the three arms, in the score-table layout.
    scores = {
        arm: fitted["models"][arm].final_scores(data["x_val"], fitted["val_s1"]) for arm in ARMS
    }
    val_ids = data["val_ids"]
    table = pl.DataFrame(
        {
            "split": pl.Series(["validation"] * len(data["y_val"]), dtype=pl.String),
            "drive_id": val_ids["drive_id"],
            "date": val_ids["date"],
            "label": pl.Series(data["y_val"]).cast(pl.Int8),
            **{column: scores[arm] for arm, column in ARMS.items()},
        }
    )
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.scores_out, compression="zstd")

    save_stacked_two_stage(fitted["models"]["stack"], args.out)

    drive_ids, y = data["val_drives"], data["y_val"]
    tiers: dict[str, dict] = {}
    sensitivity: dict[str, list[dict]] = {}
    rates: dict[str, float] = {}
    for arm, column in ARMS.items():
        tiers[arm] = arm_tiers(drive_ids, y, scores[arm], model_config)
        sensitivity[arm] = cost_sensitivity(drive_ids, y, scores[arm])
        rates[arm] = tiers[arm]["failure_rate"]
        name = args.name if arm == "stack" else args.name.replace("stack", arm, 1)
        (args.out / f"agent_thresholds_{name}.json").write_text(
            json.dumps({"score_column": column, **tiers[arm]}, indent=2)
        )
    if args.cascade_scores.exists():
        cascade = (
            pl.scan_parquet(args.cascade_scores)
            .filter(pl.col("split") == "validation")
            .select("drive_id", "label", "score")
            .collect()
        )
        arm = "cascade (frozen LightGBM run, its validation rows)"
        c_ids, c_y, c_s = (
            cascade["drive_id"].to_numpy(),
            cascade["label"].to_numpy(),
            cascade["score"].to_numpy(),
        )
        sensitivity[arm] = cost_sensitivity(c_ids, c_y, c_s)
        rates[arm] = float(drive_level_table(c_ids, c_y, c_s)[0].mean())
    (args.out / "cost_tiers_validation.md").write_text(sensitivity_markdown(sensitivity, rates))

    report = {
        "started": started.isoformat(),
        "finished": dt.datetime.now(dt.UTC).isoformat(),
        "work_dir": args.work_dir.name,
        "horizon_days": data["horizon_days"],
        "seed": args.seed,
        "net": args.net,
        "settings": fitted["settings"],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "reproduction": check,
        "reports": {
            k: {kk: vv for kk, vv in v.items() if kk not in {"stage1"}}
            for k, v in fitted["reports"].items()
        },
        "scores_out": str(args.scores_out),
        "scores_rows": table.height,
        "tiers": tiers,
        "cost_sensitivity": sensitivity,
        "test_access": "none: x_test.npy is never opened",
    }
    (args.out / "freeze_report.json").write_text(json.dumps(report, indent=2, default=str))
    write_checksums(args.out)
    print(f"reproduction passed: {check['passed']} ({len(check['checks'])} checks)")
    for arm in ARMS:
        t = tiers[arm]
        print(
            f"{arm}: lift {t['action_tiers']}  cost {t['action_tiers_cost']}  "
            f"primary {t['threshold']:.4f}"
        )
    print(f"written to {args.out} and {args.scores_out}")


if __name__ == "__main__":
    main()
