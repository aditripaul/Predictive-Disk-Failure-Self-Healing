"""Single-shot evaluation of a frozen training run on a split no model choice may
look at: the sealed final quarter (`split=sealed`) or the SMART-Z vendor holdout
(`split=external_smartz`).

Nothing is re-chosen here. Thresholds, the calibrator, the feature list and the
two-stage model all come from the run's `frozen_spec.json` artifact, exactly as
training wrote them. The result file is written once per (split, run); a second
call refuses, because a second look at the same split is another comparison.

    uv run python pipelines/evaluate_frozen.py --run-id <mlflow run id> --split sealed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import mlflow
from pipelines.train_model import _build_feature_arrays, _row_group_parts, _write_id_columns
from src.models.access_log import record_test_access
from src.models.calibration import IsotonicCalibrator
from src.models.evaluation import compute_calibration, drive_level_metrics
from src.models.training import predict_proba_positive
from src.models.two_stage import TwoStageModel, load_two_stage
from src.models.uncertainty import drive_level_bootstrap_ci

ALLOWED_SPLITS = ("sealed", "external_smartz")
RESULTS_DIR = Path("data/audit/frozen_evaluations")


def score_frozen(
    *,
    spec: dict[str, Any],
    model: Any,
    two_stage: TwoStageModel | None,
    x: np.ndarray,
    y: np.ndarray,
    drive_ids: np.ndarray,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Every number the frozen run promises, on one split, with no thresholds
    or calibration fitted here. Pure: the caller supplies the arrays."""
    if len(y) == 0:
        raise ValueError("the split has no labeled rows")
    if x.shape[0] != len(y) or len(drive_ids) != len(y):
        raise ValueError("features, labels and drive ids must be row-aligned")

    scores = predict_proba_positive(model, x)
    if two_stage is not None:
        scores = two_stage.final_scores(x, scores)
    threshold = float(spec["drive_threshold"])
    calibrator = IsotonicCalibrator.from_dict(spec["calibrator"])

    tiers: dict[str, Any] = {}
    for tier, tier_threshold in spec["action_tier_thresholds"].items():
        tiers[tier] = (
            None
            if tier_threshold is None
            else {
                "threshold": float(tier_threshold),
                "drive": drive_level_metrics(drive_ids, y, scores, float(tier_threshold)),
            }
        )

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "evaluated_failure_rate": float(y.mean()),
        "drive_threshold": threshold,
        "drive": drive_level_metrics(drive_ids, y, scores, threshold),
        "drive_bootstrap_ci": drive_level_bootstrap_ci(
            drive_ids, y, scores, threshold, n_boot=n_boot
        ),
        "action_tiers": tiers,
        "calibrated_calibration": compute_calibration(y, calibrator.transform(scores)),
    }


def _load_model(run_id: str, model_type: str) -> Any:
    uri = f"runs:/{run_id}/model"
    if model_type == "xgboost":
        return mlflow.xgboost.load_model(uri)
    return mlflow.lightgbm.load_model(uri)


def _latest_frame() -> Path:
    frames = sorted(Path("data/tmp").glob("train_model_frame_*/frame.parquet"))
    if not frames:
        raise SystemExit("no assembled frame under data/tmp; run make train first")
    return frames[-1]


def evaluate(run_id: str, split: str, frame_path: Path, n_boot: int) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise SystemExit(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
    out_path = RESULTS_DIR / f"{split}__{run_id}.json"
    if out_path.exists():
        raise SystemExit(
            f"{out_path} exists: the {split} split was already evaluated with run "
            f"{run_id}. This evaluation is single-shot."
        )

    spec = mlflow.artifacts.load_dict(f"runs:/{run_id}/frozen_spec.json")
    model = _load_model(run_id, spec["model_type"])
    two_stage = load_two_stage(run_id, model) if spec["two_stage"] else None

    feature_columns: list[str] = spec["feature_columns"]
    id_columns = ["drive_id"]
    select_columns = [*feature_columns, "label", *id_columns]
    work = frame_path.parent / f"_frozen_{split}"
    if work.exists():
        shutil.rmtree(work)
    parts_dir = work / "parts"
    parts_dir.mkdir(parents=True)
    try:
        parts = _row_group_parts(
            frame_path,
            split_name=split,
            select_columns=select_columns,
            read_columns=sorted({*select_columns, "split"}),
            tmp_dir=parts_dir,
        )
        if not parts:
            raise SystemExit(f"the frame has no rows with split={split!r}")
        x_path = work / "x.npy"
        _, y = _build_feature_arrays(parts, feature_columns, out_path=x_path)
        ids_path = work / "ids.parquet"
        _write_id_columns(parts, id_columns, ids_path)
        x = np.load(x_path, mmap_mode="r")
        drive_ids = pl.read_parquet(ids_path)["drive_id"].to_numpy()
        result = score_frozen(
            spec=spec,
            model=model,
            two_stage=two_stage,
            x=x,
            y=np.asarray(y),
            drive_ids=drive_ids,
            n_boot=n_boot,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Record the look before writing its result, so no result exists unlogged.
    record_test_access(
        variant=f"frozen:{run_id}",
        purpose=f"frozen evaluation, split={split}",
        extra={"split": split},
    )
    result = {
        "split": split,
        "run_id": run_id,
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
        "horizon_days": spec["horizon_days"],
        **result,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-id", required=True, help="MLflow run id of the frozen model")
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--frame", type=Path, default=None, help="assembled frame.parquet")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    frame = args.frame or _latest_frame()
    result = evaluate(args.run_id, args.split, frame, args.n_boot)
    print(json.dumps({k: result[k] for k in ("split", "rows", "positives")}, indent=2))
    print(json.dumps(result["drive"], indent=2, default=float), file=sys.stderr)


if __name__ == "__main__":
    main()
