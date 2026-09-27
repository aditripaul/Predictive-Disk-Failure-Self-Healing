"""Entry point for `make score-fleet`.

Runs the latest MLflow-trained model against each drive's most recent
gold-feature row and produces typed, audit-ready `PredictionOutput`
records (plus an `ActionProposal` for any drive the action-tier policy
flags above MONITOR). `data_contracts.schemas.PredictionOutput`/
`ActionProposal` were previously declared but never used anywhere in real
code - see `src/models/serving.py` for why this offline batch-scoring
report, not the live MAPE-K agent loop, is where they genuinely apply.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import yaml

import mlflow
from src.logging_config import configure_logging, get_logger
from src.models.features import select_feature_columns
from src.models.serving import (
    join_feature_maturity,
    latest_row_per_drive,
    propose_actions,
    score_latest_drive_day,
)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
AGENT_CONFIG_PATH = Path("configs/agent.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    agent_config = yaml.safe_load(AGENT_CONFIG_PATH.read_text())

    gold_dir = Path(data_config["gold_dir"])
    silver_dir = Path(data_config["silver_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found; run `make build-features` first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found; run `make build-silver` first.")

    latest_features = latest_row_per_drive(pl.read_parquet(features_path))
    latest_features = join_feature_maturity(latest_features, pl.read_parquet(metadata_path))

    mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
    experiment_name = model_config["mlflow"]["experiment_name"]
    runs = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1
    )
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found for experiment {experiment_name!r}; run `make train` first."
        )
    run_id = runs.iloc[0]["run_id"]

    model_type = model_config["model"].get("type", "lightgbm")
    model = (
        mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        if model_type == "xgboost"
        else mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    )

    feature_columns = select_feature_columns(latest_features)
    as_of = dt.datetime.now(dt.UTC)
    predictions = score_latest_drive_day(
        latest_features,
        model,
        feature_columns,
        model_name=model_type,
        model_version=str(model_config["version"]),
        horizon_days=model_config["primary_horizon_days"],
        as_of=as_of,
    )
    proposals = propose_actions(
        predictions,
        action_thresholds=agent_config["action_thresholds"],
        min_confidence_for_destructive_action=agent_config["feature_confidence"][
            "min_confidence_for_destructive_action"
        ],
    )

    out_dir = Path(data_config["audit_dir"]) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{as_of.date().isoformat()}.json"
    out_path.write_text(
        json.dumps(
            {
                "predictions": [p.model_dump() for p in predictions],
                "action_proposals": [p.model_dump() for p in proposals],
            },
            indent=2,
            default=str,
        )
    )

    logger.info(
        "fleet_scored",
        drive_count=len(predictions),
        proposal_count=len(proposals),
        run_id=run_id,
        path=str(out_path),
    )


if __name__ == "__main__":
    main()
