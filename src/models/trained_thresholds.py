"""Which per-tier thresholds a deployment should act on.

`make train` tunes a threshold per action tier from the lift targets in
`configs/model.yaml` and records them with the model as `action_tiers.json`.
Two code paths then decide what to do with a drive, and they must agree:

  * `pipelines/score_fleet.py` - batch scoring, loads the artifact.
  * `src/agent/nodes.py::make_plan_node` - the live MAPE-K loop.

They did not agree. The agent took the hand-picked cutoffs in
`configs/agent.yaml` (0.30 / 0.60 / 0.80 / 0.90) unconditionally, so the same
drive could be cordoned by one path and drained by the other, and every
precision figure in the runbook described only the batch path. Those
hand-picked numbers also read as probabilities while being applied to raw
class-weighted scores, which are not probabilities
(docs/model_status_and_runbook.md section 1.3).

This module is the single resolution both paths call, so the disagreement
cannot come back by editing one of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.logging_config import get_logger
from src.models.threshold import resolve_action_thresholds

logger = get_logger(__name__)

MODEL_CONFIG_PATH = Path("configs/model.yaml")
AGENT_CONFIG_PATH = Path("configs/agent.yaml")


def latest_run_id_for_horizon(experiment_name: str, horizon_days: int) -> str:
    """The most recent run trained for `horizon_days`.

    Filtered on the horizon so a trial run - `make train
    TRAIN_ARGS="--horizon-days 60"` - cannot silently become the model behind
    30-day predictions. Raises if there is none; callers that must keep
    running without a model use `resolve_trained_action_thresholds`."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"params.horizon_days = '{horizon_days}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs with horizon_days={horizon_days} found for experiment "
            f"{experiment_name!r}; run `make train` first."
        )
    return str(runs.iloc[0]["run_id"])


def action_thresholds_for_run(run_id: str, fallback: dict[str, float]) -> dict[str, float]:
    """The run's tuned per-tier thresholds, or `fallback` if it has none.

    Runs trained before `action_tiers.json` existed have no artifact; they
    fall back to the configured cutoffs and say so in the log rather than
    failing, because an older model is still scoreable."""
    import mlflow

    try:
        tier_thresholds = mlflow.artifacts.load_dict(f"runs:/{run_id}/action_tiers.json")
    except Exception:
        logger.warning("action_tiers_missing_using_agent_config", run_id=run_id)
        tier_thresholds = None
    return resolve_action_thresholds(tier_thresholds, fallback)


def resolve_trained_action_thresholds(
    *,
    model_config: dict[str, Any] | None = None,
    agent_config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, float], str | None]:
    """`(thresholds, run_id)` for a live deployment, never raising.

    Returns the configured cutoffs with `run_id=None` when no trained model
    can be resolved - no MLflow store, no run for the horizon, MLflow not
    reachable. The agent must still make a decision in that case, but the
    caller can see from the returned run id which thresholds it got, and the
    reason is logged."""
    model_config = model_config or yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    agent_config = agent_config or yaml.safe_load(AGENT_CONFIG_PATH.read_text())
    fallback: dict[str, float] = dict(agent_config["action_thresholds"])

    try:
        import mlflow

        mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
        resolved_run_id = run_id or latest_run_id_for_horizon(
            model_config["mlflow"]["experiment_name"], model_config["primary_horizon_days"]
        )
    except Exception as exc:
        logger.warning(
            "trained_action_thresholds_unavailable_using_agent_config",
            reason=str(exc),
            thresholds=fallback,
        )
        return fallback, None

    thresholds = action_thresholds_for_run(resolved_run_id, fallback)
    logger.info("trained_action_thresholds_resolved", run_id=resolved_run_id, thresholds=thresholds)
    return thresholds, resolved_run_id
