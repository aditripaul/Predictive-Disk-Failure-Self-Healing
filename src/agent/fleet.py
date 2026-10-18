"""Real fleet dependencies for the MAPE-K loop.

`src/agent/demo.py` wires the loop to two hardcoded drives with literal
`p_fail` values. This module wires it to the actual gold feature table and
the trained model instead, so the fleet, the scores, the thresholds and
every decision rule between them are real. The executor and validator stay
simulated: there is no datacenter to cordon a drive in, and pretending
otherwise would be worse than logging.

Two windows matter, and they answer different questions:

  * **After `test_end`** (June 2026 in the shipped config) - the inference
    window. No metric scores it, so this proves the plumbing without
    touching evaluation data. There are no labels, so it cannot tell you
    whether the agent was right.
  * **The test split** (April-May 2026) - a backtest with ground truth, and
    the only way to measure the *agent* rather than the model. Between a
    score and an action sit the guardrails, the confidence floor,
    hysteresis, the cooldown and one-action-per-cycle; all of them can only
    remove actions, and nobody has measured by how much. Building a snapshot
    in that window records a test-split access (`src/models/access_log.py`).

Two constraints shape the design:

  * **State is checkpointed.** `src/agent/state.py` keeps the state lean
    deliberately, and LangGraph writes all of it to SQLite every cycle. The
    fleet cannot go in there, so the snapshot is reduced to the top
    `max_candidates` drives at or above the lowest action threshold. The plan
    node only ever takes `max(drives, key=p_fail)`, so this changes nothing
    about which drive is chosen - but it does mean `flagged_drive_ids` is
    the candidate list, not every drive above a threshold.
  * **`fleet_state_provider` is called twice per cycle**, by `monitor` and
    again by `analyze`. The snapshot is therefore built once, eagerly, and
    returned from memory. A long-running deployment would need a refresh
    policy; there isn't one, so rebuild the factory to advance the clock.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.agent.deps import ActionLedger, AgentDependencies, InMemoryActionLedger
from src.guardrails.adapter import build_guardrail_evaluator
from src.logging_config import get_logger
from src.models.access_log import record_test_access
from src.models.features import select_feature_columns
from src.models.serving import join_feature_maturity, latest_row_per_drive
from src.models.trained_thresholds import (
    action_thresholds_for_run,
    latest_run_id_for_horizon,
)
from src.models.training import predict_proba_positive
from src.models.two_stage import load_two_stage

logger = get_logger(__name__)

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
AGENT_CONFIG_PATH = Path("configs/agent.yaml")

#: How many drives reach checkpointed state. Only the highest-scoring one is
#: ever acted on, so this bounds the checkpoint rather than the decision.
DEFAULT_MAX_CANDIDATES = 100
DEFAULT_JOIN_CHUNK_ROWS = 2_000_000


@dataclass(frozen=True)
class FleetAgent:
    """Everything an entry point needs to run the loop against real data."""

    dependencies: AgentDependencies
    plan_kwargs: dict[str, Any]
    snapshot_id: str
    run_id: str | None
    drive_count: int
    candidate_count: int
    #: Drives at or above the lowest threshold BEFORE max_candidates
    #: truncated. The gap is drives the agent was never shown, which is not
    #: the same as drives it declined to act on.
    above_threshold_count: int
    window: tuple[dt.date | None, dt.date | None]
    #: The SAME list object `fleet_state_provider` returns. A multi-cycle
    #: caller must advance it - see `pipelines/run_agent_fleet.py`. The plan
    #: node takes `max(drives, key=p_fail)` from whatever this holds, so an
    #: unchanged list means the same drive is chosen every cycle forever.
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: drive_id -> the date of the row it was scored from, so a backtest can
    #: join decisions to labels on (drive_id, date, horizon_days). Kept here
    #: rather than in the proposal because AgentState is checkpointed.
    as_of_by_drive: dict[str, dt.date] = field(default_factory=dict)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)


def _configs(
    data_config: dict | None, model_config: dict | None, agent_config: dict | None
) -> tuple[dict, dict, dict]:
    return (
        data_config or yaml.safe_load(DATA_CONFIG_PATH.read_text()),
        model_config or yaml.safe_load(MODEL_CONFIG_PATH.read_text()),
        agent_config or yaml.safe_load(AGENT_CONFIG_PATH.read_text()),
    )


def overlaps_test_split(start: dt.date | None, end: dt.date | None, splits: dict[str, Any]) -> bool:
    """Whether `[start, end]` touches the test split.

    Test runs from the day after `validation_end` through `test_end`
    (`src/labels/splits.py`, `pipelines/train_model.py`). An open-ended
    window counts as overlapping, because it cannot be shown not to."""
    test_start = dt.date.fromisoformat(splits["validation_end"]) + dt.timedelta(days=1)
    test_end = dt.date.fromisoformat(splits["test_end"])
    return (start is None or start <= test_end) and (end is None or end >= test_start)


def load_fleet_snapshot(
    features_path: Path,
    metadata_path: Path,
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    chunk_rows: int = DEFAULT_JOIN_CHUNK_ROWS,
) -> pl.DataFrame:
    """One row per drive - its most recent gold-feature row inside the
    window - with `feature_maturity` joined in from Silver.

    Scanned rather than read: the gold table is ~196 columns over every
    drive-day, and only one row per drive survives
    (`src/models/serving.py::latest_row_per_drive`). Filtering the window
    before that keeps the date predicate in the scan, so a narrow window
    reads proportionally less."""
    lf = pl.scan_parquet(features_path)
    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)
    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)
    latest = latest_row_per_drive(lf, chunk_rows=chunk_rows)
    return join_feature_maturity(latest, pl.read_parquet(metadata_path))


def to_candidate_drives(
    scored: pl.DataFrame,
    *,
    action_thresholds: dict[str, float],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[list[dict[str, Any]], int]:
    """The lean per-drive dicts the agent state carries, highest risk first.

    Only drives at or above the lowest action threshold are candidates -
    below it the policy returns MONITOR anyway, so carrying them would cost
    memory for no decision. `stale_telemetry` defaults to False when the
    column is absent rather than being omitted, because the action-tier
    policy treats a missing flag as "not stale" and the guardrail engine
    expects the key to exist.

    Returns `(candidates, above_threshold_count)`. The second value is the
    count BEFORE `max_candidates` truncates, so a scorer can tell drives the
    agent rejected from drives it was never shown - without it, a capped run
    reports every drive as reached and the attrition reads as zero."""
    floor = min(action_thresholds.values()) if action_thresholds else 0.0
    candidates = (
        scored.filter(pl.col("p_fail") >= floor).sort("p_fail", descending=True)
        if "p_fail" in scored.columns
        else scored
    )
    above_threshold = candidates.height
    drives: list[dict[str, Any]] = []
    for row in candidates.head(max_candidates).to_dicts():
        drives.append(
            {
                "drive_id": row["drive_id"],
                "p_fail": float(row["p_fail"]),
                "feature_confidence": float(row["feature_confidence"]),
                "feature_maturity": row["feature_maturity"],
                "stale_telemetry": bool(row.get("stale_telemetry_flag") or False),
            }
        )
    return drives, above_threshold


def _snapshot_id(window: tuple[dt.date | None, dt.date | None], drive_count: int) -> str:
    """Stable for the same window and fleet size, so a resumed thread is
    recognisably the same snapshot rather than a new random id each run."""
    key = f"{window[0]}|{window[1]}|{drive_count}"
    return f"fleet-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def load_scoring_model(run_id: str, model_type: str) -> Any:
    """The run's model, with its second stage when it has one.

    A run trained with a two-stage cascade must be scored with it: its
    recorded thresholds were tuned on the combined score, so applying them
    to Stage 1 alone would be applying them to a different distribution."""
    import mlflow

    model = (
        mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        if model_type == "xgboost"
        else mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    )
    if model_type != "lightgbm":
        return model
    two_stage = load_two_stage(run_id, model)
    if two_stage is not None:
        logger.info("two_stage_model_loaded", run_id=run_id)
        return two_stage
    return model


def score_snapshot(snapshot: pl.DataFrame, model: Any) -> pl.DataFrame:
    """`snapshot` with a `p_fail` column. Nulls are filled with 0.0, the same
    convention batch scoring uses (`src/models/serving.py`), so a drive is
    never dropped for a missing feature."""
    feature_columns = select_feature_columns(snapshot)
    x = snapshot.select(feature_columns).fill_null(0.0).to_numpy()
    return snapshot.with_columns(
        pl.Series("p_fail", predict_proba_positive(model, x)).cast(pl.Float64)
    )


def _recording_executor(executed: list[dict[str, Any]]):
    """Simulated execution: there is no datacenter to cordon a drive in, so
    the action is recorded and logged rather than performed. `executed`
    accumulates them so a backtest can score what the agent would have
    done."""

    def execute(action: dict[str, Any]) -> dict[str, Any]:
        result = {
            "action_id": action["action_id"],
            "drive_id": action["drive_id"],
            "proposed_action": action["proposed_action"],
            "success": True,
            "simulated": True,
        }
        executed.append({**result, "p_fail": action.get("p_fail")})
        logger.info(
            "fleet_action_simulated",
            drive_id=action["drive_id"],
            proposed_action=action["proposed_action"],
        )
        return result

    def validate(_execution_result: dict[str, Any]) -> dict[str, Any]:
        # Nothing was actually done, so nothing can have been broken. Stated
        # rather than computed, so no one mistakes this for a real check.
        return {"data_integrity_ok": True, "service_continuity_ok": True, "quorum_ok": True}

    return execute, validate


def build_fleet_dependencies(
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    run_id: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    action_ledger: ActionLedger | None = None,
    data_config: dict | None = None,
    model_config: dict | None = None,
    agent_config: dict | None = None,
) -> FleetAgent:
    """Bind the MAPE-K loop to the real gold feature table and trained model.

    The snapshot is built once, here, because `fleet_state_provider` is
    called twice per cycle and a real scan is not free. To advance the clock,
    build another one."""
    data_config, model_config, agent_config = _configs(data_config, model_config, agent_config)

    gold_dir = Path(data_config["gold_dir"])
    silver_dir = Path(data_config["silver_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    metadata_path = silver_dir / "drive_metadata" / "part.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found; run `make build-features` first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found; run `make build-silver` first.")

    import mlflow

    mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
    resolved_run_id = run_id or latest_run_id_for_horizon(
        model_config["mlflow"]["experiment_name"], model_config["primary_horizon_days"]
    )
    action_thresholds = action_thresholds_for_run(
        resolved_run_id, agent_config["action_thresholds"]
    )

    if overlaps_test_split(start_date, end_date, model_config["splits"]):
        record_test_access(
            variant="agent_fleet",
            purpose="MAPE-K loop scored a fleet snapshot overlapping the test split",
            extra={
                "run_id": resolved_run_id,
                "start_date": str(start_date),
                "end_date": str(end_date),
            },
        )

    snapshot = load_fleet_snapshot(
        features_path,
        metadata_path,
        start_date=start_date,
        end_date=end_date,
        chunk_rows=data_config.get("resource_limits", {}).get(
            "training_join_chunk_rows", DEFAULT_JOIN_CHUNK_ROWS
        ),
    )
    scored = score_snapshot(
        snapshot, load_scoring_model(resolved_run_id, model_config["model"].get("type", "lightgbm"))
    )
    drives, above_threshold = to_candidate_drives(
        scored, action_thresholds=action_thresholds, max_candidates=max_candidates
    )
    snapshot_id = _snapshot_id((start_date, end_date), scored.height)
    logger.info(
        "fleet_snapshot_built",
        snapshot_id=snapshot_id,
        run_id=resolved_run_id,
        drive_count=scored.height,
        candidate_count=len(drives),
        above_threshold_count=above_threshold,
        start_date=str(start_date),
        end_date=str(end_date),
    )

    as_of_by_drive = {
        row["drive_id"]: row["date"]
        for row in scored.select(["drive_id", "date"]).to_dicts()
        if "date" in scored.columns
    }

    def fleet_state_provider() -> dict[str, Any]:
        """The highest-risk candidate still in the pool, and only it.

        The pool itself stays here, in ordinary Python memory. Returning all
        of it would put every candidate into `AgentState`, which LangGraph
        writes to SQLite on every node of every cycle - at a pool size large
        enough to measure recall that is gigabytes of checkpoint for a list
        the plan node reduces to one drive anyway (`state.py` keeps the state
        lean deliberately). One drive in, same decision out, and the pool can
        be as large as the measurement needs."""
        if not drives:
            return {"fleet_snapshot_id": snapshot_id, "drives": []}
        return {
            "fleet_snapshot_id": snapshot_id,
            "drives": [max(drives, key=lambda d: d["p_fail"])],
        }

    executed: list[dict[str, Any]] = []
    execute, validate = _recording_executor(executed)

    dependencies = AgentDependencies(
        fleet_state_provider=fleet_state_provider,
        predictor=lambda snap: {"drives": snap["drives"]},
        guardrail_evaluator=build_guardrail_evaluator(),
        executor=execute,
        validator=validate,
        action_ledger=action_ledger or InMemoryActionLedger(),
    )
    return FleetAgent(
        dependencies=dependencies,
        plan_kwargs={"action_thresholds": action_thresholds},
        snapshot_id=snapshot_id,
        run_id=resolved_run_id,
        drive_count=scored.height,
        candidate_count=len(drives),
        above_threshold_count=above_threshold,
        window=(start_date, end_date),
        candidates=drives,
        as_of_by_drive=as_of_by_drive,
        executed_actions=executed,
    )
