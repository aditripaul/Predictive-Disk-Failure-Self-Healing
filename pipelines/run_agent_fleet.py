"""Entry point for `make agent-fleet`.

Runs the MAPE-K loop against the real gold feature table and the trained
model instead of `src/agent/demo.py`'s two hardcoded drives
(`src/agent/fleet.py`). Execution stays simulated - there is no datacenter
to cordon a drive in - so what becomes real is the fleet, the scores, the
thresholds and every decision rule between them.

Two windows, answering different questions:

    # Inference: the window after test_end. Proves the plumbing; no labels,
    # so it cannot say whether the agent was right.
    uv run python pipelines/run_agent_fleet.py --start-date 2026-06-01

    # Inference window, scored. June has no observable 30-day label - the
    # data ends before the horizon does - but it does have 7-day labels up
    # to 2026-06-23, so this measures the agent WITHOUT touching the test
    # split. Report it as 7-day precision, never beside a 30-day figure.
    uv run python pipelines/run_agent_fleet.py \\
        --start-date 2026-06-01 --end-date 2026-06-23 \\
        --cycles 200 --score --score-horizon-days 7

    # Backtest: the test split. Has ground truth, and is the only way to
    # measure the AGENT rather than the model - between a score and an
    # action sit the guardrails, the confidence floor, hysteresis, the
    # cooldown and one-action-per-cycle, all of which can only remove
    # actions. Records a test-split access (src/models/access_log.py).
    uv run python pipelines/run_agent_fleet.py \\
        --start-date 2026-04-01 --end-date 2026-05-31 --cycles 50

One action is taken per cycle, by design (the plan node acts on the
highest-scoring drive), so `--cycles` bounds how many drives the agent can
reach. The snapshot is fixed for the whole run: this replays one point in
time repeatedly, not a moving clock.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from collections import Counter
from pathlib import Path

import yaml

from src.agent.backtest import ACTIVE_ACTIONS, load_labels, score_decisions
from src.agent.fleet import DEFAULT_MAX_CANDIDATES, build_fleet_dependencies
from src.agent.graph import compiled_agent
from src.config import AgentSettings
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

logger = get_logger(__name__)

DEFAULT_OUT = Path("data/audit/agent_fleet_run.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")


def _date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--run-id", default=None, help="MLflow run; default: latest for horizon")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--checkpoint-path",
        default=":memory:",
        help="LangGraph checkpoint store. Defaults to :memory: - this is a "
        "measurement tool, and a durable store would both grow by a "
        "checkpoint per node per cycle and let one run resume another's "
        "per-drive hysteresis and cooldown state. Pass the configured path "
        "to exercise durability deliberately.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Resume a specific LangGraph thread. Default: a fresh one per "
        "run, so two runs over the same window give the same answer.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score the decisions against the label table",
    )
    parser.add_argument(
        "--score-horizon-days",
        type=int,
        default=None,
        metavar="N",
        help="Horizon to score against (default: primary_horizon_days). The "
        "inference window after test_end has no observable 30-day label - the "
        "data ends before the horizon does - but DOES have 7- and 14-day "
        "labels, so `--score-horizon-days 7` scores it without touching the "
        "test split. Report such a figure as 7-day precision and never beside "
        "a 30-day one (runbook 1.7): the model is trained for 30 days, so a "
        "drive failing on day 20 is a 7-day false positive.",
    )
    return parser.parse_args(argv)


def advance_pool(
    candidates: list[dict],
    proposal: dict,
    selections: dict[str, int],
    *,
    max_selections: int,
) -> str | None:
    """Remove a decided drive from the candidate pool, returning why.

    The plan node takes `max(drives, key=p_fail)` from a snapshot that does
    not change on its own, so without this the same drive is chosen every
    cycle forever: it escalates once, enters cooldown, and then caps at WARN
    on identical evidence while every lower-scoring drive is never reached.
    The action ledger makes those repeats no-ops, so not even the executor
    sees them.

    A drive that was actually acted on leaves because that is what happens
    in a live loop - a cordoned drive is no longer a candidate next cycle.
    A drive that keeps being selected without ever being actioned is dropped
    after `max_selections` tries and recorded as blocked: in a live loop it
    would keep reappearing, which is realistic but stalls a fixed-snapshot
    replay behind one undrainable drive."""
    drive_id = proposal["drive_id"]
    acted = proposal["proposed_action"] in ACTIVE_ACTIONS
    selections[drive_id] = selections.get(drive_id, 0) + 1
    reason = None
    if acted:
        reason = "acted"
    elif selections[drive_id] >= max_selections:
        reason = "blocked"
    if reason is not None:
        candidates[:] = [d for d in candidates if d["drive_id"] != drive_id]
    return reason


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    apply_memory_limit_from_config()
    args = parse_args(argv)

    fleet = build_fleet_dependencies(
        start_date=_date(args.start_date),
        end_date=_date(args.end_date),
        run_id=args.run_id,
        max_candidates=args.max_candidates,
    )

    decisions: list[dict] = []
    selections: dict[str, int] = {}
    removed: dict[str, str] = {}
    # Enough tries for hysteresis to escalate a drive before we give up on
    # it, plus one; fewer and no drive could ever reach a destructive tier.
    max_selections = AgentSettings.load().hysteresis_cycles_required + 1
    # One thread for the whole run, so hysteresis counters and cooldown carry
    # across cycles the way they would in a live loop - but a FRESH one per
    # run. The thread id used to be derived from the snapshot alone, which is
    # deterministic, so re-running a window resumed the previous run's
    # cooldowns and advanced hysteresis counters and gave a different answer
    # from the same inputs.
    thread_id = args.thread_id or f"fleet-{fleet.snapshot_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    with compiled_agent(
        fleet.dependencies,
        checkpoint_path=args.checkpoint_path,
        plan_kwargs=fleet.plan_kwargs,
    ) as app:
        for cycle in range(args.cycles):
            result = app.invoke({}, config=config)
            proposal = result.get("proposed_action")
            if proposal is None:
                logger.info("agent_cycle_no_candidates", cycle=cycle)
                break
            decisions.append(
                {
                    "cycle": cycle,
                    "as_of": fleet.as_of_by_drive.get(proposal["drive_id"]),
                    "drive_id": proposal["drive_id"],
                    "proposed_action": proposal["proposed_action"],
                    "p_fail": proposal["p_fail"],
                    "control_arm": proposal.get("control_arm", False),
                    "withheld_action": proposal.get("withheld_action"),
                    "guardrail_passed": result["guardrail_result"]["passed"],
                    "human_review_required": result["human_review_required"],
                }
            )
            reason = advance_pool(
                fleet.candidates, proposal, selections, max_selections=max_selections
            )
            if reason is not None:
                removed[proposal["drive_id"]] = reason
            if not fleet.candidates:
                logger.info("agent_candidate_pool_exhausted", cycle=cycle)
                break

    report = {
        "snapshot_id": fleet.snapshot_id,
        "run_id": fleet.run_id,
        "window": [str(fleet.window[0]), str(fleet.window[1])],
        "thread_id": thread_id,
        "drive_count": fleet.drive_count,
        "candidate_count": fleet.candidate_count,
        "above_threshold_count": fleet.above_threshold_count,
        "action_thresholds": fleet.plan_kwargs["action_thresholds"],
        "cycles_run": len(decisions),
        "actions_by_tier": dict(Counter(d["proposed_action"] for d in decisions)),
        "control_arm_decisions": sum(d["control_arm"] for d in decisions),
        "sent_to_human_review": sum(d["human_review_required"] for d in decisions),
        "removed_from_pool": dict(sorted(removed.items())),
        "blocked_drives": sum(1 for r in removed.values() if r == "blocked"),
        "executed": fleet.executed_actions,
        "decisions": decisions,
    }
    if args.score:
        report["scoring"] = _score(fleet, decisions, horizon_days=args.score_horizon_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "agent_fleet_run_complete",
        path=str(args.out),
        drive_count=fleet.drive_count,
        candidate_count=fleet.candidate_count,
        cycles_run=len(decisions),
        actions_by_tier=report["actions_by_tier"],
    )


def _score(fleet, decisions: list[dict], *, horizon_days: int | None = None) -> dict:
    """Join the decisions to the label table and report agent-level numbers.

    Returns a `skipped` note rather than zeros when the window has no
    observable labels - the inference window after `test_end` has none by
    construction, and reporting 0.0 precision there would read as a result
    rather than an absence."""
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    labels_path = Path(data_config["gold_dir"]) / "labels" / "part.parquet"
    if not labels_path.exists():
        return {"skipped": f"{labels_path} not found; run `make build-labels` first"}

    start, end = fleet.window
    splits = model_config["splits"]
    start = start or dt.date.fromisoformat(splits["validation_end"]) + dt.timedelta(days=1)
    end = end or dt.date.fromisoformat(splits["test_end"])
    resolved_horizon = horizon_days or model_config["primary_horizon_days"]

    # A row is only scoreable if the data outlives its horizon. Snapshotting
    # past that boundary silently yields nothing to score, because
    # latest_row_per_drive keeps each drive's LAST row - the censored one.
    data_end = dt.date.fromisoformat(data_config["sources"]["backblaze"]["end_date"])
    last_scoreable = data_end - dt.timedelta(days=resolved_horizon)
    if end > last_scoreable:
        logger.warning(
            "scoring_window_extends_past_label_observability",
            window_end=str(end),
            last_scoreable_row=str(last_scoreable),
            horizon_days=resolved_horizon,
            hint=f"re-run with --end-date {last_scoreable}",
        )

    labels = load_labels(
        labels_path,
        horizon_days=resolved_horizon,
        start_date=start,
        end_date=end,
    )
    if labels.is_empty():
        return {
            "skipped": "no observable labels in this window - the inference window "
            "after test_end is censored by construction, so there is nothing to score"
        }
    scored = score_decisions(
        decisions,
        labels,
        candidate_count=fleet.candidate_count,
        above_threshold_count=fleet.above_threshold_count,
    )
    scored["horizon_days"] = resolved_horizon
    scored["window"] = [str(start), str(end)]
    logger.info("agent_backtest_scored", **{k: v for k, v in scored.items() if k != "attrition"})
    return scored


if __name__ == "__main__":
    main()
