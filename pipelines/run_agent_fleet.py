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
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.agent.backtest import ACTIVE_ACTIONS, load_labels, score_decisions
from src.agent.fleet import DEFAULT_MAX_CANDIDATES, build_fleet_dependencies
from src.agent.graph import compiled_agent
from src.config import AgentSettings
from src.logging_config import configure_logging, get_logger
from src.outputs import refuse_overwrite
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
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Agent cycles; each decides ONE drive. 0 = run until the candidate "
        "pool is exhausted (bounded by candidates x max selections per drive).",
    )
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        metavar="N",
        help="Log progress every N cycles (0 = off).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --out if it exists. Without it the run refuses to start, so a run on "
        "merged scores (--scores-path) needs its own --out to be compared with another.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="LangGraph checkpoint store. Default: a fresh temporary FILE, "
        "deleted on exit. Not :memory: - the saver retains every checkpoint, "
        "so a long run holds its whole history in RAM (10,000 cycles x a "
        "checkpoint per node exhausted 20 GB). Not the configured path "
        "either, because that would let one run resume another's per-drive "
        "hysteresis and cooldown state. Pass a path to override.",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=None,
        help="Act on precomputed scores (merged approach: <scratch>/merged/"
        "merged_scores_h30.parquet from `make train-merged`) instead of an MLflow "
        "model. The column and action tiers come from agent_thresholds_h30.json beside it.",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=None,
        help="With --scores-path: a thresholds file (score_column, action_tiers) other than "
        "agent_thresholds_*.json beside the scores, e.g. one model's tiers from "
        "`scripts/run_agent_models.py`.",
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
        "--knowledge",
        choices=["none", "memory", "sqlite"],
        default=None,
        help="Attach the knowledge base (docs/agent_knowledge_base.md): candidates get a "
        "knowledge context and decisions are recorded in it. Default: configs/agent.yaml "
        "knowledge.backend. It changes no decision (no rule is enabled).",
    )
    parser.add_argument("--knowledge-path", type=Path, default=None)
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


def open_knowledge(args: argparse.Namespace) -> Any:
    """The knowledge base the run attaches, or None (`--knowledge` / configs/agent.yaml).
    `sqlite` opens an existing store (`scripts/build_agent_knowledge.py build`), which
    carries its own scorer and thresholds; `memory` starts an empty one for this run."""
    settings = AgentSettings.load()
    backend = args.knowledge or settings.knowledge_backend
    if backend == "none":
        return None
    from src.agent.knowledge import KnowledgeBase

    if backend == "sqlite":
        path = args.knowledge_path or Path(settings.knowledge_path)
        if not path.exists():
            raise SystemExit(
                f"{path} not found; build it with scripts/build_agent_knowledge.py build"
            )
        return KnowledgeBase(path)
    thresholds = settings.action_thresholds
    return KnowledgeBase.in_memory(
        run_id="agent_fleet",
        thresholds={
            "warn": thresholds["warn"],
            "cordon": thresholds["cordon"],
            "primary": thresholds["migrate"],
        },
        candidate_threshold=min(thresholds.values()),
        horizon_days=yaml.safe_load(Path("configs/model.yaml").read_text())["primary_horizon_days"],
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    apply_memory_limit_from_config()
    args = parse_args(argv)
    refuse_overwrite([args.out], overwrite=args.overwrite)

    knowledge = open_knowledge(args)
    fleet = build_fleet_dependencies(
        start_date=_date(args.start_date),
        end_date=_date(args.end_date),
        run_id=args.run_id,
        max_candidates=args.max_candidates,
        scores_path=args.scores_path,
        thresholds_path=args.thresholds_path,
        knowledge=knowledge,
    )

    decisions: list[dict] = []
    selections: dict[str, int] = {}
    removed: dict[str, str] = {}
    # Carried across cycles and pruned to the live pool each time. Left
    # alone it grows one entry per drive ever decided and - because
    # apply_hysteresis rebuilds it as {**state, drive: ...} - the whole dict
    # is copied every cycle and every copy is checkpointed. Pruning holds it
    # at roughly one entry: a drive is decided twice, WARN then an action,
    # and then leaves the pool.
    risk_state: dict[str, dict] = {}
    # Reported so a regression in the pruning is visible in the output
    # rather than only as memory growth.
    max_risk_entries = 0
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
    with (
        tempfile.TemporaryDirectory(prefix="agent_fleet_ckpt_") as ckpt_dir,
        compiled_agent(
            fleet.dependencies,
            checkpoint_path=args.checkpoint_path or str(Path(ckpt_dir) / "checkpoints.sqlite"),
            plan_kwargs=fleet.plan_kwargs,
        ) as app,
    ):
        # --cycles 0: until the pool empties. Every cycle either removes a
        # drive or counts a selection toward its max_selections, so this
        # bound is never reached before the pool is exhausted.
        n_cycles = args.cycles or len(fleet.candidates) * max_selections
        started = time.monotonic()
        decided: set[str] = set()
        for cycle in range(n_cycles):
            result = app.invoke({"drive_risk_state": risk_state}, config=config)
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
                    **(
                        {"knowledge_context": proposal.get("knowledge_context")}
                        if knowledge is not None
                        else {}
                    ),
                }
            )
            decided.add(proposal["drive_id"])
            reason = advance_pool(
                fleet.candidates, proposal, selections, max_selections=max_selections
            )
            if reason is not None:
                removed[proposal["drive_id"]] = reason
            if args.progress_every and (cycle + 1) % args.progress_every == 0:
                elapsed = time.monotonic() - started
                per_drive = elapsed / len(decided)
                remaining = len(fleet.candidates)
                logger.info(
                    "agent_fleet_progress",
                    cycle=cycle + 1,
                    elapsed_s=round(elapsed, 1),
                    drives_decided=len(decided),
                    pool_remaining=remaining,
                    sec_per_decided_drive=round(per_drive, 3),
                    projected_remaining_s=round(remaining * per_drive, 1),
                    projected_total_s=round(elapsed + remaining * per_drive, 1),
                )
            # Prune to drives still in the pool: a drive that has left can
            # never be selected again, so its counters are dead weight that
            # every subsequent checkpoint would carry.
            alive = {d["drive_id"] for d in fleet.candidates}
            risk_state = {
                drive_id: entry
                for drive_id, entry in result.get("drive_risk_state", {}).items()
                if drive_id in alive
            }
            max_risk_entries = max(max_risk_entries, len(risk_state))
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
        "peak_risk_state_entries": max_risk_entries,
        "actions_by_tier": dict(Counter(d["proposed_action"] for d in decisions)),
        "control_arm_decisions": sum(d["control_arm"] for d in decisions),
        "sent_to_human_review": sum(d["human_review_required"] for d in decisions),
        "removed_from_pool": dict(sorted(removed.items())),
        "blocked_drives": sum(1 for r in removed.values() if r == "blocked"),
        "executed": fleet.executed_actions,
        "decisions": decisions,
    }
    if knowledge is not None:
        report["knowledge"] = {
            "path": str(knowledge.path) if knowledge.path else ":memory:",
            "run_id": knowledge.run_id,
            "decisions_recorded": knowledge.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[
                0
            ],
            "with_context": sum(1 for d in decisions if d.get("knowledge_context") is not None),
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

    if end < data_end:
        # Not point-in-time. Two look-aheads (runbook 4.7): feature_maturity is
        # assigned from the dataset's LAST date, so the migrate/drain gate
        # reads whether a drive survived the snapshot; and "latest row" for a
        # drive that left mid-window is its final row before it disappeared.
        # The figures are real arithmetic on a biased setup - quote none of
        # them as the agent's performance. scripts/check_agent_lookahead.py
        # measures how much each contributes.
        logger.warning(
            "scoring_a_past_window_is_not_point_in_time",
            window_end=str(end),
            data_end=str(data_end),
            see="docs/model_status_and_runbook.md section 4.7",
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
        action_thresholds=fleet.plan_kwargs.get("action_thresholds"),
    )
    scored["horizon_days"] = resolved_horizon
    scored["window"] = [str(start), str(end)]
    logger.info(
        "agent_backtest_scored",
        **{k: v for k, v in scored.items() if k not in {"attrition", "by_tier"}},
    )
    for row in scored["by_tier"]:
        logger.info("agent_tier_precision", **row)
    return scored


if __name__ == "__main__":
    main()
