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
from collections import Counter
from pathlib import Path

from src.agent.fleet import DEFAULT_MAX_CANDIDATES, build_fleet_dependencies
from src.agent.graph import compiled_agent
from src.config import AgentSettings
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

logger = get_logger(__name__)

DEFAULT_OUT = Path("data/audit/agent_fleet_run.json")


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
    return parser.parse_args(argv)


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

    decisions = []
    # One thread for the whole run, so hysteresis counters and cooldown
    # carry across cycles the way they would in a live loop.
    config = {"configurable": {"thread_id": f"fleet-{fleet.snapshot_id}"}}
    with compiled_agent(
        fleet.dependencies,
        checkpoint_path=AgentSettings.load().checkpoint_path,
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
                    "drive_id": proposal["drive_id"],
                    "proposed_action": proposal["proposed_action"],
                    "p_fail": proposal["p_fail"],
                    "control_arm": proposal.get("control_arm", False),
                    "withheld_action": proposal.get("withheld_action"),
                    "guardrail_passed": result["guardrail_result"]["passed"],
                    "human_review_required": result["human_review_required"],
                }
            )

    report = {
        "snapshot_id": fleet.snapshot_id,
        "run_id": fleet.run_id,
        "window": [str(fleet.window[0]), str(fleet.window[1])],
        "drive_count": fleet.drive_count,
        "candidate_count": fleet.candidate_count,
        "action_thresholds": fleet.plan_kwargs["action_thresholds"],
        "cycles_run": len(decisions),
        "actions_by_tier": dict(Counter(d["proposed_action"] for d in decisions)),
        "control_arm_decisions": sum(d["control_arm"] for d in decisions),
        "sent_to_human_review": sum(d["human_review_required"] for d in decisions),
        "executed": fleet.executed_actions,
        "decisions": decisions,
    }
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


if __name__ == "__main__":
    main()
