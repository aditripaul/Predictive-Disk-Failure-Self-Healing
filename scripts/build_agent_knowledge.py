"""Build, resolve and check the agent's knowledge store (docs/agent_knowledge_base.md).

    uv run python scripts/build_agent_knowledge.py build     # fill from the frozen stack's scores
    uv run python scripts/build_agent_knowledge.py check     # self-checks -> knowledge_checks.json
    uv run python scripts/build_agent_knowledge.py coverage --as-of 2026-03-02
    uv run python scripts/build_agent_knowledge.py drift --reference 2026-01-01:2026-03-31 \\
        --current 2026-04-01:2026-05-01
    uv run python scripts/build_agent_knowledge.py resolve --as-of 2026-06-30  # + trust scores

Nothing dated on or after the sealed start (configs/model.yaml splits.sealed_start, the
first Q3 day) is read unless `--allow-test-outcomes` is given, and then the read is logged
in the test-access log. The store itself (`data/audit/agent_knowledge.sqlite`) is local and
rebuilt by `build`; the JSON reports go to `data/audit/agent_knowledge_h60/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.knowledge.build import backfill, scope_drive_ids  # noqa: E402
from src.agent.knowledge.checks import (  # noqa: E402
    check_idempotence,
    check_incremental_equals_batch,
    check_outcomes_vs_labels,
    check_replay,
    episode_summary,
)
from src.agent.knowledge.store import KnowledgeBase  # noqa: E402
from src.logging_config import configure_logging, get_logger  # noqa: E402
from src.models.access_log import record_test_access  # noqa: E402
from src.resource_limits import apply_memory_limit_from_config  # noqa: E402

MODEL_CONFIG_PATH = Path("configs/model.yaml")
DEFAULT_KB = Path("data/audit/agent_knowledge.sqlite")
DEFAULT_OUT = Path("data/audit/agent_knowledge_h60")
logger = get_logger(__name__)


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _range(value: str) -> tuple[dt.date, dt.date]:
    a, b = value.split(":")
    return _date(a), _date(b)


def sealed_start() -> dt.date:
    return _date(yaml.safe_load(MODEL_CONFIG_PATH.read_text())["splits"]["sealed_start"])


def guard_test_dates(last: dt.date, *, allow: bool, purpose: str, extra: dict[str, Any]) -> None:
    """Refuses a read reaching the sealed (test) period unless allowed; logs it if allowed."""
    if last < sealed_start():
        return
    if not allow:
        raise SystemExit(
            f"{last} is on or after the sealed start {sealed_start()} (test data); "
            "pass --allow-test-outcomes to read it, which is logged as a test access"
        )
    record_test_access(variant="agent_knowledge", purpose=purpose, extra=extra)


def scorer_settings(run_dir: Path, thresholds_file: str, tier_key: str) -> dict[str, Any]:
    """Run id, flag thresholds (warn, cordon from the tier set; primary) and candidate
    threshold from a frozen scorer's folder."""
    tiers_meta = json.loads((run_dir / thresholds_file).read_text())
    stack = json.loads((run_dir / "stack.json").read_text())
    tiers = tiers_meta[tier_key]
    return {
        "run_id": f"{run_dir.name}:{tiers_meta['score_column']}:{tier_key}",
        "thresholds": {
            "warn": float(tiers["warn"]),
            "cordon": float(tiers["cordon"]),
            "primary": float(tiers_meta["threshold"]),
        },
        "candidate_threshold": float(stack["candidate_threshold"]),
    }


def _write(out_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def cmd_build(args: argparse.Namespace) -> None:
    presence_end = args.presence_end or args.end
    guard_test_dates(
        presence_end,
        allow=args.allow_test_outcomes,
        purpose="knowledge store: presence and failures in the test period",
        extra={"presence_end": str(presence_end)},
    )
    if args.overwrite and args.knowledge.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(f"{args.knowledge}{suffix}").unlink(missing_ok=True)
    settings = scorer_settings(args.run_dir, args.thresholds_file, args.tier_key)
    kb = KnowledgeBase(args.knowledge, horizon_days=args.horizon_days, **settings)
    summary = backfill(
        kb,
        scores_path=args.scores,
        drive_day_path=args.drive_day,
        split=args.split or None,
        start=args.start,
        end=args.end,
        presence_end=presence_end,
        column=args.score_column,
        chunk_days=args.chunk_days,
        on_chunk=lambda c: logger.info(
            "knowledge_chunk", **{k: v for k, v in c.items() if k != "resolved"}
        ),
    )
    summary["settings"] = settings
    path = _write(args.out, "build.json", summary)
    print(f"built {args.knowledge}: {summary['row_counts']}  ({path})")


def cmd_resolve(args: argparse.Namespace) -> None:
    guard_test_dates(
        args.as_of,
        allow=args.allow_test_outcomes,
        purpose="knowledge store: outcomes resolved in the test period",
        extra={"as_of": str(args.as_of)},
    )
    kb = KnowledgeBase(args.knowledge)
    written = kb.resolve_outcomes(args.as_of)
    trust = kb.resolve_trust_scores()
    print(f"outcomes as of {args.as_of}: {written}; final trust scores: {trust}")


def cmd_check(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.knowledge)
    meta = kb.metadata()
    end = _date(meta["built_from_end"])
    checks = [
        check_incremental_equals_batch(
            kb, drive_day_path=args.drive_day, start=args.slice_start, days=14
        ),
        check_replay(kb),
        check_outcomes_vs_labels(kb, labels_path=args.labels),
        check_idempotence(kb, drive_day_path=args.drive_day, day=end),
    ]
    pit = Path(args.pit_report)
    if pit.exists():
        cross = json.loads(pit.read_text()).get("cross_check") or []
        checks.append(
            {
                "check": "h_score_agreement",
                "source": str(pit),
                "cross_check": cross,
                "passed": bool(cross) and all(c["over_1e-6"] == 0 for c in cross),
            }
        )
    report = {
        "knowledge": str(args.knowledge),
        "metadata": {k: v for k, v in meta.items() if k not in {"replay_sample", "replay_hashes"}},
        "row_counts": kb.row_counts(),
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "episodes": {
            day: episode_summary(kb, _date(day))
            for day in (meta["built_from_end"], meta.get("presence_end", meta["built_from_end"]))
        },
    }
    path = _write(args.out, "knowledge_checks.json", report)
    for c in checks:
        print(f"  {c['check']:<28} {'PASS' if c['passed'] else 'FAIL'}")
    print(f"all passed: {report['passed']}  ({path})")


def cmd_coverage(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.knowledge)
    meta = kb.metadata()
    scores_path, split = Path(meta["source_scores"]), meta.get("source_split")
    scan = pl.scan_parquet(scores_path).filter(pl.col("date") == args.as_of)
    if split is not None:
        scan = scan.filter(pl.col("split") == split)
    fleet = scan.select("drive_id").unique().collect()["drive_id"]
    report = {
        "fleet": f"drives in {scores_path.name} on {args.as_of} (split {split})",
        "drives_in_scope": scope_drive_ids(scores_path, split=split).len(),
        **kb.coverage(as_of=args.as_of, fleet_drive_ids=fleet.to_list()),
    }
    path = _write(args.out, f"coverage_{args.as_of}.json", report)
    print(json.dumps(report, indent=2), f"\n({path})")


def cmd_drift(args: argparse.Namespace) -> None:
    kb = KnowledgeBase(args.knowledge)
    report = kb.drift_report(reference=args.reference, current=args.current)
    path = _write(args.out, f"drift_{args.current[0]}_{args.current[1]}.json", report)
    print(json.dumps(report, indent=2, default=str), f"\n({path})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--drive-day", type=Path, default=Path("data/silver/drive_day/part.parquet")
    )
    parser.add_argument("--allow-test-outcomes", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument(
        "--scores", type=Path, default=Path("data/tmp/stack_scores/scores_h60_seed0.parquet")
    )
    b.add_argument("--score-column", default="score")
    b.add_argument("--split", default="validation")
    b.add_argument("--start", type=_date, default=_date("2026-01-01"))
    b.add_argument("--end", type=_date, default=_date("2026-05-01"))
    b.add_argument("--presence-end", type=_date, default=_date("2026-06-30"))
    b.add_argument("--run-dir", type=Path, default=Path("frozen_runs/stack_h60_seed0"))
    b.add_argument("--thresholds-file", default="agent_thresholds_stack_h60_seed0.json")
    b.add_argument("--tier-key", default="action_tiers")
    b.add_argument("--horizon-days", type=int, default=60)
    b.add_argument("--chunk-days", type=int, default=7)
    b.add_argument("--overwrite", action="store_true")

    r = sub.add_parser("resolve")
    r.add_argument("--as-of", type=_date, required=True)

    c = sub.add_parser("check")
    c.add_argument("--labels", type=Path, default=Path("data/gold/labels/part.parquet"))
    c.add_argument("--slice-start", type=_date, default=_date("2026-02-01"))
    c.add_argument("--pit-report", default="data/audit/agent_pit_validation_h60/stack.json")

    v = sub.add_parser("coverage")
    v.add_argument("--as-of", type=_date, required=True)

    d = sub.add_parser("drift")
    d.add_argument("--reference", type=_range, required=True)
    d.add_argument("--current", type=_range, required=True)

    args = parser.parse_args(argv)
    configure_logging()
    apply_memory_limit_from_config()
    {
        "build": cmd_build,
        "resolve": cmd_resolve,
        "check": cmd_check,
        "coverage": cmd_coverage,
        "drift": cmd_drift,
    }[args.command](args)


if __name__ == "__main__":
    main()
