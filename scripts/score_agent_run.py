"""Per-tier precision for an agent run that has already happened.

`pipelines/run_agent_fleet.py --score` reports one overall precision. On a
real run 94% of the actions were `cordon` - the cheapest and only reversible
active tier - so that single figure is dominated by it and says nothing
about whether `migrate` and `drain` are trustworthy. Which is the question
that decides whether a 38% false-action rate is a problem or a price.

Reads the saved report rather than re-running the agent, so an existing
run can be re-analysed for free:

    uv run python scripts/score_agent_run.py
    uv run python scripts/score_agent_run.py --report path/to/run.json

Cordons are split two ways. `determine_action_tier` returns CORDON when a
drive scores high enough for migrate or drain but fails the confidence
floor, the maturity gate or the staleness check - it still acts, just more
gently. Those are the drives whose evidence the system itself distrusts, so
their precision is the evidence for or against downgrading them to WARN
instead of acting at all.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml

from src.agent.backtest import load_labels, precision_by_tier

DEFAULT_REPORT = Path("data/audit/agent_fleet_run.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")


def _print(rows: list[dict], report: dict, horizon_days: int) -> None:
    window = report.get("window", ["?", "?"])
    print(f"\n== agent run {report.get('snapshot_id', '?')}")
    print(f"   window {window[0]} .. {window[1]}, {horizon_days}-day horizon")
    print(f"   model run {report.get('run_id', '?')}")
    print(
        f"   {report.get('drive_count', 0):,} drives, "
        f"{report.get('above_threshold_count', 0):,} above the lowest threshold\n"
    )

    print(f"{'tier':<22}{'drives':>9}{'labelled':>10}{'failed':>8}{'precision':>11}")
    for row in rows:
        precision = "n/a" if row["precision"] is None else f"{row['precision']:.1%}"
        print(
            f"{row['tier']:<22}{row['drives']:>9,}{row['with_labels']:>10,}"
            f"{row['caught']:>8,}{precision:>11}"
        )

    active = [r for r in rows if r["tier"].startswith(("cordon", "migrate", "drain"))]
    acted = sum(r["with_labels"] for r in active)
    caught = sum(r["caught"] for r in active)
    if acted:
        print(f"\n{'all active tiers':<22}{'':>9}{acted:>10,}{caught:>8,}{caught / acted:>10.1%}")
    print("\n  `cordon (downgraded)` scored migrate/drain but failed the confidence")
    print("  floor, maturity gate or staleness check. It still acted. If its")
    print("  precision is materially worse, that gate should return WARN instead.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.report.exists():
        raise SystemExit(f"{args.report} not found; run `make agent-fleet` first")
    report = json.loads(args.report.read_text())

    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    labels_path = Path(data_config["gold_dir"]) / "labels" / "part.parquet"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} not found; run `make build-labels` first")

    # The horizon the run was scored at, not the configured primary: a run
    # over the inference window is scored at 7 days (runbook 2.3).
    horizon_days = (
        args.horizon_days
        or report.get("scoring", {}).get("horizon_days")
        or model_config["primary_horizon_days"]
    )
    start, end = report["window"]
    labels = load_labels(
        labels_path,
        horizon_days=horizon_days,
        start_date=dt.date.fromisoformat(start),
        end_date=dt.date.fromisoformat(end),
    )
    if labels.is_empty():
        raise SystemExit(
            f"no observable {horizon_days}-day labels in {start}..{end} - the window "
            "ends after the data outlives the horizon (runbook 2.3)"
        )

    rows = precision_by_tier(
        report["decisions"], labels, action_thresholds=report.get("action_thresholds")
    )
    _print(rows, report, horizon_days)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"by_tier": rows}, indent=2, default=str))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
