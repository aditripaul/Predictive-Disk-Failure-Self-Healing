"""Print the markdown tables for a `train-merged` run (src/reporting/merged_tables.py).

    make merged-tables REPORT=<merged_evaluation_report.json> [AGENT_DIR=<dir>]

Always prints the per-model table. With --agent-dir (the output of
`scripts/run_agent_models.py`) it adds the table of what the agent loop did with each
model's scores. --status adds runbook 1.1.3's test and validation status tables
(caught, missed, false alarms); --recall-policy (a `scripts/run_recall_policy.py`
summary.json) its recall-first tier table; --cascade (a `make experiment-model
--two-stage` experiment_two_stage.json) its wide-net cascade table. --out also writes
the tables to a file (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.outputs import refuse_overwrite
from src.reporting.merged_tables import (
    agent_table,
    cascade_table,
    model_table,
    recall_policy_table,
    status_table,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="merged_evaluation_report*.json")
    parser.add_argument("--agent-dir", type=Path, help="run_agent_models.py output directory")
    parser.add_argument("--status", action="store_true", help="add runbook 1.1.3 tables")
    parser.add_argument("--recall-policy", type=Path, help="run_recall_policy summary.json")
    parser.add_argument("--cascade", type=Path, help="experiment_two_stage.json")
    parser.add_argument(
        "--cascade-levels",
        type=float,
        nargs="+",
        default=[0.10, 0.60],
        metavar="R",
        help="With --cascade: recall levels to tabulate (default 0.10 0.60; the Q3 runbook "
        "tables use 0.10 0.35 0.60).",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out:
        refuse_overwrite([args.out], overwrite=args.overwrite)

    report = json.loads(args.report.read_text())
    parts = ["## Models\n", model_table(report)]
    if args.agent_dir:
        runs = {}
        for name in report["models"]:
            path = args.agent_dir / f"{name}.json"
            runs[name] = json.loads(path.read_text()) if path.exists() else None
        parts += [
            "\n\n## Agent loop on each model\n",
            agent_table(runs, ensembles=report.get("ensembles", {})),
        ]
    if args.status:
        parts += [
            "\n\n## Status, test\n",
            status_table(report, "test"),
            "\n\n## Status, validation\n",
            status_table(report, "validation"),
        ]
    if args.recall_policy:
        summary = json.loads(args.recall_policy.read_text())
        parts += ["\n\n## Recall-first policy, test (k = 1)\n", recall_policy_table(summary)]
    if args.cascade:
        production = yaml.safe_load(Path("configs/model.yaml").read_text())["model"]
        net = production.get("two_stage", {}).get("candidate_recall")
        parts += [
            "\n\n## Cascade, wide Stage 1 nets (test, means over seeds)\n",
            cascade_table(
                json.loads(args.cascade.read_text()),
                levels=tuple(args.cascade_levels),
                production_net=net,
            ),
        ]
    text = "\n".join(parts) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
