"""Print the markdown tables for a `train-merged` run (src/reporting/merged_tables.py).

    make merged-tables REPORT=<merged_evaluation_report.json> [AGENT_DIR=<dir>]

Always prints the per-model table. With --agent-dir (the output of
`scripts/run_agent_models.py`) it adds the table of what the agent loop did with each
model's scores. --out also writes the tables to a file (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.outputs import refuse_overwrite
from src.reporting.merged_tables import agent_table, model_table

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="merged_evaluation_report*.json")
    parser.add_argument("--agent-dir", type=Path, help="run_agent_models.py output directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out:
        refuse_overwrite([args.out], overwrite=args.overwrite)

    report = json.loads(args.report.read_text())
    agent_ensemble = yaml.safe_load(MODEL_CONFIG_PATH.read_text())["merged_models"][
        "agent_ensemble"
    ]
    parts = ["## Models\n", model_table(report, agent_ensemble=agent_ensemble)]
    if args.agent_dir:
        runs = {}
        for name in report["models"]:
            path = args.agent_dir / f"{name}.json"
            runs[name] = json.loads(path.read_text()) if path.exists() else None
        parts += [
            "\n\n## Agent loop on each model\n",
            agent_table(runs, ensembles=report.get("ensembles", {}), agent_ensemble=agent_ensemble),
        ]
    text = "\n".join(parts) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
