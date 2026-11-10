"""Run the agent loop once per model of a `train-merged` run, each with its own tiers.

    make agent-models REPORT=<merged_evaluation_report.json> SCORES=<merged_scores.parquet> \\
        OUT_DIR=<dir>

For every model in the report it writes `<OUT_DIR>/thresholds_<model>.json` (that model's
score column and the action tiers tuned for it on validation) and runs
`pipelines/run_agent_fleet.py --scores-path ... --thresholds-path ... --score` in a
subprocess, so each run's snapshot memory is released. Output per model:
`<OUT_DIR>/<model>.json` (the run report) and `<OUT_DIR>/<model>.log`.

The window defaults to the test split (validation_end + 1 to test_end in configs/model.yaml;
61 days with the current splits). The run replays ONE snapshot of that window, each drive's
last row, with no clock: it looks ahead and is not point-in-time (runbook Section 4.7).
It refuses to start if any output exists; --skip-existing resumes a partial run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.labels.splits import first_date_after_validation
from src.outputs import refuse_overwrite

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def write_thresholds(report: dict, model: str, path: Path) -> dict:
    """The thresholds file `run_agent_fleet --thresholds-path` reads for `model`."""
    result = report["models"][model]
    meta = {
        "score_column": model,
        "action_tiers": result["action_tiers"],
        "threshold": result["threshold"]["threshold"],
    }
    path.write_text(json.dumps(meta, indent=2))
    return meta


def default_window() -> tuple[dt.date, dt.date]:
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    start = first_date_after_validation(model_config)
    assert start is not None, "configs/model.yaml splits.validation_end is not set"
    return start, dt.date.fromisoformat(str(model_config["splits"]["test_end"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="merged_evaluation_report*.json")
    parser.add_argument("scores", type=Path, help="merged_scores_*.parquet")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--models", nargs="*", help="default: every model in the report")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--cycles", type=int, default=0, help="0 = until the pool is exhausted")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    models = args.models or list(report["models"])
    unknown = [m for m in models if m not in report["models"]]
    if unknown:
        raise SystemExit(f"not in {args.report.name}: {unknown}")
    start, end = default_window()
    start, end = args.start_date or str(start), args.end_date or str(end)

    outputs = {m: args.out_dir / f"{m}.json" for m in models}
    if args.skip_existing:
        done = [m for m, p in outputs.items() if p.exists()]
        models = [m for m in models if m not in done]
        print(f"skipping finished: {done}")
    else:
        refuse_overwrite(outputs.values(), overwrite=args.overwrite)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        thresholds = args.out_dir / f"thresholds_{model}.json"
        write_thresholds(report, model, thresholds)
        command = [
            sys.executable,
            "pipelines/run_agent_fleet.py",
            "--scores-path", str(args.scores),
            "--thresholds-path", str(thresholds),
            "--start-date", start,
            "--end-date", end,
            "--cycles", str(args.cycles),
            "--score",
            "--out", str(outputs[model]),
        ]  # fmt: skip
        if args.overwrite:
            command.append("--overwrite")
        print(f"[{model}] {start}..{end}", flush=True)
        t0 = time.monotonic()
        with (args.out_dir / f"{model}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[{model}] exit {code} in {time.monotonic() - t0:.0f}s", flush=True)
        if code:
            raise SystemExit(f"{model} failed (see {args.out_dir / f'{model}.log'})")


if __name__ == "__main__":
    main()
