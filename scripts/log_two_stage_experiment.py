"""Logs a finished `make experiment-model EXPERIMENT_ARGS="--two-stage ..."`
comparison to MLflow from its output directory: for a run made before the
experiment logged itself, or one run with --no-mlflow.

    uv run python scripts/log_two_stage_experiment.py data/audit/cascade_wide_net_h30 \
        --artifact data/audit/cascade_logs/cascade_wide_net_h30.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.experiment_model import log_two_stage_to_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, help="Directory with experiment_two_stage.json")
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Extra file to attach (repeatable), e.g. the console log.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Log again even if logged.")
    args = parser.parse_args()
    run_id = log_two_stage_to_mlflow(
        args.out_dir, extra_artifacts=tuple(args.artifact), overwrite=args.overwrite
    )
    print(f"Logged {args.out_dir} to MLflow run {run_id}")


if __name__ == "__main__":
    main()
