"""Draw the LSTM-against-LightGBM comparison from the reports already written.

Needs the runs and the comparison of runbook sections 5.9 to 5.11:

    make train-sequence                                   # s0
    make train-sequence SEQUENCE_ARGS="--seed 1 --tag s1"  # s1..s3
    make train-sequence SEQUENCE_ARGS="--train-only --tag t1 ..."  # tuning, optional
    uv run --extra torch python scripts/compare_sequence_vs_tree.py

then:

    uv run python scripts/plot_sequence_comparison.py

Writes to data/audit/plots/: sequence_training_curves.png,
sequence_recall_at_budgets.png, sequence_catch_overlap.png and
sequence_blend_gain.png. Reads reports only, so it re-scores nothing and logs no
test-split access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.reporting.plots import save_figure
from src.reporting.sequence_figures import (
    plot_blend_gain,
    plot_catch_overlap,
    plot_recall_at_budgets,
    plot_training_curves,
)

DATA_CONFIG_PATH = Path("configs/data.yaml")
BUDGETS = ("0.0001", "0.0005", "0.001")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    reports = Path(data_config["audit_dir"]) / "data_quality_reports"
    out_dir = args.out_dir or Path(data_config["audit_dir"]) / "plots"

    comparison_path = reports / "sequence_vs_tree_seeds.json"
    if not comparison_path.exists():
        raise SystemExit(f"{comparison_path} not found; run scripts/compare_sequence_vs_tree.py")
    comparison = json.loads(comparison_path.read_text())

    histories = {}
    for path in sorted(reports.glob("sequence_evaluation_report*.json")):
        if ".smoke" not in path.name:
            histories[path.stem] = json.loads(path.read_text())["history"]
    for path in sorted(reports.glob("sequence_tuning_*.json")):
        histories[path.stem] = json.loads(path.read_text())["history"]
    save_figure(plot_training_curves(histories), out_dir / "sequence_training_curves.png")

    per_run = {r["lstm"]: r for r in comparison["per_run_and_ensemble"]}
    ensemble = per_run["lstm_ens"]
    tree = comparison["lightgbm"]
    rows = [
        {"name": "LightGBM", "kind": "tree", **tree},
        {
            "name": f"LSTM, {len(comparison['runs'])}-seed ensemble, alone",
            "kind": "lstm",
            **ensemble["alone"],
        },
        {"name": "Blend of the two", "kind": "blend", **ensemble["blend"]},
    ]
    save_figure(plot_recall_at_budgets(rows, BUDGETS), out_dir / "sequence_recall_at_budgets.png")
    save_figure(
        plot_catch_overlap(comparison["ensemble_overlap_with_lightgbm"]),
        out_dir / "sequence_catch_overlap.png",
    )
    gains = [
        {
            "name": "Ensemble" if name == "lstm_ens" else f"Seed {name.removeprefix('lstm_s')}",
            "gain": run["blend_minus_lightgbm_recall"],
            "ci": run["blend_minus_lightgbm_recall_95ci"],
        }
        for name, run in per_run.items()
    ]
    save_figure(plot_blend_gain(gains, BUDGETS), out_dir / "sequence_blend_gain.png")
    print(f"figures written to {out_dir}/sequence_*.png")


if __name__ == "__main__":
    main()
