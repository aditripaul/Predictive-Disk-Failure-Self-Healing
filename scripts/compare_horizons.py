"""Compare models trained for different label horizons on the same test drives.

    uv run python scripts/compare_horizons.py \\
        --scores 30=data/tmp/lightgbm_scores_h30_h30_q1q2_q3.parquet \\
                 44=data/tmp/lightgbm_scores_h44_h44_q1q2_q3.parquet \\
                 60=data/tmp/lightgbm_scores_h60_h60_q1q2_q3.parquet

Reads each run's saved test scores (`make train TRAIN_ARGS="--save-scores"`) and
the gold labels of every horizon, restricts all of them to the window every
horizon can observe, and scores each model against each label definition
(`src/models/horizon_comparison.py`). Writes `horizon_comparison.json`,
`tables.md` and figures to `--out`, and records one test-split look.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
import yaml

from src.models.access_log import _git_sha, record_test_access
from src.models.evaluation import drive_level_table
from src.models.horizon_comparison import (
    DEFAULT_BUDGETS,
    common_frame,
    compare_horizons,
    comparison_tables,
)
from src.outputs import refuse_overwrite
from src.reporting.evaluation_figures import (
    operating_point,
    plot_confusion_matrices,
    plot_precision_recall,
    pr_curve,
)
from src.reporting.plots import save_figure

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def _scores_arg(value: str) -> tuple[int, Path]:
    horizon, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected HORIZON=PATH, got {value!r}")
    return int(horizon), Path(path)


def write_figures(frame: pl.DataFrame, result: dict, out_dir: Path) -> list[Path]:
    """Per label definition: every model's drive precision-recall curve, and
    confusion matrices at the strictest matched false-alarm rate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_ids = frame["drive_id"].to_numpy()
    paths = []
    for label_h, block in result["labels"].items():
        y = frame[f"label_{label_h}"].to_numpy()
        curves, points = [], []
        for m, model in block["models"].items():
            labels, scores = drive_level_table(drive_ids, y, frame[f"score_{m}"].to_numpy())
            curves.append(pr_curve(f"{m}-day model", labels, scores))
            first = model["at_false_alarm_rate"][0]
            if first.get("reachable"):
                points.append(
                    operating_point(
                        f"{m}-day model, FAR {first['false_alarm_rate_budget']:.1%}",
                        labels,
                        scores,
                        first["threshold"],
                    )
                )
        title = f"Scored against {label_h}-day labels ({block['failing_drives']:,} failing drives)"
        paths.append(
            save_figure(
                plot_precision_recall(
                    curves,
                    points=[(p.name, p.precision, p.recall) for p in points],
                    base_rate=block["base_rate"],
                    title=title,
                ),
                out_dir / f"labels_{label_h}d_precision_recall.png",
            )
        )
        if points:
            paths.append(
                save_figure(
                    plot_confusion_matrices(points, title=title),
                    out_dir / f"labels_{label_h}d_confusion.png",
                )
            )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=_scores_arg, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/gold/labels/part.parquet"))
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-boot-auroc", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/audit/horizon_q3"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    splits = model_config.get("splits", {})
    scores_paths = dict(args.scores)
    horizons = sorted(scores_paths)
    start = args.start or dt.date.fromisoformat(str(splits["sealed_start"]))
    end = args.end or (
        dt.date.fromisoformat(str(splits["observation_end"])) - dt.timedelta(days=max(horizons))
    )
    refuse_overwrite(
        [args.out / "horizon_comparison.json", args.out / "tables.md"], overwrite=args.overwrite
    )

    scores = {
        h: pl.read_parquet(path, columns=["split", "drive_id", "date", "score"])
        .filter(pl.col("split") == "test")
        .drop("split")
        for h, path in scores_paths.items()
    }
    labels = (
        pl.scan_parquet(args.labels)
        .filter(pl.col("date").is_between(start, end) & pl.col("horizon_days").is_in(horizons))
        .select("drive_id", "date", "horizon_days", "label", "days_to_event")
        .collect()
    )
    frame = common_frame(scores, labels, start=start, end=end)
    del scores, labels

    record_test_access(
        variant="horizon_comparison",
        purpose="scripts/compare_horizons.py paired horizon comparison",
        extra={"horizons": horizons, "start": str(start), "end": str(end)},
    )
    top_k = list(model_config.get("evaluation", {}).get("top_k", []))
    result = compare_horizons(
        frame,
        horizons,
        budgets=tuple(args.budgets),
        top_k=top_k,
        n_boot=args.n_boot,
        n_boot_auroc=args.n_boot_auroc,
    )
    settings = {
        "start": str(start),
        "end": str(end),
        "horizons": horizons,
        "scores": {h: str(p) for h, p in scores_paths.items()},
        "rows": frame.height,
        "budgets": list(args.budgets),
        "top_k": top_k,
        "n_boot": args.n_boot,
        "n_boot_auroc": args.n_boot_auroc,
        "git_commit": _git_sha(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "horizon_comparison.json").write_text(
        json.dumps({"settings": settings, "result": result}, indent=2, default=float)
    )
    (args.out / "tables.md").write_text(comparison_tables(result, settings))
    figures = write_figures(frame, result, args.out / "figures")
    print((args.out / "tables.md").read_text())
    print(f"written to {args.out} ({len(figures)} figures, {frame.height:,} drive-days)")


if __name__ == "__main__":
    main()
