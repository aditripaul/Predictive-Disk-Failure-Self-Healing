"""Replay a recall-first alerting policy day by day over a train-merged score table.

    make recall-policy PACK_DIR=<packed scores dir> OUT_DIR=<dir>

Builds the recall-tuned blend (default weights 0.2 / 0 / 0.5 / 0.3 over LightGBM, LSTM, AFT,
TCN), sets the WARN threshold on VALIDATION so that `--recall-target` of failing drives were
warned at least `--lead-min` days before failing, keeps the precision-tuned cordon / migrate /
drain thresholds from the existing lift targets, and replays validation and then test in date
order (src/agent/recall_replay.py). The test replay is one recorded look at test.

This does not improve the model's ranking: it picks a looser operating point and measures what
it costs in alerts. The live loop's confidence, maturity and staleness gates and topology
guardrails are not modelled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.agent.recall_replay import (
    failure_dates,
    policy_metrics,
    warn_threshold_for_recall,
    window_max,
)
from src.models.access_log import record_test_access
from src.models.blend_weights import DriveRows, recall_at_far
from src.models.ensemble import rank_blend, validation_rank
from src.models.evaluation import drive_level_table
from src.models.score_pack import MANIFEST, rebuild
from src.models.threshold import precision_targets_from_lift, tune_action_tiers
from src.outputs import refuse_overwrite

MODEL_CONFIG_PATH = Path("configs/model.yaml")
DATA_CONFIG_PATH = Path("configs/data.yaml")
#: Weight vectors whose validation recall at low false-alarm rates is printed for comparison.
COMPARE = {"equal": None, "precision-first best": [0.3, 0.0, 0.6, 0.1]}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def table_rows(results: dict) -> list[str]:
    lines = [
        "| Split | k | Recall (lead >= L) | Recall (lead >= 7) | Warn precision | Median lead (d) "
        "| Caught then escalated | Cordon precision | Migrate precision | Drain precision "
        "| Carried in | New warns/day | Watchlist mean / peak | Late hits |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, by_k in results.items():
        for k, m in by_k.items():
            late = m["late_hits"]
            lead = "n/a" if m["median_lead_days"] is None else f"{m['median_lead_days']:.1f}"
            lines.append(
                f"| {split} | {k} | {_pct(m['recall'])} | {_pct(m['recall_lead_7'])} "
                f"| {_pct(m['warn_precision'])} | {lead} | {_pct(m['caught_then_escalated'])} "
                f"| {_pct(m['tiers']['cordon']['precision'])} "
                f"| {_pct(m['tiers']['migrate']['precision'])} "
                f"| {_pct(m['tiers']['drain']['precision'])} "
                f"| {m['carried_in_at_start']} | {m['new_warns_per_day']:.0f} "
                f"| {m['watchlist_mean']:.0f} / {m['watchlist_peak']} "
                f"| {late['late_hits']} of {late['judged_warned_drives']} |"
            )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--blend", default="lgbm_lstm_aft_tcn")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.0, 0.5, 0.3])
    parser.add_argument("--recall-target", type=float, default=0.6)
    parser.add_argument("--lead-min", type=int, default=1)
    parser.add_argument("--hysteresis", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="replay validation only: no test look is recorded and nothing is written",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary_path = args.out_dir / "summary.json"
    refuse_overwrite([summary_path, args.out_dir / "table.md"], overwrite=args.overwrite)

    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    horizon = model_config["primary_horizon_days"]
    data_end = dt.date.fromisoformat(str(data_config["sources"]["backblaze"]["end_date"]))
    members = json.loads((args.pack_dir / MANIFEST).read_text())["ensembles"][args.blend]
    weights = np.asarray(args.weights, dtype=np.float64)
    if len(weights) != len(members) or (weights < 0).any() or weights.sum() <= 0:
        raise SystemExit(f"--weights must be {len(members)} non-negative numbers for {members}")
    weights = weights / weights.sum()

    t0 = time.perf_counter()
    table = rebuild(args.pack_dir)
    splits = {name: table.filter(pl.col("split") == name) for name in ("validation", "test")}
    del table
    val = splits["validation"]
    reference = {m: val[m].to_numpy() for m in members}
    equal_val = rank_blend(reference, reference, members)
    if not np.allclose(equal_val, val[args.blend].to_numpy(), rtol=0, atol=1e-12):
        raise SystemExit(f"equal weights do not reproduce the stored {args.blend} column")
    loaded = time.perf_counter() - t0
    print(f"loaded and checked in {loaded:.0f}s; weights {weights.round(3).tolist()}")

    scores = {
        name: rank_blend(reference, {m: frame[m].to_numpy() for m in members}, members, weights)
        for name, frame in splits.items()
    }
    daily = {
        name: pl.DataFrame(
            {"drive_id": frame["drive_id"], "date": frame["date"], "s": scores[name]}
        ).sort("drive_id", "date")
        for name, frame in splits.items()
    }
    windows = {name: (f["date"].min(), f["date"].max()) for name, f in splits.items()}

    # Does the chosen blend give up ranking at the low-false-alarm end? Validation only.
    ranks = np.column_stack([validation_rank(reference[m], reference[m]) for m in members])
    rows = DriveRows.build(val["drive_id"].to_numpy(), val["label"].to_numpy())
    arranged = rows.arrange(ranks)
    compare = {
        "chosen": weights.tolist(),
        **{k: v or [1 / len(members)] * len(members) for k, v in COMPARE.items()},
    }
    far = {
        name: [
            round(x, 4)
            for x in recall_at_far(rows.labels, rows.drive_scores(arranged @ np.asarray(w)))
        ]
        for name, w in compare.items()
    }
    for name, values in far.items():
        print(f"  validation recall at false-alarm 1e-4/2e-4/5e-4, {name}: {values}")

    failures = failure_dates(
        Path(data_config["gold_dir"]) / "labels" / "part.parquet",
        horizon_days=horizon,
        since=windows["validation"][0],
    )
    start, end = windows["validation"]
    in_window = failures.filter(
        pl.col("failure_date").is_between(
            start + dt.timedelta(days=1), end + dt.timedelta(days=horizon)
        )
    )
    wm = window_max(daily["validation"], in_window, horizon_days=horizon, lead=args.lead_min)
    warn = warn_threshold_for_recall(wm["score"].to_numpy(), args.recall_target)

    drive_labels, _ = drive_level_table(
        val["drive_id"].to_numpy(), val["label"].to_numpy(), scores["validation"]
    )
    targets = precision_targets_from_lift(
        model_config["threshold"]["action_tier_lift"], float(drive_labels.mean())
    )
    tuned = tune_action_tiers(
        val["drive_id"].to_numpy(),
        val["label"].to_numpy(),
        scores["validation"],
        precision_targets=targets,
    )
    tiers = {
        t: float("inf") if tuned.get(t) is None else float(tuned[t])
        for t in ("cordon", "migrate", "drain")
    }
    print(
        f"warn threshold {warn:.6f} (validation recall target {args.recall_target:.0%}, "
        f"lead >= {args.lead_min}, {wm.height} failing drives judged); tiers {tiers}"
    )

    def replay(name: str) -> dict:
        return {
            k: policy_metrics(
                daily[name], failures, warn=warn, tiers=tiers, k=k, horizon_days=horizon,
                window=windows[name], data_end=data_end, lead_min=args.lead_min,
            )
            for k in args.hysteresis
        }  # fmt: skip

    results = {"validation": replay("validation")}
    if args.validation_only:
        print("\n" + "\n".join(table_rows(results)))
        return
    record_test_access(
        variant="recall_policy_replay",
        purpose="daily replay of the recall-first policy, test window",
        extra={"weights": weights.tolist(), "recall_target": args.recall_target,
               "hysteresis": args.hysteresis, "lead_min": args.lead_min},
    )  # fmt: skip
    results["test"] = replay("test")

    lines = [*table_rows(results)]
    text = "\n".join(lines)
    print("\n" + text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table.md").write_text(text + "\n")
    summary = {
        "members": members,
        "weights": weights.tolist(),
        "recall_target": args.recall_target,
        "lead_min": args.lead_min,
        "warn_threshold": warn,
        "tiers": tiers,
        "validation_recall_at_far": far,
        "windows": {k: [str(a), str(b)] for k, (a, b) in windows.items()},
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
