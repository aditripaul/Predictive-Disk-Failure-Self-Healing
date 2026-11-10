"""Entry point for `make agent-pit`.

Measures the flagging policy the way it would run: at fixed dates, on the
drives reporting that day, judged on what happens next
(`src/agent/point_in_time.py`). This replaces the first agent backtest, which
looked ahead in two ways and reported 61.8% precision where the fleet as of a
single date gives 17.9% (runbook 4.7).

    uv run python pipelines/run_point_in_time.py
    uv run python pipelines/run_point_in_time.py --step-days 3 --bootstrap 2000

Snapshots default to the test window, weekly, anchored on the last date whose
30-day label is observable. They are TEST-split data, so the run is recorded
in the test-access log (`src/models/access_log.py`).

Reported, per tier: precision and recall with cluster-bootstrap intervals, the
false-alarm rate and likelihood ratio, and the base rate that precision lives
at. Then the one-look against two-look comparison: does flagging only drives
that score above threshold at both T-lag and T beat flagging on T alone, at a
matched false-alarm rate?

One cycle's memory is a single date's fleet (~0.7 GB at 356k drives x 246
features) plus its scores; dates are processed one at a time and only
(drive_id, date, p_fail, label) is kept.

Options for scorers that are not MLflow runs and for validation-only baselines
(docs/agent_knowledge_base.md, Phase 0):

    --drives-from-split validation   score only the validation holdout's drives
    --outcome-cutoff 2026-05-01      refuse a date whose horizon ends after it
    --observations-dir DIR           one parquet per date, written as it is scored;
                                     dates already on disk are read, not re-scored
    --scores-path / --score-column   p_fail from a score table instead of a model
    --model-dir DIR                  a frozen stacked scorer (load_stacked_two_stage)
    --thresholds-path FILE           tiers from a JSON file (`action_tiers` /
                                     `action_tiers_cost` via --tier-key, or a flat dict)
    --cross-check-dates D1,D2        score these dates with the model AND the table and
                                     report the largest difference
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.agent.fleet import load_scoring_model, overlaps_test_split
from src.agent.point_in_time import (
    CONTEXT_COLUMNS,
    DEFAULT_FAR_BUDGETS,
    TIERS,
    bootstrap_tier_ci,
    compare_looks,
    last_observable_date,
    pair_observations,
    read_observations_partition,
    select_snapshot_dates,
    snapshot_observations,
    split_drive_ids,
    table_scores_for_day,
    tier_counts,
    tier_metrics,
    write_observations_partition,
)
from src.labels.splits import first_date_after_validation
from src.logging_config import configure_logging, get_logger
from src.models.access_log import record_test_access
from src.models.trained_thresholds import action_thresholds_for_run, latest_run_id_for_horizon
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
AGENT_CONFIG_PATH = Path("configs/agent.yaml")
DEFAULT_OUT = Path("data/audit/agent_point_in_time.json")
DEFAULT_OBSERVATIONS = Path("data/audit/agent_point_in_time_observations.parquet")

logger = get_logger(__name__)


def _date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument(
        "--lag-days",
        type=int,
        default=None,
        help="Gap between the two looks (default: --step-days, i.e. the "
        "previous snapshot). Must be a multiple of --step-days.",
    )
    parser.add_argument("--run-id", default=None, help="MLflow run; default: latest for horizon")
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--observations-out",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
        help="Where to save every scored drive-snapshot with its context columns, "
        "so scripts/analyze_false_positives.py can profile them without "
        "re-scoring. Pass an empty string to skip.",
    )
    parser.add_argument("--drives-from-split", default=None, metavar="SPLIT")
    parser.add_argument("--outcome-cutoff", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--observations-dir", type=Path, default=None)
    parser.add_argument("--scores-path", type=Path, default=None)
    parser.add_argument(
        "--scores-split", default="validation", help="split filter on the score table; '' for none"
    )
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--thresholds-path", type=Path, default=None)
    parser.add_argument("--tier-key", default="action_tiers")
    parser.add_argument("--cross-check-dates", default="", metavar="D1,D2")
    parser.add_argument("--label", default=None, help="name of the scorer in the report")
    return parser.parse_args(argv)


def load_thresholds_file(
    path: Path, *, tier_key: str, fallback: dict[str, float]
) -> dict[str, float]:
    """Tier thresholds from a scorer's JSON (`action_tiers` or another `tier_key`) or
    from a flat `{warn, cordon, migrate, drain}` file such as a frozen run's
    `action_tiers.json`; an unreachable tier (None) never fires."""
    from src.models.threshold import resolve_action_thresholds

    meta = json.loads(path.read_text())
    if tier_key in meta:
        tiers = meta[tier_key]
    elif set(TIERS) <= set(meta):
        tiers = {t: meta[t] for t in TIERS}
    else:
        raise SystemExit(f"{path} has neither {tier_key!r} nor the tier keys {list(TIERS)}")
    return resolve_action_thresholds(tiers, fallback)


def _fmt_ci(value: float, interval: tuple[float, float]) -> str:
    return f"{value:6.1%} [{interval[0]:4.0%},{interval[1]:4.0%}]"


def print_report(report: dict[str, Any]) -> None:
    print(f"\n== point-in-time, {report['horizon_days']}-day horizon, model run {report['run_id']}")
    print(
        f"   {len(report['dates'])} snapshots {report['dates'][0]} .. {report['dates'][-1]}, "
        f"every {report['step_days']} days"
    )
    pooled = report["pooled"]
    print(
        f"   {pooled['observations']:,} drive-snapshots over {pooled['drives']:,} drives; "
        f"{pooled['failing_observations']:,} failing"
    )
    print(f"   base rate {pooled['base_rate']:.3%}  (precision lives at this rate)\n")

    print(
        f"{'>= tier':<9}{'flagged':>9}{'caught':>8}{'precision (95% CI)':>22}"
        f"{'recall (95% CI)':>21}{'FAR':>10}{'PLR':>6}{'lift':>6}"
    )
    for tier, m in pooled["tiers"].items():
        ci = pooled["ci"][tier]
        plr = (
            "-"
            if m["positive_likelihood_ratio"] is None
            else f"{m['positive_likelihood_ratio']:.0f}"
        )
        print(
            f"{tier:<9}{m['flagged']:>9,.0f}{m['caught']:>8,.0f}"
            f"{_fmt_ci(m['precision'], ci['precision']):>22}"
            f"{_fmt_ci(m['recall'], ci['recall']):>21}"
            f"{m['false_alarm_rate']:>10.2e}{plr:>6}{m['lift']:>6.0f}"
        )

    print("\nper snapshot, >= cordon:")
    header = ("date", "alive", "failed", "flagged", "caught", "precision", "recall")
    print("  " + f"{header[0]:<12}" + "".join(f"{h:>10}" for h in header[1:]))
    for row in report["per_date"]:
        c = row["cordon"]
        cells = (
            f"{row['alive']:,}",
            f"{row['failing']:,}",
            f"{c['flagged']:,.0f}",
            f"{c['caught']:,.0f}",
            f"{c['precision']:.1%}",
            f"{c['recall']:.1%}",
        )
        print("  " + f"{row['date']:<12}" + "".join(f"{x:>10}" for x in cells))

    looks = report.get("two_look")
    if looks:
        print(
            f"\none look vs two (min of T-{report['lag_days']}d and T), recall at matched "
            f"false-alarm rate, {report['two_look_pairs']:,} drive-pairs"
        )
        print(f"  {'FAR budget':>11}{'one look':>10}{'two looks':>11}{'difference (95% CI)':>26}")
        for row in looks:
            lo, hi = row["difference_ci"]
            print(
                f"  {row['false_alarm_budget']:>11.0e}{row['one_look_recall']:>10.1%}"
                f"{row['two_look_recall']:>11.1%}{row['difference']:>+12.1%} [{lo:+.1%},{hi:+.1%}]"
            )
        print("  A difference whose interval excludes zero is a second look adding information;")
        print("  one that straddles it is only a stricter threshold.")
    else:
        print("\n(no two-look comparison: no snapshot has an earlier one --lag-days before it)")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    apply_memory_limit_from_config()
    args = parse_args(argv)

    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    agent_config = yaml.safe_load(AGENT_CONFIG_PATH.read_text())

    horizon = args.horizon_days or model_config["primary_horizon_days"]
    data_end = dt.date.fromisoformat(data_config["sources"]["backblaze"]["end_date"])
    splits = model_config["splits"]
    latest_ok = last_observable_date(data_end, horizon)
    start = _date(args.start_date) or first_date_after_validation(model_config)
    assert start is not None, "configs/model.yaml splits.validation_end is not set"
    end = _date(args.end_date) or min(dt.date.fromisoformat(splits["test_end"]), latest_ok)
    if end > latest_ok:
        raise SystemExit(
            f"{end} has no observable {horizon}-day label (data ends {data_end}); "
            f"the latest usable date is {latest_ok}"
        )
    lag = args.lag_days or args.step_days
    if lag % args.step_days:
        raise SystemExit("--lag-days must be a multiple of --step-days so the earlier look exists")
    dates = select_snapshot_dates(start, end, args.step_days)
    cutoff = _date(args.outcome_cutoff)
    if cutoff is not None and max(dates) + dt.timedelta(days=horizon) > cutoff:
        raise SystemExit(
            f"{max(dates)} + {horizon} days ends after the outcome cutoff {cutoff}; "
            f"the latest date allowed is {cutoff - dt.timedelta(days=horizon)}"
        )

    if args.thresholds_path is not None:
        thresholds = load_thresholds_file(
            args.thresholds_path, tier_key=args.tier_key, fallback=agent_config["action_thresholds"]
        )
        run_id = (
            args.label
            or args.run_id
            or (
                args.model_dir.name if args.model_dir else f"{args.scores_path}:{args.score_column}"
            )
        )
    else:
        import mlflow

        mlflow.set_tracking_uri(model_config["mlflow"]["tracking_uri"])
        run_id = args.run_id or latest_run_id_for_horizon(
            model_config["mlflow"]["experiment_name"], horizon
        )
        thresholds = action_thresholds_for_run(run_id, agent_config["action_thresholds"])
    logger.info("point_in_time_thresholds", run_id=run_id, thresholds=thresholds)

    if overlaps_test_split(start, end, splits):
        record_test_access(
            variant="agent_point_in_time",
            purpose="fixed-date snapshots scored for the flagging policy, test window",
            extra={"run_id": run_id, "dates": [str(d) for d in dates], "horizon_days": horizon},
        )

    cross_check_dates = [_date(d) for d in args.cross_check_dates.split(",") if d.strip()]
    if args.model_dir is not None:
        from src.models.stacking import load_stacked_two_stage

        model = load_stacked_two_stage(args.model_dir)
    elif args.scores_path is not None and not cross_check_dates:
        model = None
    else:
        model = load_scoring_model(run_id, model_config["model"].get("type", "lightgbm"))
    gold = Path(data_config["gold_dir"])
    features_path, labels_path = (
        gold / "features" / "part.parquet",
        gold / "labels" / "part.parquet",
    )
    for path in (features_path, labels_path):
        if not path.exists():
            raise SystemExit(f"{path} not found; run the pipeline first")

    drive_ids = (
        split_drive_ids(labels_path, horizon_days=horizon, split=args.drives_from_split)
        if args.drives_from_split
        else None
    )
    scores_split = args.scores_split or None

    def score_day(day: dt.date, *, use_table: bool) -> pl.DataFrame:
        table = (
            table_scores_for_day(
                args.scores_path, day, split=scores_split, column=args.score_column
            )
            if use_table
            else None
        )
        return snapshot_observations(
            features_path,
            labels_path,
            model,
            day,
            horizon_days=horizon,
            context_columns=CONTEXT_COLUMNS,
            drive_ids=drive_ids,
            scores=table,
        )

    by_date: dict[dt.date, pl.DataFrame] = {}
    for day in dates:
        cached = (
            read_observations_partition(args.observations_dir, day)
            if args.observations_dir is not None
            else None
        )
        if cached is not None:
            obs = cached
            logger.info("point_in_time_partition_reused", date=str(day))
        else:
            obs = score_day(day, use_table=args.scores_path is not None)
            if args.observations_dir is not None and not obs.is_empty():
                write_observations_partition(obs, args.observations_dir, day)
        if obs.is_empty():
            logger.warning("point_in_time_no_rows", date=str(day))
            continue
        by_date[day] = obs
        logger.info(
            "point_in_time_snapshot",
            date=str(day),
            alive=obs.height,
            failing=int((obs["label"] == 1).sum()),
        )
    if not by_date:
        raise SystemExit("no snapshot had any labelled rows")

    pooled = pl.concat(by_date.values(), how="diagonal_relaxed")
    if str(args.observations_out):
        args.observations_out.parent.mkdir(parents=True, exist_ok=True)
        pooled.write_parquet(args.observations_out, compression="zstd")
        logger.info(
            "point_in_time_observations_saved",
            path=str(args.observations_out),
            rows=pooled.height,
            columns=pooled.columns,
        )
    counts = tier_counts(pooled, thresholds)
    point = tier_metrics(counts)
    ci = bootstrap_tier_ci(counts, n_boot=args.bootstrap, seed=args.seed)
    n_drives = int(counts.positives.shape[0])
    failing_obs = int((pooled["label"] == 1).sum())

    per_date = []
    for day, obs in by_date.items():
        m = tier_metrics(tier_counts(obs, thresholds))
        per_date.append(
            {
                "date": str(day),
                "alive": obs.height,
                "failing": int((obs["label"] == 1).sum()),
                **m,
            }
        )

    pairs: list[pl.DataFrame] = []
    for day, obs in by_date.items():
        earlier = by_date.get(day - dt.timedelta(days=lag))
        if earlier is not None:
            pairs.append(pair_observations(earlier, obs))
    pair_frame = pl.concat(pairs) if pairs else pl.DataFrame()
    two_look = (
        compare_looks(
            pair_frame, budgets=DEFAULT_FAR_BUDGETS, n_boot=min(args.bootstrap, 500), seed=args.seed
        )
        if pairs and int((pair_frame["label"] == 1).sum()) > 0
        else []
    )

    cross_check = []
    for day in cross_check_dates:
        if args.scores_path is None or model is None:
            raise SystemExit("--cross-check-dates needs both --scores-path and a model")
        a = score_day(day, use_table=True).select("drive_id", pl.col("p_fail").alias("table"))
        b = score_day(day, use_table=False).select("drive_id", pl.col("p_fail").alias("model"))
        both = a.join(b, on="drive_id", how="inner")
        delta = (both["table"] - both["model"]).abs()
        cross_check.append(
            {
                "date": str(day),
                "drives": both.height,
                "only_table": a.height - both.height,
                "only_model": b.height - both.height,
                "max_abs_delta": float(delta.max()) if both.height else None,
                "over_1e-6": int((delta > 1e-6).sum()),
            }
        )

    report = {
        "run_id": run_id,
        "label": args.label,
        "drive_filter": args.drives_from_split,
        "outcome_cutoff": args.outcome_cutoff,
        "scores_path": str(args.scores_path) if args.scores_path else None,
        "score_column": args.score_column if args.scores_path else None,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "thresholds_path": str(args.thresholds_path) if args.thresholds_path else None,
        "tier_key": args.tier_key if args.thresholds_path else None,
        "cross_check": cross_check,
        "horizon_days": horizon,
        "step_days": args.step_days,
        "lag_days": lag,
        "dates": [str(d) for d in by_date],
        "thresholds": thresholds,
        "pooled": {
            "observations": pooled.height,
            "drives": n_drives,
            "failing_observations": failing_obs,
            "base_rate": failing_obs / pooled.height,
            "tiers": point,
            "ci": {t: {k: list(v) for k, v in c.items()} for t, c in ci.items()},
        },
        "per_date": per_date,
        "two_look_pairs": pair_frame.height if pairs else 0,
        "two_look": two_look,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print_report(report)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
