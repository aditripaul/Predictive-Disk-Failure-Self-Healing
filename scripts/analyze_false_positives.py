"""Profile the model's false positives from saved point-in-time snapshots.

Run `make agent-pit` first; it saves every scored drive-snapshot to
data/audit/agent_point_in_time_observations.parquet. This then answers,
without re-scoring, whether the false positives share anything a policy could
act on (src/agent/false_positives.py):

    uv run python scripts/analyze_false_positives.py
    uv run python scripts/analyze_false_positives.py --tier migrate

Reads the thresholds the run used from data/audit/agent_point_in_time.json,
and the label table for late hits. Test-window data, already logged as a
test-split access by the run that produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import polars as pl
import yaml

from src.agent.false_positives import (
    age_bucket,
    attribute_profile,
    flag_streaks,
    flagged_at,
    late_hit_dates,
    late_hits,
    precision_by_group,
)
from src.agent.point_in_time import COMBINERS, compare_looks, pair_observations

DEFAULT_OBSERVATIONS = Path("data/audit/agent_point_in_time_observations.parquet")
DEFAULT_REPORT = Path("data/audit/agent_point_in_time.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
PRIORITY = (
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
    "command_timeout",
    "reallocation_event_count",
)


def _table(frame: pl.DataFrame, title: str) -> None:
    print(f"\n{title}")
    if frame.is_empty():
        print("   (column not present in the saved observations)")
        return
    # A fixed, readable column set: a wide table gets its middle columns
    # elided by the terminal, and the columns that matter are in the middle.
    wanted = [
        c
        for c in (
            frame.columns[0],
            "flagged",
            "caught",
            "precision",
            "share_of_fp",
            "base_rate",
            "recall",
            "plr",
        )
        if c in frame.columns
    ]
    with pl.Config(
        tbl_rows=30,
        tbl_cols=-1,
        tbl_width_chars=160,
        tbl_hide_dataframe_shape=True,
        float_precision=4,
    ):
        print(frame.select(wanted) if len(wanted) > 1 else frame)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tier", default="cordon", choices=("warn", "cordon", "migrate", "drain"))
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)

    for path in (args.observations, args.report):
        if not path.exists():
            raise SystemExit(f"{path} not found; run `make agent-pit` first")
    obs = pl.read_parquet(args.observations)
    report = json.loads(args.report.read_text())
    threshold = report["thresholds"][args.tier]
    horizon = report["horizon_days"]
    step = report["step_days"]
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    data_end = dt.date.fromisoformat(data_config["sources"]["backblaze"]["end_date"])
    labels_path = Path(data_config["gold_dir"]) / "labels" / "part.parquet"

    flagged = flagged_at(obs, threshold)
    caught = int((flagged["label"] == 1).sum())
    print(f"\n== false positives at >= {args.tier} (p_fail >= {threshold:.4f})")
    print(
        f"   {flagged.height:,} flagged drive-snapshots, {caught:,} caught, "
        f"{flagged.height - caught:,} false alarms, precision {caught / max(flagged.height, 1):.1%}"
    )

    # 1. late hits
    dates = sorted(obs["date"].unique().to_list())
    exact = late_hit_dates(dates, data_end=data_end, horizon_days=horizon)
    print(
        f"\n1. late hits: flagged, healthy for {horizon} days, failing in days "
        f"{horizon + 1}-{2 * horizon}"
    )
    if exact:
        future_dates = [d + dt.timedelta(days=horizon) for d in exact]
        future = (
            pl.scan_parquet(labels_path)
            .filter(
                (pl.col("horizon_days") == horizon)
                & pl.col("date").is_in(future_dates)
                & (pl.col("label") == 1)
            )
            .select("drive_id", "date")
            .collect()
        )
        result = late_hits(
            flagged.filter(pl.col("date").is_in(exact)), future, horizon_days=horizon
        )
        print(
            f"   on the {len(exact)} snapshots whose days {horizon + 1}-{2 * horizon} are "
            f"fully observed ({exact[0]}..{exact[-1]}):"
        )
        print(
            f"   {result['false_alarms']:,} false alarms, "
            f"{result['late_hits']:,} of them late hits "
            f"({result['late_share_of_false_alarms']:.1%})"
        )
        print(
            f"   precision {result['precision_30d']:.1%} by the {horizon}-day label, "
            f"{result['precision_with_late_credit']:.1%} crediting late hits"
        )
        print(f"   Report beside the {horizon}-day figure, never instead of it.")
    else:
        print(f"   no snapshot is {2 * horizon} days before the data ends; nothing exact to report")

    # 2. composition
    # Population = every scored drive-snapshot, so each group's own base rate
    # and lift are shown. Low precision alone may just mean a low failure rate.
    _table(
        precision_by_group(flagged, "drive_model", population=obs),
        "2a. by drive model (most false alarms first; compare families on plr, "
        "which ignores the base rate)",
    )
    _table(
        precision_by_group(
            age_bucket(flagged), "age_years", population=age_bucket(obs), min_flagged=1
        ),
        "2b. by age",
    )

    # 3. streaks
    by_date = {d: obs.filter(pl.col("date") == d) for d in dates}
    streaks = flag_streaks(by_date, threshold)
    if streaks.is_empty():
        print("\n3. flag streaks: need at least 4 snapshots")
    else:
        summary = (
            streaks.group_by("streak")
            .agg(pl.len().alias("flagged"), (pl.col("label") == 1).sum().alias("caught"))
            .with_columns((pl.col("caught") / pl.col("flagged")).alias("precision"))
            .sort("streak")
        )
        _table(summary, f"3. by consecutive earlier snapshots also flagged (every {step}d; 3 = 3+)")

    # 4. attributes
    _table(attribute_profile(flagged, PRIORITY), "4. share with each attribute non-zero")

    # 5. combiners
    pairs = [
        pair_observations(by_date[d - dt.timedelta(days=step)], by_date[d])
        for d in dates
        if d - dt.timedelta(days=step) in by_date
    ]
    if pairs:
        pair_frame = pl.concat(pairs, how="diagonal_relaxed")
        print(
            f"\n5. two looks {step}d apart vs one, recall at matched false-alarm rate "
            f"({pair_frame.height:,} pairs; difference and 95% CI)"
        )
        print(f"   {'budget':>8}" + "".join(f"{c:>24}" for c in COMBINERS))
        rows = {c: compare_looks(pair_frame, combiner=c, n_boot=args.bootstrap) for c in COMBINERS}
        for i, budget in enumerate(rows["min"]):
            cells = []
            for c in COMBINERS:
                r = rows[c][i]
                lo, hi = r["difference_ci"]
                cells.append(f"{r['difference']:+.1%} [{lo:+.1%},{hi:+.1%}]")
            print(f"   {budget['false_alarm_budget']:>8.0e}" + "".join(f"{x:>24}" for x in cells))
        print(
            "   One look's recall is the baseline; only an interval clear of zero is a real gain."
        )


if __name__ == "__main__":
    main()
