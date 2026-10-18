"""Profile the failing drives the model misses, and bound the recall SMART allows.

Run `make agent-pit` first; it saves every scored drive-snapshot to
data/audit/agent_point_in_time_observations.parquet. This joins those
snapshots to the gold table's trailing-window columns for the core counters
and answers, without re-scoring (src/agent/missed_failures.py):

  * what share of failing drives showed nothing in their core counters
    (silent or no_data) - recall no threshold on these counters can reach;
  * how the model's recall splits across what the drives did show;
  * how far under the threshold the missed drives sat, as the false-alarm
    rate it would cost to catch them;
  * which drive models lose recall to missing telemetry and which to the model.

    uv run python scripts/analyze_missed_failures.py
    uv run python scripts/analyze_missed_failures.py --tier migrate

Reads the thresholds the run used from data/audit/agent_point_in_time.json.
Test-window data, already logged as a test-split access by the run that
produced it. The figures are for the model `make agent-pit` scored (its run
id is printed), at drive level over the snapshots, not the frozen run's
training-time evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

from src.agent.missed_failures import (
    class_base_rates,
    class_by_family,
    failing_drives,
    false_alarm_rate_to_catch,
    far_quantiles,
    recall_by_class,
    recall_ceiling,
    signal_class,
    window_columns,
)
from src.agent.point_in_time import TIERS

DEFAULT_OBSERVATIONS = Path("data/audit/agent_point_in_time_observations.parquet")
DEFAULT_REPORT = Path("data/audit/agent_point_in_time.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")


def _print(frame: pl.DataFrame, title: str) -> None:
    print(f"\n{title}")
    if frame.is_empty():
        print("   (nothing to show)")
        return
    with pl.Config(
        tbl_rows=30,
        tbl_cols=-1,
        tbl_width_chars=160,
        tbl_hide_dataframe_shape=True,
        float_precision=4,
    ):
        print(frame)


def _with_windows(
    obs: pl.DataFrame, features_path: Path, attributes: tuple[str, ...], window_days: int
) -> pl.DataFrame:
    """`obs` joined to the gold window columns on (drive_id, date). Only the
    needed columns and the snapshot dates are read: the gold table is wide."""
    available = set(pl.scan_parquet(features_path).collect_schema().names())
    columns = [c for c in window_columns(attributes, window_days) if c in available]
    if not columns:
        raise SystemExit(
            f"{features_path} has no {window_days}-day max/range columns for the core counters"
        )
    dates = obs["date"].unique().to_list()
    windows = (
        pl.scan_parquet(features_path)
        .filter(pl.col("date").is_in(dates))
        .select(["drive_id", "date", *columns])
        .collect()
        .with_columns(pl.lit(True).alias("_in_gold"))
    )
    joined = obs.join(windows, on=["drive_id", "date"], how="left")
    # A snapshot with no gold row would read as no_data; it must not.
    unmatched = int(joined["_in_gold"].is_null().sum())
    if unmatched:
        raise SystemExit(
            f"{unmatched:,} of {joined.height:,} snapshots have no gold row: the gold table "
            "is from another build than the observations; re-run `make agent-pit`"
        )
    return joined.drop("_in_gold")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tier", default="cordon", choices=TIERS)
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args(argv)

    for path in (args.observations, args.report):
        if not path.exists():
            raise SystemExit(f"{path} not found; run `make agent-pit` first")
    report = json.loads(args.report.read_text())
    thresholds = report["thresholds"]
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    features_path = Path(data_config["gold_dir"]) / "features" / "part.parquet"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found; run the pipeline first")
    attributes = tuple(
        yaml.safe_load(FEATURES_CONFIG_PATH.read_text())["priority_smart_attributes"]
    )

    obs = signal_class(
        _with_windows(
            pl.read_parquet(args.observations), features_path, attributes, args.window_days
        ),
        attributes,
        window_days=args.window_days,
    )
    print(
        f"\n== missed failures, model run {report.get('run_id', '?')}, "
        f"{report.get('horizon_days', '?')}-day horizon, {obs['date'].n_unique()} snapshots, "
        f"core counters over a trailing {args.window_days}-day window"
    )

    # 1. ceiling, and the model's recall at every tier
    ceiling = recall_ceiling(failing_drives(obs, thresholds[args.tier]))
    print(
        f"\n1. recall ceiling from the core counters ({ceiling['failing_drives']:,} failing drives)"
    )
    print(
        f"   active only (a counter rose):      {ceiling['ceiling_active_only']:.1%}\n"
        f"   active or static (any non-zero):   {ceiling['ceiling_with_static']:.1%}"
        "   (optimistic: many healthy drives are static too, see 3)"
    )
    print("   model recall by tier:")
    for tier in TIERS:
        threshold = thresholds.get(tier)
        if threshold is None:
            print(f"     {tier:>8}: (no threshold)")
            continue
        r = recall_ceiling(failing_drives(obs, threshold))["model_recall"]
        print(f"     {tier:>8} (p_fail >= {threshold:.4f}): {r:.1%}")

    # 2. recall by class at the chosen tier
    drives = failing_drives(obs, thresholds[args.tier])
    _print(recall_by_class(drives), f"2. failing drives by signal class, at >= {args.tier}")

    # 3. how common each class is among healthy snapshots
    _print(
        class_base_rates(obs),
        "3. snapshots by signal class (lift = share of failing / share of healthy)",
    )

    # 4. how far under the threshold the misses sat
    healthy = obs.filter(pl.col("label") == 0)["p_fail"].to_numpy()
    with_far = false_alarm_rate_to_catch(drives, healthy)
    _print(
        far_quantiles(with_far),
        f"4. missed drives: share of healthy snapshots scoring at or above each "
        f"(the false-alarm rate to catch it; the {args.tier} tier runs at about "
        f"{float((healthy >= thresholds[args.tier]).mean()):.1e})",
    )

    # 5. by drive model
    _print(
        class_by_family(with_far),
        "5. by drive model (quiet = silent or no_data; recall_in_active = what the model "
        "does with the signal it had)",
    )


if __name__ == "__main__":
    main()
