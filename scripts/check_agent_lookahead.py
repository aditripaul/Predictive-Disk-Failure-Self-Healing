"""Does the agent backtest peek at the future?

`pipelines/run_agent_fleet.py` scores each drive's LATEST row inside the
window, and `determine_action_tier` gates migrate/drain on `feature_maturity`
being MATURE. Both are fine for scoring a fleet *now*. Both are look-ahead in
a backtest over a past window:

  * `feature_maturity` is assigned from the dataset's last date. A drive whose
    final row is earlier than that is DECOMMISSIONED (src/preprocess/
    feature_maturity.py), checked before every other state. In a window ending
    2026-05-31 of data ending 2026-06-30, "not DECOMMISSIONED" therefore means
    "was still reporting a month after the snapshot" - and a drive that
    reports on the last day cannot have failed inside that window's 30-day
    horizon. The gate is reading the outcome.
  * "latest row" for a drive that died mid-window is its last row before it
    died: the moment of maximum signal and zero lead time. A fleet at a fixed
    date T does not contain drives that already died before T.

This reads a saved run and reports, rather than asserts, how much each
affects it:

    uv run python scripts/check_agent_lookahead.py

Prediction if the gate is reading the outcome: every migrate/drain drive is
MATURE with precision ~0, and the drives it downgraded to cordon are
DECOMMISSIONED with high precision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.agent.backtest import strongest_per_drive

DEFAULT_REPORT = Path("data/audit/agent_fleet_run.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")
ACTIVE = {"cordon", "migrate", "drain"}


def _mean(series: pl.Series) -> float:
    """Mean via numpy: Polars types its scalar results loosely, and an empty
    group should read as 0, not raise."""
    values = series.to_numpy()
    return float(values.mean()) if len(values) else 0.0


def _median(series: pl.Series) -> float:
    values = series.to_numpy()
    return float(np.median(values)) if len(values) else 0.0


def _frame(report: dict, metadata: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    rows = [
        {
            "drive_id": d["drive_id"],
            "tier": d["proposed_action"],
            "p_fail": d["p_fail"],
            "as_of": dt.date.fromisoformat(str(d["as_of"])[:10]) if d.get("as_of") else None,
        }
        for d in strongest_per_drive(report["decisions"]).values()
    ]
    frame = pl.DataFrame(rows).join(
        metadata.select(["drive_id", "feature_maturity", "last_seen_date"]),
        on="drive_id",
        how="left",
    )
    # The label at the exact row that was scored, for lead time.
    scored_labels = labels.rename({"date": "as_of"})
    return frame.join(
        scored_labels.select(["drive_id", "as_of", "label", "days_to_event"]),
        on=["drive_id", "as_of"],
        how="left",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text())
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    gold, silver = Path(data_config["gold_dir"]), Path(data_config["silver_dir"])
    horizon = report.get("scoring", {}).get("horizon_days") or model_config["primary_horizon_days"]
    start, end = (dt.date.fromisoformat(x) for x in report["window"])
    data_end = dt.date.fromisoformat(data_config["sources"]["backblaze"]["end_date"])

    metadata = pl.read_parquet(silver / "drive_metadata" / "part.parquet")
    labels = (
        pl.scan_parquet(gold / "labels" / "part.parquet")
        .filter(
            (pl.col("horizon_days") == horizon)
            & (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & pl.col("label").is_not_null()
        )
        .select(["drive_id", "date", "label", "days_to_event"])
        .collect()
    )
    failing = set(labels.filter(pl.col("label") == 1)["drive_id"].to_list())
    frame = _frame(report, metadata, labels).with_columns(
        pl.col("drive_id").is_in(failing).alias("failed")
    )

    print(f"\n== window {start}..{end}, data ends {data_end}, {horizon}-day horizon")
    print(f"   {frame.height:,} drives decided\n")

    print("1. tier x feature_maturity  (the gate reads maturity)")
    print(f"{'tier':<10}{'maturity':<16}{'drives':>8}{'failed':>8}{'precision':>11}")
    for (tier, maturity), g in frame.group_by(["tier", "feature_maturity"], maintain_order=True):
        precision = _mean(g["failed"])
        print(
            f"{tier:<10}{str(maturity):<16}{g.height:>8,}{g['failed'].sum():>8,}{precision:>11.1%}"
        )

    acted = frame.filter(pl.col("tier").is_in(list(ACTIVE)))
    gone = acted.filter(pl.col("last_seen_date") < data_end)
    print("\n2. acted drives that stopped reporting before the data ended")
    print(f"   {gone.height:,} of {acted.height:,} ({gone.height / max(acted.height, 1):.1%})")
    still_reporting = acted.filter(pl.col("last_seen_date") >= data_end)
    gone_precision = _mean(gone["failed"])
    rest_precision = _mean(still_reporting["failed"])
    print(f"   precision among them {gone_precision:.1%}; among the rest {rest_precision:.1%}")

    died_in_window = acted.filter(pl.col("as_of") < end)
    print("\n3. acted drives whose scored row is NOT the window's last day")
    print(f"   {died_in_window.height:,} of {acted.height:,} - these left the fleet mid-window,")
    print("   so their 'latest row' is the last one before they disappeared")

    caught = acted.filter(pl.col("failed") & pl.col("days_to_event").is_not_null())
    if caught.height:
        lead = caught["days_to_event"]
        print(f"\n4. lead time of caught drives, scored row to failure, n={caught.height:,}")
        print(f"   median {_median(lead):.0f}d   mean {_mean(lead):.1f}d")
        for cutoff in (1, 3, 7):
            share = _mean(lead <= cutoff)
            print(f"   caught with <= {cutoff}d notice: {share:.1%}")
        print("   A drive 'caught' the day it dies is not a save.")


if __name__ == "__main__":
    main()
