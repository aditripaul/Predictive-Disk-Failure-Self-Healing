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


def lead_days(caught: pl.DataFrame) -> pl.Series:
    """Days from the scored row to the drive's last row ever. For a failed
    drive the last row is the failure day, so this is the notice the agent
    had. Zero means it was scored on the day the drive died."""
    return (caught["last_seen_date"] - caught["as_of"]).dt.total_days()


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

    # Lead time is measured to the drive's LAST row, not read from the label.
    # A drive's final row - the failure day - has a NULL label
    # (src/labels/labeling.py: event_within_horizon needs event_date > date,
    # and the drive is not observable to the horizon's end), so filtering to
    # labelled scored rows silently drops exactly the drives caught on their
    # last day and reports the easy 15% as if it were the whole.
    caught = acted.filter(pl.col("failed") & pl.col("last_seen_date").is_not_null())
    if caught.height:
        lead = lead_days(caught)
        print(f"\n4. lead time of ALL caught drives, scored row to last row, n={caught.height:,}")
        print(f"   median {_median(lead):.0f}d   mean {_mean(lead):.1f}d")
        for cutoff in (0, 1, 3, 7, 14):
            print(f"   caught with <= {cutoff:>2}d notice: {_mean(lead <= cutoff):.1%}")
        print("   A drive 'caught' the day it dies is not a save.")

    _clean_subset(report, frame, labels, end)


def _clean_subset(report: dict, frame: pl.DataFrame, labels: pl.DataFrame, end: dt.date) -> None:
    """The point-in-time answer hiding inside the run.

    Drives whose scored row IS the window's last day were alive at a fixed
    date T=end, so they are a legitimate snapshot: no dying-drive selection,
    and a real label at T (did it fail in the next 30 days). Tiers are
    assigned from p_fail and the thresholds alone, ignoring the maturity
    gate, which downgrades rather than blocks and is the thing that reads the
    outcome. Recall is against every drive alive at T, not just candidates."""
    thresholds = report["action_thresholds"]
    at_end = labels.filter(pl.col("date") == end)
    positives, labelled = int((at_end["label"] == 1).sum()), at_end.height
    if not labelled:
        print("\n5. clean subset: no labelled rows on the window's last day")
        return
    base_rate = positives / labelled

    pool = frame.filter((pl.col("as_of") == end) & pl.col("label").is_not_null())
    print(f"\n5. clean point-in-time subset: the fleet as of {end}")
    print(
        f"   {labelled:,} drives alive and labelled, {positives:,} failed in the next 30 days "
        f"(base rate {base_rate:.3%})"
    )
    if report.get("candidate_count") != report.get("above_threshold_count"):
        print("   !! candidates were truncated; recall below is a floor")
    print(
        f"   {'threshold':<14}{'flagged':>9}{'failed':>8}{'precision':>11}{'recall':>9}{'lift':>8}"
    )
    for tier in ("warn", "cordon", "migrate", "drain"):
        flagged = pool.filter(pl.col("p_fail") >= thresholds[tier])
        caught = int((flagged["label"] == 1).sum())
        precision = caught / flagged.height if flagged.height else 0.0
        recall = caught / positives if positives else 0.0
        lift = precision / base_rate if base_rate else 0.0
        print(
            f"   >= {tier:<11}{flagged.height:>9,}{caught:>8,}{precision:>11.1%}"
            f"{recall:>9.1%}{lift:>8.0f}"
        )
    print("   No maturity gate, no vanished-drive rows. This is the number the")
    print("   agent's Apr-May headline was standing in for.")


if __name__ == "__main__":
    main()
