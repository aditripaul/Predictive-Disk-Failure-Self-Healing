"""Draw the alerts the 30-day label calls false that were really early.

Run `make agent-pit` first; it saves every scored drive-snapshot to
data/audit/agent_point_in_time_observations.parquet. This takes the drives
flagged at one tier, follows each one forward in the label table, and shows
what became of them (src/agent/false_positives.py::alert_outcomes):

    uv run python scripts/plot_late_hits.py
    uv run python scripts/plot_late_hits.py --tier migrate --max-lanes 20

Writes three figures and a summary to data/audit/plots/:

  late_hits_outcomes.png    what became of every flagged drive-snapshot
  late_hits_lead_time.png   days from the alert to the failure, with the
                            30-day label boundary marked
  late_hits_timelines.png   one row per early alert: alert at day 0, failure
                            at day N, the label window shaded
  late_hits_by_tier.png     precision per tier by the label and over 60 days
  late_hits_vs_background.png  share of each tier's false alarms that failed
                            in days 31-60, against every healthy drive
  late_hits_profile.png     median bad-sector counts at the alert, by outcome
  late_hits_summary.json    the numbers behind the figures

Only snapshots whose days 31-60 are fully observed are used, as in
scripts/analyze_false_positives.py, otherwise early alerts are undercounted.
The 30-day figure is reported beside the early-alert one, never replaced by it:
the goal is pinned to a 30-day label (runbook section 7). Test-window data,
already logged as a test-split access by the run that produced it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.agent.false_positives import (
    OUTCOME_ORDER,
    alert_outcomes,
    flagged_at,
    late_hit_dates,
    late_hits,
)
from src.reporting.late_hit_figures import (
    plot_alert_outcomes,
    plot_alert_profile,
    plot_alert_timelines,
    plot_later_failure_rate,
    plot_lead_time_histogram,
    plot_precision_by_tier,
)
from src.reporting.plots import save_figure

DEFAULT_OBSERVATIONS = Path("data/audit/agent_point_in_time_observations.parquet")
DEFAULT_REPORT = Path("data/audit/agent_point_in_time.json")
DATA_CONFIG_PATH = Path("configs/data.yaml")
TIERS = ("warn", "cordon", "migrate", "drain")
PROFILE_ATTRIBUTES = {
    "reallocated_sector_count": "Reallocated sectors",
    "current_pending_sector_count": "Pending sectors",
    "reallocation_event_count": "Reallocation events",
}


def tier_rows(obs: pl.DataFrame, events: pl.DataFrame, thresholds: dict, horizon: int) -> list:
    """Per tier: alerts, precision by the label and with days 31-60 credited,
    and the share of the label's false alarms that failed in days 31-60."""
    rows = []
    for tier in TIERS:
        out = alert_outcomes(flagged_at(obs, thresholds[tier]), events, horizon_days=horizon)
        n = out.height
        correct = int((out["outcome"] == "correct").sum())
        early = int((out["outcome"] == "early").sum())
        false_alarms = n - correct
        rows.append(
            {
                "tier": tier,
                "alerts": n,
                "false_alarms": false_alarms,
                "precision_label": correct / n if n else 0.0,
                "precision_credited": (correct + early) / n if n else 0.0,
                "failed_later_share": early / false_alarms if false_alarms else 0.0,
            }
        )
    return rows


def outcome_profile(outcomes: pl.DataFrame) -> tuple[dict, dict]:
    """Median of each profile attribute at the alert, by outcome group."""
    group = (
        pl.when(pl.col("outcome") == "correct")
        .then(pl.lit("correct"))
        .when(pl.col("outcome").is_in(["early", "later"]))
        .then(pl.lit("failed_later"))
        .otherwise(pl.lit("never_failed"))
    )
    table = (
        outcomes.with_columns(group.alias("group"))
        .group_by("group")
        .agg(pl.len().alias("n"), *[pl.col(c).median().alias(c) for c in PROFILE_ATTRIBUTES])
    )
    profile = {
        r["group"]: {c: float(r[c] or 0.0) for c in PROFILE_ATTRIBUTES}
        for r in table.iter_rows(named=True)
    }
    for g in ("correct", "failed_later", "never_failed"):
        profile.setdefault(g, dict.fromkeys(PROFILE_ATTRIBUTES, 0.0))
    return profile, {r["group"]: int(r["n"]) for r in table.iter_rows(named=True)}


def early_alert_rows(outcomes: pl.DataFrame, max_lanes: int) -> tuple[list[dict], int]:
    """One row per drive that alerted early (its first such alert), sampled
    evenly across lead times when there are more than `max_lanes`.

    Returns `(rows sorted by lead time, number of early-alert drives)`."""
    early = (
        outcomes.filter(pl.col("outcome") == "early")
        .sort("date")
        .group_by("drive_id", maintain_order=True)
        .first()
        .sort("days_to_failure")
    )
    total = early.height
    if total > max_lanes:
        pick = np.unique(np.linspace(0, total - 1, max_lanes).round().astype(int))
        early = early[pick.tolist()]
    ids = [str(i) for i in early["drive_id"].to_list()]
    prefix = 5
    while len({i[:prefix] for i in ids}) < len(ids) and prefix < max(map(len, ids), default=0):
        prefix += 1  # shortest prefix that tells these rows apart: serial suffixes repeat by model
    rows = [
        {
            "label": f"{r['drive_model']}  {str(r['drive_id'])[:prefix]}",
            "lead_days": int(r["days_to_failure"]),
        }
        for r in early.iter_rows(named=True)
    ]
    return rows, total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tier", default="cordon", choices=("warn", "cordon", "migrate", "drain"))
    parser.add_argument("--max-lanes", type=int, default=30)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    for path in (args.observations, args.report):
        if not path.exists():
            raise SystemExit(f"{path} not found; run `make agent-pit` first")
    obs = pl.read_parquet(args.observations)
    report = json.loads(args.report.read_text())
    horizon = int(report["horizon_days"])
    threshold = report["thresholds"][args.tier]
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    data_end = dt.date.fromisoformat(str(data_config["sources"]["backblaze"]["end_date"]))
    labels_path = Path(data_config["gold_dir"]) / "labels" / "part.parquet"
    out_dir = args.out_dir or Path(data_config["audit_dir"]) / "plots"

    dates = late_hit_dates(
        sorted(obs["date"].unique().to_list()), data_end=data_end, horizon_days=horizon
    )
    if not dates:
        raise SystemExit("no snapshot has its days 31-60 inside the data; nothing to show")
    observed = obs.filter(pl.col("date").is_in(dates))
    flagged = flagged_at(observed, threshold)
    events = (
        pl.scan_parquet(labels_path)
        .filter((pl.col("horizon_days") == horizon) & pl.col("date").is_in(dates))
        .select("drive_id", "date", "event_date", "event_type")
        .collect()
    )
    outcomes = alert_outcomes(flagged, events, horizon_days=horizon)
    if int((outcomes["label"] == 1).sum()) != int((outcomes["outcome"] == "correct").sum()):
        raise SystemExit("outcome 'correct' disagrees with the label table; refusing to draw")

    # The runbook's method (analyze_false_positives.py) needs the drive to still report
    # telemetry on exactly day 30; the event date does not. Report both so they reconcile.
    future = (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("horizon_days") == horizon)
            & pl.col("date").is_in([d + dt.timedelta(days=horizon) for d in dates])
            & (pl.col("label") == 1)
        )
        .select("drive_id", "date")
        .collect()
    )
    runbook_method = late_hits(flagged, future, horizon_days=horizon)

    counts = {k: int((outcomes["outcome"] == k).sum()) for k in OUTCOME_ORDER}
    failed_days = outcomes["days_to_failure"].drop_nulls().to_list()
    rows, early_drives = early_alert_rows(outcomes, args.max_lanes)

    save_figure(
        plot_alert_outcomes(counts, horizon_days=horizon, tier=args.tier, snapshots=len(dates)),
        out_dir / "late_hits_outcomes.png",
    )
    save_figure(
        plot_lead_time_histogram([int(d) for d in failed_days], horizon_days=horizon),
        out_dir / "late_hits_lead_time.png",
    )
    save_figure(
        plot_alert_timelines(rows, horizon_days=horizon, total_early=early_drives),
        out_dir / "late_hits_timelines.png",
    )

    by_tier = tier_rows(observed, events, report["thresholds"], horizon)
    everyone = alert_outcomes(observed, events, horizon_days=horizon)
    healthy = everyone.filter(pl.col("outcome") != "correct")
    background = float((healthy["outcome"] == "early").mean()) if healthy.height else 0.0
    save_figure(
        plot_precision_by_tier(by_tier, horizon_days=horizon), out_dir / "late_hits_by_tier.png"
    )
    save_figure(
        plot_later_failure_rate(by_tier, background=background, horizon_days=horizon),
        out_dir / "late_hits_vs_background.png",
    )
    profile, group_counts = outcome_profile(outcomes)
    save_figure(
        plot_alert_profile(profile, attributes=PROFILE_ATTRIBUTES, counts=group_counts),
        out_dir / "late_hits_profile.png",
    )

    total = sum(counts.values())
    false_alarms = total - counts["correct"]
    summary = {
        "tier": args.tier,
        "threshold": threshold,
        "horizon_days": horizon,
        "snapshot_dates": [d.isoformat() for d in dates],
        "flagged_drive_snapshots": total,
        "outcome_counts": counts,
        "false_alarms_by_30_day_label": false_alarms,
        "early_alerts_failed_days_31_to_60": counts["early"],
        "early_alert_share_of_false_alarms": counts["early"] / false_alarms if false_alarms else 0,
        "failed_later_share_of_false_alarms": counts["later"] / false_alarms if false_alarms else 0,
        "precision_30_day_label": counts["correct"] / total if total else 0,
        "precision_with_days_31_to_60_credited": (counts["correct"] + counts["early"]) / total
        if total
        else 0,
        "early_alert_drives": early_drives,
        "by_tier": by_tier,
        "background_failed_in_days_31_to_60": background,
        "profile_medians_at_alert": profile,
        "profile_group_counts": group_counts,
        "runbook_method_late_hits": runbook_method["late_hits"],
        "runbook_method_note": (
            "analyze_false_positives.py counts a late hit only if the drive still has a "
            "telemetry row on day 30; the event date used here also counts drives that "
            "failed in days 31-60 after they stopped reporting, so it is the larger count"
        ),
        "note": "later = failed after day 60, seen only to the end of the data (a lower bound)",
    }
    (out_dir / "late_hits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nfigures written to {out_dir}/late_hits_*.png")


if __name__ == "__main__":
    main()
