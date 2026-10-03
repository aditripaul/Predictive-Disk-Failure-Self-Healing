"""Drive-level alerting analysis: persistence rules and false-alarm
forensics.

A row-level model scores each drive-day on its own, so a healthy drive
that has one noisy day alerts as loudly as one that is really failing -
and at ~345k healthy drives, a tiny per-row false-positive rate is
hundreds of false alarms. Two tools for the model goal's precision gap:

- `smooth_scores`: causal rolling statistics over each drive's last n
  observations ("alert only if the score stayed high"), which suppress
  sporadic spikes while sustained degradation persists.
- `analyze_false_alarms`: what the "healthy" drives that alerted actually
  are - failures just beyond the label horizon, drives pulled without
  failing, or genuinely healthy - to separate model error from label and
  evaluation artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.labels.event_types import FAILURE_EVENT_TYPES

#: name -> (rolling statistic, window in observations)
SMOOTHERS: dict[str, tuple[str, int]] = {
    "raw": ("raw", 1),
    "mean3": ("mean", 3),
    "mean5": ("mean", 5),
    "mean7": ("mean", 7),
    "min3": ("min", 3),
    "min5": ("min", 5),
    "median3": ("median", 3),  # 2 of the last 3 observations at/above the threshold
    "median5": ("median", 5),  # 3 of the last 5
}


def _date_series(dates: np.ndarray) -> pl.Series:
    """Polars Date series from a datetime64 array (what `Series.to_numpy()`
    gives for a Date column) or an object array of `datetime.date`."""
    return pl.Series("date", dates.tolist() if dates.dtype == object else dates)


def smooth_scores(
    drive_ids: np.ndarray, dates: np.ndarray, scores: np.ndarray, smoother: str
) -> np.ndarray:
    """Causal per-drive rolling statistic of `scores` (only the current and
    previous rows of the same drive, ordered by date), returned in the
    input row order. `mean` uses whatever history exists so far; `min` and
    `median` need a full window (earlier rows score 0.0), so a brand-new
    drive cannot alert on its first day. Windows count observations, not
    calendar days - a telemetry gap simply widens the window."""
    stat, window = SMOOTHERS[smoother]
    if stat == "raw":
        return np.asarray(scores, dtype=float)
    df = pl.DataFrame(
        {
            "row": np.arange(len(scores)),
            "drive_id": drive_ids,
            "date": _date_series(dates),
            "score": scores,
        }
    ).sort(["drive_id", "date", "row"])
    col = pl.col("score")
    rolled = {
        "mean": col.rolling_mean(window_size=window, min_samples=1),
        "min": col.rolling_min(window_size=window, min_samples=window),
        "median": col.rolling_median(window_size=window, min_samples=window),
    }[stat].over("drive_id")
    out = df.with_columns(rolled.fill_null(0.0).alias("smoothed")).sort("row")
    return out["smoothed"].to_numpy()


def _bucket_days(days: float | None) -> str:
    if days is None:
        return "unknown"
    if days <= 30:
        return "15-30d"
    if days <= 60:
        return "31-60d"
    return ">60d"


def analyze_false_alarms(
    drive_ids: np.ndarray,
    dates: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    event_types: np.ndarray,
    days_to_event: np.ndarray,
) -> dict[str, Any]:
    """Breaks down the "false alarm" drives at `threshold`: healthy drives
    (no positive-label row) whose score reached it. Each is classed by its
    `event_type` (`src/labels/event_types.py`); drives that went on to FAIL
    just outside the label horizon are additionally bucketed by how many
    days after the first alert the failure came. `base_rate` is the same
    breakdown over ALL healthy drives, so a class that is over-represented
    among the alerts (lift > 1) is one the model is genuinely picking up."""
    df = pl.DataFrame(
        {
            "drive_id": drive_ids,
            "date": _date_series(dates),
            "y": y_true,
            "score": scores,
            "event_type": event_types,
            "days_to_event": days_to_event,
        }
    )
    failing = df.filter(pl.col("y") == 1).select("drive_id").unique()
    healthy = df.join(failing, on="drive_id", how="anti")
    per_drive = healthy.group_by("drive_id").agg(
        pl.col("event_type").first().fill_null("still_active").alias("event_type"),
        (pl.col("score") >= threshold).any().alias("alerted"),
    )
    first_alert = (
        healthy.filter(pl.col("score") >= threshold)
        .sort("date")
        .group_by("drive_id")
        .agg(pl.col("days_to_event").first().alias("days_to_event_at_alert"))
    )
    alerted = per_drive.filter(pl.col("alerted")).join(first_alert, on="drive_id", how="left")

    def breakdown(frame: pl.DataFrame) -> dict[str, int]:
        return dict(
            frame.group_by("event_type")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .iter_rows()
        )

    alerted_counts = breakdown(alerted)
    base_counts = breakdown(per_drive)
    total_alerted = alerted.height
    total_healthy = per_drive.height
    failure_rows = alerted.filter(pl.col("event_type").is_in(sorted(FAILURE_EVENT_TYPES)))
    late_failure_buckets: dict[str, int] = {}
    for days in failure_rows["days_to_event_at_alert"].to_list():
        bucket = _bucket_days(days)
        late_failure_buckets[bucket] = late_failure_buckets.get(bucket, 0) + 1
    lift = {
        event_type: (
            (count / total_alerted) / (base_counts[event_type] / total_healthy)
            if total_alerted and base_counts.get(event_type)
            else None
        )
        for event_type, count in alerted_counts.items()
    }
    return {
        "threshold": threshold,
        "false_alarm_drives": total_alerted,
        "healthy_drives": total_healthy,
        "alerted_by_event_type": alerted_counts,
        "base_rate_by_event_type": base_counts,
        "lift_by_event_type": lift,
        "failed_later_days_after_alert": late_failure_buckets,
    }
