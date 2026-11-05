"""A daily, point-in-time replay of a recall-first alerting policy.

The agent-fleet loop replays one look-ahead snapshot (runbook 4.7), so it cannot say what a
looser, higher-recall warn tier would cost day to day. A `train-merged` score table has a
score for every drive on every reporting day of the validation and test windows, so the
policy can be replayed in order and judged on what actually happened next.

Policy: a drive is WARNED the first day its score reaches `warn` (non-disruptive: a
watchlist), and ESCALATED to cordon / migrate / drain the first day its score has cleared
that tier's threshold on `k` consecutive reporting days. `warn` is set on validation so that
a target share of failing drives were warned at least `lead_min` days before they failed.

Conventions the numbers depend on:

* a failing drive is CAUGHT at `lead` if some scored day in [failure - horizon, failure -
  lead] is at or above `warn`; a warning on the day of the failure is not a catch;
* the recall denominator is the failing drives that have at least one scored day in
  [failure - horizon, failure - 1]; a drive with no scored day in that window cannot be
  judged either way and is left out (and counted);
* precision is over drives, not drive-days, with the same notion of "caught" as recall: for
  WARN it is caught drives / warned drives, for a tier it is the failing drives that had a
  qualifying day (score above the tier for `k` consecutive days) in [failure - horizon,
  failure - lead] / drives that ever qualified, so the tiers nest;
* drives already at or above `warn` on the window's first day are "carried in" and reported
  separately; daily rates and the watchlist are measured after a warm-up of `horizon` days.

Nothing here models the confidence, maturity or staleness gates or topology guardrails of
the live loop: the score table has no such columns.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import polars as pl

from src.labels.event_types import FAILURE_EVENT_TYPES

TIERS = ("cordon", "migrate", "drain")


def failure_dates(labels_path: Path, *, horizon_days: int, since: dt.date) -> pl.DataFrame:
    """`drive_id, failure_date`: each drive's first failure on or after `since`."""
    return (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("horizon_days") == horizon_days)
            & pl.col("event_type").is_in(sorted(FAILURE_EVENT_TYPES))
            & (pl.col("event_date") >= since)
        )
        .group_by("drive_id")
        .agg(pl.col("event_date").min().alias("failure_date"))
        .collect()
    )


def _with_days_before(daily: pl.DataFrame, failures: pl.DataFrame) -> pl.DataFrame:
    """The scored days of failing drives, with how many days before the failure each is."""
    return daily.join(failures, on="drive_id").with_columns(
        (pl.col("failure_date") - pl.col("date")).dt.total_days().alias("days_before")
    )


def window_max(
    daily: pl.DataFrame, failures: pl.DataFrame, *, horizon_days: int, lead: int = 1
) -> pl.DataFrame:
    """`drive_id, failure_date, score`: each failing drive's highest score over its scored
    days in [failure - horizon, failure - lead]. Drives with no such day are absent."""
    return (
        _with_days_before(daily, failures)
        .filter(pl.col("days_before").is_between(lead, horizon_days))
        .group_by("drive_id", "failure_date")
        .agg(pl.col("s").max().alias("score"))
    )


def warn_threshold_for_recall(window_maxima: np.ndarray, recall_target: float) -> float:
    """The highest threshold at which at least `recall_target` of the failing drives would
    have been warned: a drive is warned iff its window maximum is >= the threshold."""
    values = np.sort(np.asarray(window_maxima, dtype=np.float64))[::-1]
    if len(values) == 0:
        raise ValueError("no failing drive has a scored day inside its warning window")
    needed = max(math.ceil(recall_target * len(values)), 1)
    return float(values[needed - 1])


def streak_lengths(above: np.ndarray, drive_codes: np.ndarray) -> np.ndarray:
    """For rows sorted by (drive, date): the run of consecutive `above` rows ending at each
    row, restarting at every drive boundary; 0 where `above` is False."""
    n = len(above)
    position = np.arange(n)
    new_drive = np.r_[True, drive_codes[1:] != drive_codes[:-1]]
    reset = np.where(~above, position, np.where(new_drive, position - 1, -1))
    last_reset = np.maximum.accumulate(reset)
    return np.where(above, position - last_reset, 0)


def first_dates(daily: pl.DataFrame, threshold: float, k: int = 1) -> pl.DataFrame:
    """`drive_id, first, last`: the first date a drive's score has been >= `threshold` for `k`
    consecutive scored days, and the last date it is >= `threshold`, over rows sorted by
    (drive, date). `last` ignores `k`, so it is only meaningful for k = 1."""
    codes = daily["drive_id"].cast(pl.Categorical).to_physical().to_numpy()
    above = daily["s"].to_numpy() >= threshold
    streak = streak_lengths(above, codes)
    hit = daily.filter(pl.Series(streak >= k))
    firsts = hit.group_by("drive_id").agg(pl.col("date").min().alias("first"))
    if k == 1:
        return firsts.join(
            hit.group_by("drive_id").agg(pl.col("date").max().alias("last")), on="drive_id"
        )
    return firsts.with_columns(pl.col("first").alias("last"))


def qualifying_rows(daily: pl.DataFrame, threshold: float, k: int) -> pl.DataFrame:
    """`drive_id, date`: every scored day on which the score has been >= `threshold` for `k`
    consecutive scored days, over rows sorted by (drive, date)."""
    codes = daily["drive_id"].cast(pl.Categorical).to_physical().to_numpy()
    streak = streak_lengths(daily["s"].to_numpy() >= threshold, codes)
    return daily.filter(pl.Series(streak >= k)).select("drive_id", "date")


def policy_metrics(
    daily: pl.DataFrame,
    failures: pl.DataFrame,
    *,
    warn: float,
    tiers: dict[str, float],
    k: int,
    horizon_days: int,
    window: tuple[dt.date, dt.date],
    data_end: dt.date,
    lead_min: int = 1,
) -> dict:
    """Replay the policy over `daily` (columns drive_id, date, s, sorted by drive and date)."""
    start, end = window
    in_window = failures.filter(
        pl.col("failure_date").is_between(
            start + dt.timedelta(days=1), end + dt.timedelta(days=horizon_days)
        )
    )
    eligible = window_max(daily, in_window, horizon_days=horizon_days, lead=lead_min)
    n_failing = in_window.height
    eligible_ids = eligible["drive_id"]
    failing_rows = _with_days_before(
        daily, in_window.filter(pl.col("drive_id").is_in(eligible_ids))
    )
    warned_rows = failing_rows.filter(pl.col("s") >= warn)

    def caught_at(lead: int) -> pl.DataFrame:
        return (
            warned_rows.filter(pl.col("days_before").is_between(lead, horizon_days))
            .group_by("drive_id", "failure_date")
            .agg(pl.col("days_before").max().alias("lead_days"))
        )

    caught = {lead: caught_at(lead) for lead in (lead_min, 7)}
    warn_dates = first_dates(daily, warn, 1)
    n_eligible = eligible.height
    n_warned = warn_dates.height
    n_caught = caught[lead_min].height

    days = (end - start).days + 1
    offsets = (
        warn_dates.filter(pl.col("first").is_between(start, end))
        .select((pl.col("first") - pl.lit(start)).dt.total_days())
        .to_series()
        .to_numpy()
    )
    # Drives already at or above `warn` on the window's first day were warned before the window
    # opened (the replay has no earlier history), so they are "carried in", not new warnings.
    carried_in = int((offsets == 0).sum())
    per_day = np.bincount(offsets[offsets > 0], minlength=days)[1:]

    out: dict = {
        "failing_drives_in_range": n_failing,
        "failing_drives_judged": n_eligible,
        "warned_drives": n_warned,
        "caught_drives": n_caught,
        "recall": n_caught / n_eligible if n_eligible else None,
        "recall_lead_7": caught[7].height / n_eligible if n_eligible else None,
        "warn_precision": n_caught / n_warned if n_warned else None,
        "median_lead_days": float(np.median(caught[lead_min]["lead_days"].to_numpy()))
        if n_caught
        else None,
        "carried_in_at_start": carried_in,
        "new_warns_per_day": float(np.mean(per_day)) if len(per_day) else 0.0,
        "new_warns_peak_day": int(np.max(per_day)) if len(per_day) else 0,
        "tiers": {},
    }

    # Late hits: warned, but the failure came 31 to 60 days after the last warning. Only drives
    # whose full 60 days after the last warning lie inside the data can be judged.
    cutoff = data_end - dt.timedelta(days=2 * horizon_days)
    judged = warn_dates.filter(pl.col("last") <= cutoff)
    late = (
        judged.join(failures, on="drive_id")
        .with_columns((pl.col("failure_date") - pl.col("last")).dt.total_days().alias("gap"))
        .filter(pl.col("gap").is_between(horizon_days + 1, 2 * horizon_days))
    )
    out["late_hits"] = {"judged_warned_drives": judged.height, "late_hits": late.height}

    escalated_first: pl.DataFrame | None = None
    for tier in TIERS:
        qualifying = qualifying_rows(daily, max(tiers[tier], warn), k)
        escalated = qualifying["drive_id"].n_unique()
        caught_tier = (
            _with_days_before(qualifying, in_window)
            .filter(pl.col("days_before").is_between(lead_min, horizon_days))["drive_id"]
            .n_unique()
        )
        out["tiers"][tier] = {
            "escalated": escalated,
            "caught": caught_tier,
            "precision": caught_tier / escalated if escalated else None,
        }
        if tier == "cordon":
            escalated_first = qualifying.group_by("drive_id").agg(
                pl.col("date").min().alias("first")
            )
            # cordon implies warn (its threshold is no lower), so its caught drives are a
            # subset of the warn-caught drives
            out["caught_then_escalated"] = caught_tier / n_caught if n_caught else None
    assert escalated_first is not None

    # Watchlist: warned, still within `horizon_days` of the warning, not yet escalated or failed.
    failure_by_drive = dict(zip(failures["drive_id"], failures["failure_date"], strict=True))
    escalated_by_drive = dict(
        zip(escalated_first["drive_id"], escalated_first["first"], strict=True)
    )
    delta = np.zeros(days + 1, dtype=np.int64)
    for drive, first in zip(warn_dates["drive_id"], warn_dates["first"], strict=True):
        stop = first + dt.timedelta(days=horizon_days)
        for cutoff_day in (escalated_by_drive.get(drive), failure_by_drive.get(drive)):
            if cutoff_day is not None and cutoff_day >= first:
                stop = min(stop, cutoff_day)
        lo, hi = max((first - start).days, 0), min((stop - start).days, days)
        if hi > lo:
            delta[lo] += 1
            delta[hi] -= 1
    watch = np.cumsum(delta)[:days]
    steady = watch[horizon_days:] if days > horizon_days else watch  # after the warm-up
    out["watchlist_mean"] = float(steady.mean())
    out["watchlist_peak"] = int(steady.max())
    return out
