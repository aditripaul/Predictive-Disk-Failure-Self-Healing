"""Figures showing that many "false alarms" were early alerts, not wrong ones.

The goal is pinned to a 30-day label (runbook section 7): an alert on a drive
that is healthy for 30 days counts as a false alarm. Some of those drives then
fail in days 31 to 60. These figures put that on the page without changing the
30-day figure, which is always shown beside it, never replaced by it
(`src/agent/false_positives.py` has the numbers; this is only the drawing).

Colour does one job: blue is "counted correct by the 30-day label", orange is
"failed later" (the early alerts), grey is "never failed in the data". Orange
that falls after day 60 is also hatched, so the two orange bars stay distinct
without colour. The blue/orange pair passes the dataviz palette validator
(adjacent CVD delta E 24.7, target 8). Text is always in text colours, never in
a series colour.

Headless matplotlib, like the rest of `src/reporting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e7e6e2"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#a8a7a1"
BAND = "#2a78d6"  # drawn at low opacity: the 30-day label window


def _outcome_text(horizon: int) -> dict[str, str]:
    return {
        "correct": f"Failed within {horizon} days\n(counted correct)",
        "early": f"Failed in days {horizon + 1}-{2 * horizon}\n(counted a false alarm: early)",
        "later": f"Failed after day {2 * horizon}\n(seen to end of data)",
        "removed": "Left service,\nno failure recorded",
        "active": "Still in service\nat end of data",
    }


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _heading(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.text(
        0.01, 0.985, title, ha="left", va="top", fontsize=13, color=TEXT_PRIMARY, weight="bold"
    )
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=TEXT_SECONDARY)


def plot_alert_outcomes(
    counts: dict[str, int], *, horizon_days: int, tier: str, snapshots: int
) -> plt.Figure:
    """One bar per outcome, labelled with its count and share of the alerts."""
    from src.agent.false_positives import OUTCOME_ORDER

    names = _outcome_text(horizon_days)
    total = sum(counts.get(k, 0) for k in OUTCOME_ORDER)
    false_alarms = total - counts.get("correct", 0)
    early = counts.get("early", 0) + counts.get("later", 0)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    _style(ax)
    ax.xaxis.grid(False)
    y = np.arange(len(OUTCOME_ORDER))[::-1]
    colors = {"correct": BLUE, "early": ORANGE, "later": ORANGE, "removed": GRAY, "active": GRAY}
    peak = max((counts.get(k, 0) for k in OUTCOME_ORDER), default=1) or 1
    for key, pos in zip(OUTCOME_ORDER, y, strict=True):
        value = counts.get(key, 0)
        ax.barh(
            pos,
            value,
            height=0.5,
            color=colors[key],
            hatch="////" if key == "later" else None,
            edgecolor=SURFACE if key == "later" else colors[key],
            linewidth=0,
        )
        share = value / total if total else 0.0
        ax.text(
            value + peak * 0.015,
            pos,
            f"{value:,}  ({share:.1%})",
            va="center",
            ha="left",
            fontsize=9.5,
            color=TEXT_PRIMARY,
        )
    ax.set_yticks(y, [names[k] for k in OUTCOME_ORDER], color=TEXT_PRIMARY)
    ax.tick_params(axis="y", labelsize=9.5, colors=TEXT_PRIMARY)
    ax.set_xlim(0, peak * 1.28)
    ax.set_xticks([])
    share_early = early / false_alarms if false_alarms else 0.0
    _heading(
        fig,
        f'{early:,} of the {false_alarms:,} "false alarms" were early alerts: the drive failed',
        f"Drive-snapshots flagged at {tier} or above on {snapshots} weekly dates, each followed to "
        f"the end of the data. {share_early:.1%} of the alerts the {horizon_days}-day label "
        "counts as false went on to fail.",
    )
    fig.subplots_adjust(top=0.82, left=0.27, right=0.97, bottom=0.05)
    return fig


def plot_lead_time_histogram(days_to_failure: Sequence[int], *, horizon_days: int) -> plt.Figure:
    """Days from an alert to the drive's failure, for alerts on drives that failed.

    One bar per day, so the label boundary falls exactly between day `horizon`
    (counted correct) and day `horizon + 1` (counted a false alarm). Blue is
    inside the label window; orange is the early alerts the label calls wrong;
    the last bar gathers everything past `2 * horizon_days`."""
    days = np.asarray(list(days_to_failure), dtype=int)
    limit = 2 * horizon_days
    per_day = np.bincount(np.clip(days[days <= limit], 1, limit), minlength=limit + 1)
    later = int((days > limit).sum())

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    _style(ax)
    x = np.arange(1, limit + 1)
    ax.bar(
        x,
        per_day[1:],
        width=0.82,
        color=[BLUE if d <= horizon_days else ORANGE for d in x],
        linewidth=0,
    )
    later_x = limit + 3
    if later:
        ax.bar(
            later_x,
            later,
            width=0.82 * 2,
            color=ORANGE,
            hatch="////",
            edgecolor=SURFACE,
            linewidth=0,
        )
        ax.text(
            later_x, later, f"{later:,}", ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY
        )
    top = max(int(per_day.max()), later, 1)
    boundary = horizon_days + 0.5
    ax.axvline(boundary, color=TEXT_SECONDARY, linewidth=1.2)
    ax.text(
        boundary - 1,
        top * 1.17,
        f"{horizon_days}-day label boundary",
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    ax.set_ylim(0, top * 1.22)
    ax.set_xlim(0, later_x + 3)
    ticks = [1, *range(10, limit + 1, 10)]
    ax.set_xticks([*ticks, later_x], [*[str(t) for t in ticks], f">{limit}"])
    ax.set_xlabel("Days from the alert to the drive's failure", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Alerts", color=TEXT_SECONDARY, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label=f"Within {horizon_days} days: counted correct"),
            Patch(
                facecolor=ORANGE,
                label=f"After {horizon_days} days: counted a false alarm, but the drive failed",
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    n_in = int((days <= horizon_days).sum())
    n_after = int((days > horizon_days).sum())
    _heading(
        fig,
        "The alerts do not stop being right at the label boundary",
        f"Of {len(days):,} alerts on drives that went on to fail, {n_in:,} fell inside "
        f"{horizon_days} days and {n_after:,} came earlier than that.",
    )
    fig.subplots_adjust(top=0.84, left=0.08, right=0.97, bottom=0.12)
    return fig


def plot_alert_timelines(
    rows: Sequence[dict[str, Any]], *, horizon_days: int, total_early: int
) -> plt.Figure:
    """One row per early alert: the alert at day 0, the failure at day N.

    `rows` carry `label` (drive model and a short id) and `lead_days`; they are
    drawn in the order given. The shaded band is the label window, so a failure
    to its right is the early alert the label counts as a false alarm."""
    n = max(len(rows), 1)
    height = max(4.4, 0.24 * n + 2.8)
    fig, ax = plt.subplots(figsize=(9.2, height))
    _style(ax)
    ax.xaxis.grid(False)
    ax.axvspan(0, horizon_days, color=BAND, alpha=0.10, linewidth=0)
    y = np.arange(len(rows))[::-1]
    for row, pos in zip(rows, y, strict=True):
        lead = float(row["lead_days"])
        ax.plot([0, lead], [pos, pos], color=GRAY, linewidth=1.5, solid_capstyle="round", zorder=2)
        ax.scatter([0], [pos], s=48, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.8, zorder=3)
        ax.scatter([lead], [pos], s=56, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(lead + 1.0, pos, f"{lead:.0f} d", va="center", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_yticks(y, [str(r["label"]) for r in rows])
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT_PRIMARY)
    limit = max([horizon_days * 2, *[float(r["lead_days"]) for r in rows]])
    ax.set_xlim(-2, limit + 8)
    ax.set_ylim(-0.8, len(rows) + 0.9)
    ax.set_xticks(list(range(0, int(limit) + 1, 10)))
    ax.set_xlabel("Days after the alert", color=TEXT_SECONDARY, fontsize=9.5)
    ax.text(
        horizon_days / 2,
        len(rows) + 0.25,
        f"{horizon_days}-day label window",
        ha="center",
        va="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=SURFACE,
                markeredgecolor=BLUE,
                markeredgewidth=1.8,
                markersize=7,
                label="Alert raised",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=ORANGE,
                markeredgecolor=SURFACE,
                markersize=8,
                label="Drive failed",
            ),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, (len(rows) + 0.25 + 0.8) / (len(rows) + 1.7)),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    shown = f"{len(rows)} of {total_early:,}" if len(rows) < total_early else f"All {len(rows)}"
    _heading(
        fig,
        "Each of these alerts was early: the drive failed after the label window closed",
        f"{shown} drives alerted but healthy for {horizon_days} days, then failed. "
        "Ordered by lead time; sampled evenly across the range.",
    )
    fig.subplots_adjust(top=1 - 1.35 / height, left=0.26, right=0.97, bottom=0.8 / height)
    return fig


def plot_precision_by_tier(rows: Sequence[dict[str, Any]], *, horizon_days: int) -> plt.Figure:
    """Precision per action tier by the label, and with days `horizon+1` to
    `2*horizon` credited. `rows`: `tier`, `alerts`, `precision_label`,
    `precision_credited`, in tier order (mildest first)."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    x = np.arange(len(rows))
    width = 0.34
    label = [r["precision_label"] * 100 for r in rows]
    credited = [r["precision_credited"] * 100 for r in rows]
    ax.bar(x - width / 2 - 0.01, label, width=width, color=BLUE, linewidth=0)
    ax.bar(x + width / 2 + 0.01, credited, width=width, color=ORANGE, linewidth=0)
    for i, (a, b) in enumerate(zip(label, credited, strict=True)):
        ax.text(
            i - width / 2,
            a + 1,
            f"{a:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
        ax.text(
            i + width / 2,
            b + 1,
            f"{b:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    ax.set_xticks(x, [f"{r['tier']} and above\n{r['alerts']:,} alerts" for r in rows])
    ax.tick_params(axis="x", labelsize=9.5, colors=TEXT_PRIMARY)
    ax.set_ylabel("Precision of the alerts", color=TEXT_SECONDARY, fontsize=9.5)
    top = max([*credited, 1.0])
    ax.set_ylim(0, top * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label=f"Failed within {horizon_days} days (the label)"),
            Patch(facecolor=ORANGE, label=f"Failed within {2 * horizon_days} days"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    _heading(
        fig,
        "Followed for 60 days, every tier's alerts are more often right",
        "Same alerts, two horizons. The gain grows from warn to migrate; drain rests on "
        f"{rows[-1]['alerts'] if rows else 0} alerts.",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.97, bottom=0.14)
    return fig


def plot_later_failure_rate(
    rows: Sequence[dict[str, Any]], *, background: float, horizon_days: int
) -> plt.Figure:
    """Share of each tier's label-false alerts that failed in days
    `horizon+1` to `2*horizon`, against the share among every healthy-by-label
    drive-snapshot (`background`). `rows`: `tier`, `false_alarms`,
    `failed_later_share`."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    x = np.arange(len(rows))
    shares = [r["failed_later_share"] * 100 for r in rows]
    ax.bar(x, shares, width=0.5, color=ORANGE, linewidth=0)
    for i, r in enumerate(rows):
        lift = r["failed_later_share"] / background if background else 0.0
        ax.text(
            i,
            shares[i] + 0.6,
            f"{shares[i]:.1f}%\n{lift:,.0f}x background",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    ax.axhline(background * 100, color=TEXT_SECONDARY, linewidth=1.2)
    ax.set_xticks(x, [f"{r['tier']} and above\n{r['false_alarms']:,} false alarms" for r in rows])
    ax.tick_params(axis="x", labelsize=9.5, colors=TEXT_PRIMARY)
    ax.set_ylabel(
        f"Failed in days {horizon_days + 1}-{2 * horizon_days}", color=TEXT_SECONDARY, fontsize=9.5
    )
    ax.set_ylim(0, max([*shares, 1.0]) * 1.35)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _heading(
        fig,
        'A "false alarm" is far likelier than a typical drive to fail soon after',
        f"Share of each tier's {horizon_days}-day false alarms that failed in the next "
        f"{horizon_days} days. The grey line at the base is the same share across every drive "
        f"the label calls healthy: {background:.2%}.",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.97, bottom=0.15)
    return fig


def plot_alert_profile(
    profile: dict[str, dict[str, float]], *, attributes: dict[str, str], counts: dict[str, int]
) -> plt.Figure:
    """Median value of each attribute at the moment of the alert, per outcome
    group (`correct`, `failed_later`, `never_failed`). `attributes` maps a
    column to its display name; one panel per attribute."""
    groups = ("correct", "failed_later", "never_failed")
    names = {
        "correct": "Failed within 30 days",
        "failed_later": "Failed later",
        "never_failed": "Never failed in the data",
    }
    colors = {"correct": BLUE, "failed_later": ORANGE, "never_failed": GRAY}
    fig, axes = plt.subplots(1, len(attributes), figsize=(9.2, 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (column, title) in zip(axes, attributes.items(), strict=True):
        _style(ax)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        values = [profile[g][column] for g in groups]
        ax.bar(np.arange(3), values, width=0.6, color=[colors[g] for g in groups], linewidth=0)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_ylim(0, max([*values, 1.0]) * 1.2)
    fig.legend(
        handles=[
            Patch(facecolor=colors[g], label=f"{names[g]} ({counts.get(g, 0):,} alerts)")
            for g in groups
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    _heading(
        fig,
        "Drives that failed later had fewer bad sectors when they were alerted",
        "Median at the moment of the alert, cordon and above. Consistent with catching the same "
        "decline earlier; the counts alone do not separate the three groups.",
    )
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.14, wspace=0.25)
    return fig
