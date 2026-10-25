"""Figures for the LSTM-against-LightGBM comparison (runbook 5.9 to 5.12).

Drawn from the reports `make train-sequence` and
`scripts/compare_sequence_vs_tree.py` already wrote; nothing is re-scored here.
Colours follow `late_hit_figures`: blue is LightGBM, orange is what the LSTM
adds (the blend, or drives only it catches), grey is context (the LSTM alone,
drives both catch, individual training runs).
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

from src.reporting.late_hit_figures import (  # noqa: E402
    BLUE,
    GRAY,
    GRID,
    ORANGE,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _heading,
    _style,
)

BUDGET_TEXT = {"0.0001": "1 in 10,000", "0.0005": "1 in 2,000", "0.001": "1 in 1,000"}


def plot_training_curves(histories: dict[str, Sequence[dict[str, float]]]) -> plt.Figure:
    """Validation average precision per epoch for every training run, with the
    epoch whose weights were kept marked."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for history in histories.values():
        epochs = [h["epoch"] for h in history]
        ap = [h["validation_average_precision"] for h in history]
        ax.plot(epochs, ap, color=GRAY, linewidth=1.5, solid_capstyle="round", zorder=2)
        best = int(np.argmax(ap))
        ax.scatter(
            [epochs[best]], [ap[best]], s=56, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=3
        )
    ax.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel("Validation average precision", color=TEXT_SECONDARY, fontsize=9.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(
        handles=[
            Line2D([], [], color=GRAY, linewidth=1.5, label=f"One training run ({len(histories)})"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=ORANGE,
                markeredgecolor=SURFACE,
                markersize=8,
                label="Best epoch (weights kept)",
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    _heading(
        fig,
        "Every LSTM run peaks within two epochs, then starts to memorize",
        "Four seeds and three regularized settings. Training loss kept falling in every run while "
        "validation precision fell, whatever the learning rate, dropout or size.",
    )
    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.12)
    return fig


def plot_recall_at_budgets(rows: Sequence[dict[str, Any]], budgets: Sequence[str]) -> plt.Figure:
    """Test recall at each false-alarm budget for LightGBM, the LSTM ensemble
    alone and their blend. `rows`: `name`, `kind` (tree / lstm / blend) and
    `recall_at_far_<budget>`."""
    colors = {"tree": BLUE, "lstm": GRAY, "blend": ORANGE}
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    x = np.arange(len(budgets))
    width = 0.26
    for k, row in enumerate(rows):
        values = [row[f"recall_at_far_{b}"] * 100 for b in budgets]
        offset = (k - (len(rows) - 1) / 2) * (width + 0.02)
        ax.bar(x + offset, values, width=width, color=colors[row["kind"]], linewidth=0)
        for i, v in enumerate(values):
            ax.text(
                x[i] + offset,
                v + 0.4,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT_PRIMARY,
            )
    ax.set_xticks(x, [f"{BUDGET_TEXT.get(b, b)}\nhealthy drives alerted" for b in budgets])
    ax.tick_params(axis="x", labelsize=9.5, colors=TEXT_PRIMARY)
    ax.set_ylabel("Failing drives caught (test)", color=TEXT_SECONDARY, fontsize=9.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    top = max(row[f"recall_at_far_{b}"] for row in rows for b in budgets) * 100
    ax.set_ylim(0, top * 1.25)
    ax.legend(
        handles=[Patch(facecolor=colors[r["kind"]], label=r["name"]) for r in rows],
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    _heading(
        fig,
        "Blended with LightGBM, the LSTM catches more failing drives for the same false alarms",
        "Recall at a fixed share of healthy drives alerted, test split (1,479 failing, 353,552 "
        "healthy). The blend weight was chosen on validation.",
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.97, bottom=0.15)
    return fig


def plot_catch_overlap(rows: Sequence[dict[str, Any]]) -> plt.Figure:
    """Failing test drives caught at each budget, split into caught by both,
    only by LightGBM and only by the LSTM ensemble. `rows`:
    `false_alarm_budget`, `both`, `only_tree`, `only_ensemble`."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]
    parts = (
        ("both", GRAY, "Caught by both"),
        ("only_tree", BLUE, "Only LightGBM"),
        ("only_ensemble", ORANGE, "Only the LSTM"),
    )
    peak = max(sum(r[k] for k, _, _ in parts) for r in rows) if rows else 1
    for row, pos in zip(rows, y, strict=True):
        left = 0.0
        for key, color, _ in parts:
            value = row[key]
            ax.barh(pos, value, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
            if value >= peak * 0.06:
                ax.text(
                    left + value / 2,
                    pos,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if color != GRAY else TEXT_PRIMARY,
                )
            left += value
        ax.text(
            left + peak * 0.01,
            pos,
            f"{int(left)} drives",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.set_yticks(
        y,
        [
            f"{BUDGET_TEXT.get(format(r['false_alarm_budget'], 'g'), r['false_alarm_budget'])}"
            "\nhealthy alerted"
            for r in rows
        ],
    )
    ax.tick_params(axis="y", labelsize=9.5, colors=TEXT_PRIMARY)
    ax.set_xlim(0, peak * 1.15)
    ax.set_xlabel("Failing test drives caught by either model", color=TEXT_SECONDARY, fontsize=9.5)
    ax.legend(
        handles=[Patch(facecolor=c, label=t) for _, c, t in parts],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
    )
    first = rows[0] if rows else {"both": 0, "only_tree": 0}
    tree_total = first["both"] + first["only_tree"]
    _heading(
        fig,
        "The two models catch different failing drives",
        "Each model alone at the same false-alarm rate. At the strictest rate only "
        f"{first['both']} of LightGBM's {tree_total} catches are also the LSTM's, which is what "
        "a blend can use.",
    )
    fig.subplots_adjust(top=0.72, left=0.14, right=0.97, bottom=0.16)
    return fig


def plot_blend_gain(rows: Sequence[dict[str, Any]], budgets: Sequence[str]) -> plt.Figure:
    """Blend minus LightGBM recall, in points, with its 95% paired bootstrap
    interval, per LSTM run and for the ensemble; one panel per budget.
    `rows`: `name`, `gain` {budget: float}, `ci` {budget: [lo, hi]}."""
    fig, axes = plt.subplots(1, len(budgets), figsize=(9.2, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(rows))[::-1]
    lo_all = min(r["ci"][b][0] for r in rows for b in budgets) * 100
    hi_all = max(r["ci"][b][1] for r in rows for b in budgets) * 100
    for ax, budget in zip(axes, budgets, strict=True):
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.2)
        for row, pos in zip(rows, y, strict=True):
            lo, hi = (v * 100 for v in row["ci"][budget])
            gain = row["gain"][budget] * 100
            ensemble = row["name"].startswith("Ensemble")
            color = ORANGE if ensemble else GRAY
            ax.plot([lo, hi], [pos, pos], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter(
                [gain],
                [pos],
                s=56 if ensemble else 40,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
        ax.set_title(
            f"{BUDGET_TEXT.get(budget, budget)} healthy alerted",
            fontsize=9.5,
            color=TEXT_PRIMARY,
            loc="left",
        )
        ax.set_xlim(min(lo_all, -1) - 0.5, hi_all + 0.5)
        ax.set_xlabel("Recall gain, points", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_yticks(y, [r["name"] for r in rows])
    axes[0].tick_params(axis="y", labelsize=9.5, colors=TEXT_PRIMARY)
    _heading(
        fig,
        "The blend's gain is positive for every seed; its size is small",
        "Blend minus LightGBM recall on test, with the 95% paired drive-level bootstrap interval. "
        "Intervals share test drives, so read them as optimistic.",
    )
    fig.subplots_adjust(top=0.76, left=0.13, right=0.98, bottom=0.16, wspace=0.12)
    return fig
