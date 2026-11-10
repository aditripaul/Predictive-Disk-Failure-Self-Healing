"""Figures of the chronological split run (runbook sections 1.1.8 and 1.1.9).

Five figures, drawn from files the run already wrote (no model is scored, no test row is read):

- `chrono_split_timeline.png`: the chronological split with gaps (runbook 1.1.9) on a date
  axis. Dates and failing-drive counts are the ones
  in runbook 1.1.8 / 1.1.9 (counted from the 30-day label table on 2026-11-10).
- `chrono_precision_recall.png`: precision against recall at the recall levels of
  `data/audit/stack_chrono_h30/tables.md` (test, drive level, mean of 3 seeds) for Stage 1 alone,
  the LightGBM and Cox Stage 2, and their stack, with the goal's region shaded.
- `chrono_goal_validation_vs_test.png`: precision at the 10% recall level on validation and on
  test, per Stage 2 kind, a dot per seed, from `experiment_two_stage.json`.
- `chrono_base_rate_transfer.png`: base rate and precision at the primary threshold, validation
  against test, for the drive-holdout design and the chronological split.
- `chrono_true_positives_isolated.png`: how many of the failing test drives (count and share) each
  stage isolates: the Stage 1 net, then Stage 1 alone, the LightGBM and Cox Stage 2 and the stack at
  the 10, 20 and 35% recall levels, from `experiment_two_stage.json` (mean of 3 seeds).

    uv run python scripts/plot_chrono_split.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

D = dt.date
# The reference palette (dataviz skill references/palette.md), light mode. Roles keep one colour in
# every figure: train blue, validation orange, test aqua. Three Stage 2 kinds reuse the same
# slots, so the two uses are never on one chart.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
UNUSED = "#d9d8d3"

STACK_DIR = Path("data/audit/stack_chrono_h30")
REPORT = Path("data/audit/data_quality_reports/model_evaluation_report_chrono_h30.json")
OUT = Path("docs/figures/chrono_split")
GOAL_PRECISION, GOAL_RECALL = 68.0, 10.0


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def _span(ax: plt.Axes, y: float, h: float, start: dt.date, end: dt.date, **kw) -> None:
    """A bar from `start` to `end` inclusive, with a half-day of surface at each end."""
    left = mdates.date2num(start) + 0.4
    width = mdates.date2num(end) - mdates.date2num(start) + 1 - 0.8
    ax.barh(y, width, left=left, height=h, zorder=3, linewidth=0, **kw)


def timeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)

    def hatched(start, end):
        _span(ax, 0, 0.5, start, end, color=UNUSED, hatch="////", edgecolor=SURFACE)

    _span(ax, 0, 0.5, D(2026, 1, 1), D(2026, 4, 20), color=BLUE)
    hatched(D(2026, 4, 21), D(2026, 5, 20))
    _span(ax, 0, 0.5, D(2026, 5, 21), D(2026, 6, 16), color=ORANGE)
    hatched(D(2026, 6, 17), D(2026, 7, 16))
    _span(ax, 0, 0.5, D(2026, 7, 17), D(2026, 8, 31), color=AQUA)
    hatched(D(2026, 9, 1), D(2026, 9, 30))
    ax.text(
        D(2026, 3, 1),
        0,
        "train: 2,018 failing",
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
    )
    ax.text(D(2026, 5, 5), 0, "gap\n30 d", ha="center", va="center", fontsize=8, color=INK_2)
    ax.text(
        D(2026, 6, 3), 0, "validation\n684", ha="center", va="center", fontsize=8, color="white"
    )
    ax.text(D(2026, 7, 2), 0, "gap\n30 d", ha="center", va="center", fontsize=8, color=INK_2)
    ax.text(
        D(2026, 8, 8),
        0,
        "test: 864 failing",
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
    )
    for day, label, align in (
        (D(2026, 5, 20), "train_end 05-20 ", "right"),
        (D(2026, 6, 16), " validation_end 06-16", "left"),
    ):
        ax.plot([day, day], [-0.28, -0.42], color=INK_2, linewidth=1)
        ax.text(day, -0.46, label, ha=align, va="top", fontsize=7.5, color=INK_2)
    ax.text(D(2026, 9, 15), 0.32, "no 30-day label yet", ha="center", fontsize=7.5, color=INK_2)
    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.6)
    ax.set_xlim(D(2026, 1, 1), D(2026, 9, 30))
    ax.set_xlabel("Row date, 2026 (Q4 2025 is feature history only)", fontsize=9, color=INK_2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(
        "Chronological split with 30-day gaps (runbook 1.1.9)",
        loc="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    ax.legend(
        handles=[
            Patch(color=BLUE, label="train"),
            Patch(color=ORANGE, label="validation"),
            Patch(color=AQUA, label="test"),
            Patch(facecolor=UNUSED, hatch="////", edgecolor=SURFACE, label="no row used"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    _save(fig, path)


def _levels(table: str, name: str) -> dict[str, list[float]]:
    """{level: [stage1 P, stage1 R, cascade P, cascade R]} from a `## Net ...` table."""
    start = table.index(f"## {name}\n")
    end = table.find("\n## ", start + 5)
    rows: dict[str, list[float]] = {}
    for line in table[start:end].splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if re.fullmatch(r"\d+%", cells[0]):
            nums = [float(x) for x in re.findall(r"[\d.]+", cells[1] + " " + cells[2])]
            rows[cells[0]] = nums
    return rows


def precision_recall(path: Path) -> None:
    table = (STACK_DIR / "tables.md").read_text()
    lgbm, cox, stack = (_levels(table, n) for n in ("Net 80%", "Net 80% cox", "Net 80% stack"))
    levels = list(lgbm)
    series = [
        ("Stage 1 alone", [(lgbm[k][1], lgbm[k][0]) for k in levels], MUTED, "--"),
        ("LightGBM Stage 2", [(lgbm[k][3], lgbm[k][2]) for k in levels], BLUE, "-"),
        ("Cox Stage 2", [(cox[k][3], cox[k][2]) for k in levels], ORANGE, "-"),
        ("Stack (LightGBM + Cox)", [(stack[k][3], stack[k][2]) for k in levels], AQUA, "-"),
    ]
    fig, (ax, zoom) = plt.subplots(
        1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for panel, (xmax, ymin, ymax) in ((ax, (90, 0, 78)), (zoom, (30, 40, 72))):
        _style(panel)
        panel.grid(axis="x", color=GRID, linewidth=1)
        panel.fill_between(
            [GOAL_RECALL, xmax], GOAL_PRECISION, ymax, color=AQUA, alpha=0.10, zorder=1
        )
        for name, pts, colour, style in series:
            xs, ys = zip(*pts, strict=True)
            panel.plot(
                xs,
                ys,
                color=colour,
                linewidth=2,
                linestyle=style,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=name,
                zorder=4,
            )
        panel.set_xlim(0, xmax)
        panel.set_ylim(ymin, ymax)
        panel.set_xlabel("Recall of failing drives (%), test", fontsize=9, color=INK_2)
    ax.set_ylabel("Precision (%), test", fontsize=9, color=INK_2)
    ax.text(
        88,
        76.3,
        f"goal: precision >= {GOAL_PRECISION:.0f}% at recall >= {GOAL_RECALL:.0f}%",
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set_title("All recall levels", loc="left", fontsize=10, color=INK_2)
    zoom.text(29.5, 71.4, "goal region", ha="right", va="top", fontsize=8.5, color=INK_2)
    zoom.axvline(GOAL_RECALL, color=INK_2, linewidth=1, zorder=2)
    zoom.annotate(
        "stack, 10% level:\n55.4% at 16.2% recall",
        xy=(16.2, 55.4),
        xytext=(10.6, 43.5),
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 1},
    )
    zoom.set_title("Zoom on the goal (recall 0 to 30%)", loc="left", fontsize=10, color=INK_2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Stage 2 variants at each recall level: none reaches the goal",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.095,
        "Mean of 3 seeds, thresholds set on validation, 0.8 net. Points are the recall "
        "levels 5, 10, 20, 35, 50, 60, 70 and 80%.",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    _save(fig, path)


def goal_validation_vs_test(path: Path) -> None:
    runs = json.loads((STACK_DIR / "experiment_two_stage.json").read_text())["runs"]
    kinds = (
        ("lightgbm", "LightGBM Stage 2"),
        ("cox", "Cox Stage 2"),
        ("stack", "Stack (LightGBM + Cox)"),
    )
    val: dict[str, list[float]] = {k: [] for k, _ in kinds}
    test: dict[str, list[float]] = {k: [] for k, _ in kinds}
    for run in sorted(runs, key=lambda r: r["seed"]):
        if run["stage2_kind"] not in val or abs(run["candidate_recall"] - 0.8) > 1e-9:
            continue
        for level in run["two_stage"]:
            if abs(level["target_recall"] - 0.10) < 1e-9:
                val[run["stage2_kind"]].append(100 * level["validation"]["precision"])
                test[run["stage2_kind"]].append(100 * level["test"]["precision"])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _style(ax)
    width, x = 0.11, np.arange(len(kinds))
    for offset, data, colour, label in (
        (-0.08, val, ORANGE, "validation"),
        (0.08, test, AQUA, "test"),
    ):
        means = [float(np.mean(data[k])) for k, _ in kinds]
        ax.bar(x + offset, means, width=width, color=colour, label=label, zorder=3)
        for i, (k, _) in enumerate(kinds):
            ax.scatter(
                [x[i] + offset] * len(data[k]),
                data[k],
                s=26,
                color=INK,
                zorder=5,
                edgecolors=SURFACE,
                linewidths=1.5,
            )
            ax.text(
                x[i] + offset,
                max(data[k]) + 2.2,
                f"{means[i]:.1f}%",
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    ax.axhline(GOAL_PRECISION, color=INK, linewidth=1.2, zorder=4)
    ax.text(
        -0.45,
        GOAL_PRECISION + 1.2,
        f"goal {GOAL_PRECISION:.0f}%",
        ha="left",
        fontsize=8.5,
        color=INK,
    )
    ax.set_xlim(-0.5, len(kinds) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in kinds], fontsize=9, color=INK)
    ax.set_ylim(0, 76)
    ax.set_ylabel("Precision at the 10% recall level (%)", fontsize=9, color=INK_2)
    ax.set_title(
        "The goal fails on validation as well as on test",
        loc="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, ncol=2)
    ax.text(
        0,
        -0.17,
        "Bars: mean of 3 seeds. Dots: each seed. Recall at this level is 11 to 18% "
        "(runbook 1.1.9).",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_2,
    )
    _save(fig, path)


def base_rate_transfer(path: Path) -> None:
    report = json.loads(REPORT.read_text())
    new_rate = (
        100 * report["validation_drive_level"]["fleet_failure_rate"],
        100 * report["test_drive_level"]["fleet_failure_rate"],
    )
    new_precision = (
        100 * report["validation_drive_level"]["precision"],
        100 * report["test_drive_level"]["precision"],
    )
    # Drive-holdout design (runbook 1.1.4 and 1.1.5, frozen run c459046e): 0.589% of validation
    # drives and 0.300% of test drives fail; the primary threshold gives 75.0% and 59.0%.
    old_rate, old_precision = (0.589, 0.300), (75.0, 59.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    panels = (
        ("Failing drives (base rate, %)", old_rate, new_rate, "{:.2f}%"),
        ("Precision at the primary threshold (%)", old_precision, new_precision, "{:.1f}%"),
    )
    for ax, (ylabel, old, new, fmt) in zip(axes, panels, strict=True):
        _style(ax)
        x = np.arange(2)
        for offset, idx, colour, label in (
            (-0.11, 0, ORANGE, "validation"),
            (0.11, 1, AQUA, "test"),
        ):
            vals = [old[idx], new[idx]]
            ax.bar(x + offset, vals, width=0.19, color=colour, label=label, zorder=3)
            for i, v in enumerate(vals):
                ax.text(
                    x[i] + offset,
                    v + max(old + new) * 0.02,
                    fmt.format(v),
                    ha="center",
                    fontsize=8.5,
                    color=INK,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(["Drive holdout", "Chronological\nwith gaps"], fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
        ax.set_ylim(0, max(old + new) * 1.18)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        "Validation no longer overstates test; the base rates are now close",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


def true_positives_isolated(path: Path) -> None:
    """Failing test drives isolated by each stage, as counts (bottom axis) and shares (top axis)."""
    runs = json.loads((STACK_DIR / "experiment_two_stage.json").read_text())["runs"]
    net08 = [
        r for r in sorted(runs, key=lambda r: r["seed"]) if abs(r["candidate_recall"] - 0.8) < 1e-9
    ]
    failing = net08[0]["two_stage"][0]["test"]["failing_drive_count"]

    def mean_caught(kind: str, level: float, key: str) -> float:
        picked = [r for r in net08 if r["stage2_kind"] == kind]
        vals = []
        for run in picked:
            rows = run["stage1"] if key == "stage1" else run["two_stage"]
            row = next(x for x in rows if abs(x["target_recall"] - level) < 1e-9)
            vals.append(row["test"]["caught_drive_count"])
        return float(np.mean(vals))

    pool = (
        float(np.mean([r["ceiling"]["test_recall"] for r in net08 if r["stage2_kind"] == "stack"]))
        * failing
    )
    levels = (0.10, 0.20, 0.35)
    models = (
        ("Stage 1 alone", "lightgbm", "stage1", MUTED),
        ("LightGBM Stage 2", "lightgbm", "stage2", BLUE),
        ("Cox Stage 2", "cox", "stage2", ORANGE),
        ("Stack (LightGBM + Cox)", "stack", "stage2", AQUA),
    )
    rows: list[tuple[str, float, str]] = [
        ("All failing test drives", float(failing), UNUSED),
        ("Isolated by the Stage 1 net", pool, INK_2),
    ]
    group_starts: dict[int, str] = {}
    for level in levels:
        group_starts[len(rows)] = f"Flagged at the {level:.0%} recall level"
        for label, kind, which, colour in models:
            rows.append((label, mean_caught(kind, level, which), colour))

    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    y, ys, labels = 0.0, [], []
    for i, (label, count, colour) in enumerate(rows):
        if i in group_starts:
            y += 0.55
            ax.text(
                -8,
                y - 0.45,
                group_starts[i],
                ha="left",
                va="bottom",
                fontsize=9,
                color=INK,
                fontweight="bold",
            )
        ax.barh(y, count, height=0.62, color=colour, zorder=3)
        share, of_pool = 100 * count / failing, 100 * count / pool
        text = f"{count:,.0f}  ({share:.1f}% of all failing"
        text += f", {of_pool:.1f}% of the Stage 1 pool)" if i >= 2 else ")"
        ax.text(count + 8, y, text, ha="left", va="center", fontsize=8.5, color=INK)
        ys.append(y)
        labels.append(label)
        y += 1.0
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, failing * 1.62)
    ax.set_xlabel("Failing drives isolated (count), test", fontsize=9, color=INK_2)
    top = ax.secondary_xaxis(
        "top", functions=(lambda x: 100 * x / failing, lambda x: x * failing / 100)
    )
    top.set_xlabel("Share of all failing test drives (%)", fontsize=9, color=INK_2)
    top.tick_params(colors=INK_2, labelsize=9, length=0)
    top.spines["top"].set_color(GRID)
    top.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Failing drives isolated by Stage 1, Stage 2 and the stack",
        loc="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        0,
        -0.1,
        f"{failing} failing of 368,124 test drives. Mean of 3 seeds, 0.8 net, thresholds set on "
        "validation for each recall level.\nThe Stage 1 pool is the failing drives the net passes; "
        "no Stage 2 model can catch more than it.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_2,
        va="top",
    )
    _save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    timeline(args.out / "chrono_split_timeline.png")
    precision_recall(args.out / "chrono_precision_recall.png")
    goal_validation_vs_test(args.out / "chrono_goal_validation_vs_test.png")
    base_rate_transfer(args.out / "chrono_base_rate_transfer.png")
    true_positives_isolated(args.out / "chrono_true_positives_isolated.png")


if __name__ == "__main__":
    main()
