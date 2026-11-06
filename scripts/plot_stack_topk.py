"""Plot how many more failing drives Cox and the stacked Stage 2 catch than the
LightGBM Stage 2 when exactly K drives are flagged (runbook section 1.1.4a).

Reads the saved `top_k` results of the stack runs (`data/audit/stack_q3_h<H>/
experiment_two_stage.json`, net 0.8, 3 seeds). No score file and no new look at the
test set is needed: the K drives are the highest-scored ones of each model, so
no threshold is involved. One panel per horizon; a dot per seed (the difference is
taken within a seed, against the LightGBM Stage 2 of the same seed) and a line through
their mean. Zero is the LightGBM Stage 2.

    uv run python scripts/plot_stack_topk.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HORIZONS = (30, 44, 60)
NET = 0.8
SERIES = (("stack", "Stack (LightGBM + Cox)", "#2a78d6"), ("cox", "Cox", "#eb6834"))
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def caught_by_k(path: Path) -> dict[str, dict[int, list[int]]]:
    """{kind: {K: caught drives per seed}} for the net-0.8 runs, seeds in order."""
    runs = json.loads(path.read_text())["runs"]
    out: dict[str, dict[int, list[int]]] = {}
    for run in sorted(runs, key=lambda r: r["seed"]):
        if run["candidate_recall"] != NET or "top_k" not in run:
            continue
        for row in run["top_k"]["cascade"]:
            out.setdefault(run["stage2_kind"], {}).setdefault(row["k"], []).append(row["caught"])
    return out


def differences(caught: dict[str, dict[int, list[int]]], kind: str) -> dict[int, np.ndarray]:
    """Per seed, `kind` minus the LightGBM Stage 2, at each K."""
    return {
        k: np.asarray(values) - np.asarray(caught["lightgbm"][k])
        for k, values in caught[kind].items()
    }


def draw(audit_dir: Path, out: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    for ax, horizon in zip(axes, HORIZONS, strict=True):
        caught = caught_by_k(audit_dir / f"stack_q3_h{horizon}" / "experiment_two_stage.json")
        ks = sorted(caught["lightgbm"])
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK_2, linewidth=1.2, zorder=1)
        for offset, (kind, label, color) in zip((0.96, 1.04), SERIES, strict=True):
            diff = differences(caught, kind)
            x = np.asarray(ks, dtype=float) * offset
            for seed in range(len(diff[ks[0]])):
                ax.plot(
                    x,
                    [diff[k][seed] for k in ks],
                    "o",
                    color=color,
                    markersize=7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    alpha=0.9,
                    zorder=3,
                    label=None,
                )
            ax.plot(
                x,
                [diff[k].mean() for k in ks],
                color=color,
                linewidth=2,
                zorder=2,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xticks(ks, [f"{k:,}" for k in ks], fontsize=9)
        ax.minorticks_off()
        ax.set_xlabel("Drives flagged (K)", color=INK_2)
        ax.set_title(f"{horizon}-day", loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2)
    axes[0].set_ylabel("Failing drives caught,\nminus the LightGBM Stage 2", color=INK_2)
    axes[0].text(
        0.02,
        0.02,
        "LightGBM Stage 2 = 0",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color=INK_2,
        va="bottom",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", frameon=False, fontsize=9.5)
    fig.suptitle(
        "Failing drives caught at the same number flagged, against the LightGBM Stage 2",
        x=0.01,
        y=0.99,
        ha="left",
        fontsize=12.5,
    )
    fig.text(
        0.01,
        0.925,
        "Q3 2026 test, 0.8 net; each dot is one seed (3), the line is their mean; "
        "the K drives are the highest-scored, so no threshold is involved",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audit"))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/figures/q3_2026/stack_topk_caught.png")
    )
    args = parser.parse_args()
    draw(args.audit_dir, args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
