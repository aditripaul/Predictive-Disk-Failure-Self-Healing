"""Draw the two-stage funnel of the current 30-day model on Q3 2026.

The counts are the published drive-level results of frozen run `c459046e`
(runbook sections 1.1.4, 1.1.5 and 1.9; `docs/q3_horizon_and_cascade_run.md`
section 7.5): failing drives reached and healthy drives flagged at the Stage 1
net and at each operating point of the combined score. No data is read and no
test look is made; the numbers are typed in from those tables.

    uv run python scripts/plot_cascade_funnel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAILING_TOTAL = 1_105
HEALTHY_TOTAL = 367_583

# (label with threshold, failing reached, healthy flagged), loosest first.
STAGES = [
    ("Stage 1 net\n(cutoff 0.235)", 883, 22_463),
    ("Warn\n(0.653)", 673, 3_208),
    ("Cordon\n(0.836)", 469, 877),
    ("Primary threshold\n(0.919)", 138, 96),
    ("Migrate = drain\n(0.926, fallback)", 2, 4),
]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
FAILING = "#2a78d6"
HEALTHY = "#eb6834"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("docs/figures/q3_2026/cascade_funnel_h30.png")
    )
    args = parser.parse_args()

    labels = [s[0] for s in STAGES][::-1]
    failing = [s[1] for s in STAGES][::-1]
    healthy = [s[2] for s in STAGES][::-1]
    y = range(len(labels))

    fig, (ax_f, ax_h, ax_p) = plt.subplots(
        1,
        3,
        figsize=(13, 4.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]},
        facecolor=SURFACE,
    )
    for ax in (ax_f, ax_h, ax_p):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax_f.barh(y, failing, height=0.55, color=FAILING)
    ax_f.set_xlim(0, FAILING_TOTAL * 1.18)
    ax_f.axvline(FAILING_TOTAL, color=TEXT_2, linewidth=1, linestyle=":")
    ax_f.text(
        FAILING_TOTAL - 10,
        0.45,
        "all 1,105\nfailing",
        color=TEXT_2,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    for yi, v in zip(y, failing, strict=True):
        ax_f.text(
            v + 15, yi, f"{v:,}  ({v / FAILING_TOTAL:.1%})", va="center", fontsize=9, color=TEXT
        )
    ax_f.set_title("Failing drives reached (recall)", loc="left", fontsize=11, color=TEXT)
    ax_f.set_yticks(list(y), labels, fontsize=9, color=TEXT)

    ax_h.barh(y, healthy, height=0.55, color=HEALTHY)
    ax_h.set_xscale("log")
    ax_h.set_xlim(1, 2e5)
    for yi, v in zip(y, healthy, strict=True):
        ax_h.text(
            v * 1.15, yi, f"{v:,}  ({v / HEALTHY_TOTAL:.2%})", va="center", fontsize=9, color=TEXT
        )
    ax_h.set_title("Healthy drives flagged (log scale)", loc="left", fontsize=11, color=TEXT)

    precision = [f / (f + h) for f, h in zip(failing, healthy, strict=True)]
    ax_p.barh(y, precision, height=0.55, color=TEXT_2)
    ax_p.set_xlim(0, 0.8)
    for yi, v in zip(y, precision, strict=True):
        ax_p.text(v + 0.01, yi, f"{v:.1%}", va="center", fontsize=9, color=TEXT)
    ax_p.set_title("Precision", loc="left", fontsize=11, color=TEXT)
    ax_p.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    fig.suptitle(
        "The revised two-stage cascade on Q3 2026 "
        "(30-day model c459046e, 1,105 failing of 368,688 drives)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    fig.text(
        0.01,
        0.01,
        "Stage 1 keeps 80% of failing drives at 3.8% precision and passes 6.3% of drives; "
        "the combined score then ranks that pool into the action tiers. "
        "Counts from runbook 1.1.4, 1.1.5 and 1.9.",
        fontsize=8,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
