"""Figures for the cluster-based action policy (docs/cluster_action_policy.md).

Every function takes plain data (lists, dicts, arrays) from the saved results and
returns a Figure, so the figures can be redrawn without reading test data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.reporting.evaluation_figures import Curve, plot_precision_recall
from src.reporting.plots import plt

BANDS = ("monitor", "warn", "cordon", "migrate", "drain")
TIERS = BANDS[1:]
OUTCOME_COLORS = {
    "correct": "#2f7d32",
    "early": "#8bc34a",
    "later": "#ffb300",
    "removed": "#ef6c00",
    "active": "#9e9e9e",
}


def plot_ablation(rows: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for algorithm in sorted({r["algorithm"] for r in rows}):
        part = sorted((r for r in rows if r["algorithm"] == algorithm), key=lambda r: r["param"])
        x = [r["clusters"] for r in part]
        for ax, key in zip(axes, ("separation", "stability_ari", "silhouette"), strict=True):
            ax.plot(x, [r[key] for r in part], "-o", label=algorithm, markersize=4)
    for ax, key, title in zip(
        axes,
        ("separation", "stability_ari", "silhouette"),
        (
            "Failure-rate separation within band (cross-half)",
            "Stability (ARI, half vs half)",
            "Silhouette",
        ),
        strict=True,
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("clusters")
        ax.grid(alpha=0.3)
        ax.plot([chosen["clusters"]], [chosen[key]], "k*", markersize=14, label="chosen")
    axes[1].axhline(0.7, color="grey", linestyle=":", label="stability floor 0.7")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Clustering ablation on validation suspects "
        f"(chosen: {chosen['algorithm']}, {chosen['clusters']} clusters)"
    )
    fig.tight_layout()
    return fig


def plot_cluster_signatures(
    centroids: np.ndarray, columns: Sequence[str], names: Sequence[str], counts: Sequence[int]
) -> plt.Figure:
    """Standardised centroid value per (cluster, signature feature)."""
    c = np.asarray(centroids)
    keep = np.flatnonzero(np.abs(c).max(axis=0) > 0.3)
    keep = keep if len(keep) else np.arange(c.shape[1])
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(keep) + 4), 0.55 * len(names) + 3))
    im = ax.imshow(np.clip(c[:, keep], -4, 4), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(
        range(len(keep)), [columns[int(i)] for i in keep], rotation=70, ha="right", fontsize=7
    )
    ax.set_yticks(
        range(len(names)), [f"C{i}: {n} ({counts[i]:,})" for i, n in enumerate(names)], fontsize=8
    )
    fig.colorbar(im, ax=ax, label="standardised value (median/IQR)")
    ax.set_title(
        "What each cluster is: standardised signature at the decision day (validation suspects)"
    )
    fig.tight_layout()
    return fig


def plot_cluster_families(
    shares: Mapping[str, Mapping[str, float]], names: Sequence[str]
) -> plt.Figure:
    """Family mix per cluster, with the suspect pool's mix as the last bar."""
    families = sorted(
        {f for s in shares.values() for f in s}, key=lambda f: -shares["all suspects"].get(f, 0)
    )
    labels = [*(f"C{i}" for i in range(len(names))), "all suspects"]
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(labels) + 3), 4.5))
    bottom = np.zeros(len(labels))
    for fam in families:
        vals = np.array([shares[lab].get(fam, 0.0) for lab in labels])
        ax.bar(labels, vals, bottom=bottom, label=fam)
        bottom += vals
    ax.set_ylabel("share of the cluster's drives")
    ax.set_title(
        "Drive family per cluster: the clusters are not vendor in disguise if the mixes overlap"
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def plot_tier_matrix(
    cells: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    rates: Mapping[str, float | None],
    moves: Mapping[tuple[int, str], int],
    title: str,
) -> plt.Figure:
    """Failure rate per (cluster, band) cell, drive count inside the cell; cells
    the action map moves are outlined with an arrow (up = escalate)."""
    grid = np.full((len(names), len(BANDS)), np.nan)
    counts = np.zeros((len(names), len(BANDS)), dtype=int)
    for c in cells:
        j = BANDS.index(c["band"])
        grid[c["cluster"], j] = c["failure_rate"]
        counts[c["cluster"], j] = c["drives"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(names) + 2.5))
    im = ax.imshow(
        grid,
        cmap="viridis",
        vmin=0,
        vmax=max(0.05, np.nanmax(grid) if np.isfinite(grid).any() else 1),
        aspect="auto",
    )
    for i in range(len(names)):
        for j, band in enumerate(BANDS):
            if counts[i, j] == 0:
                continue
            move = moves.get((i, band), 0)
            arrow = {1: " ↑", -1: " ↓"}.get(int(np.sign(move)), "")
            ax.text(
                j,
                i,
                f"{grid[i, j]:.0%}{arrow}\n{counts[i, j]:,}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if grid[i, j] < 0.6 * np.nanmax(grid) else "black",
            )
            if move:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2
                    )
                )
    ax.set_xticks(
        range(len(BANDS)),
        [
            f"{b}\n(band {rates[b]:.0%})" if rates.get(b) is not None else f"{b}\n(empty)"
            for b in BANDS
        ],
        fontsize=8,
    )
    ax.set_yticks(range(len(names)), [f"C{i}: {n}" for i, n in enumerate(names)], fontsize=8)
    fig.colorbar(im, ax=ax, label="failure rate of the cell's drives")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_action_outcomes(comparison: Mapping[str, Any]) -> plt.Figure:
    """What happened to the flagged drives, per tier (cumulative), for both
    policies: failed within 30 days (correct), within 60 (early), later,
    removed without failure, still active."""
    tiers = [t for t in TIERS if comparison.get(t)]
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.4 * len(tiers), 4.5), squeeze=False)
    for ax, tier in zip(axes[0], tiers, strict=True):
        outs = comparison[tier]["outcomes"]
        bottom = np.zeros(2)
        for o, color in OUTCOME_COLORS.items():
            vals = np.array([outs[p].get(o, 0) for p in ("tier_only", "cluster")], dtype=float)
            ax.bar(["tier only", "cluster"], vals, bottom=bottom, color=color, label=o)
            bottom += vals
        ax.set_title(f"{tier} and above", fontsize=10)
        ax.set_ylabel("flagged drives")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Outcome of every flagged drive, tier-only against cluster policy")
    fig.tight_layout()
    return fig


def plot_policy_comparison(comparison: Mapping[str, Any]) -> plt.Figure:
    """Precision and recall per tier for both policies, the paired gap and its
    interval written above each pair, and the drives each policy moved."""
    tiers = [t for t in TIERS if comparison.get(t)]
    x = np.arange(len(tiers))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, gap_key in (
        (axes[0], "precision", "difference"),
        (axes[1], "recall", "recall_difference"),
    ):
        a = [comparison[t]["tier_only"][key] or 0 for t in tiers]
        b = [comparison[t]["cluster"][key] or 0 for t in tiers]
        ax.bar(x - w / 2, a, w, label="tier only")
        ax.bar(x + w / 2, b, w, label="cluster policy")
        for i, t in enumerate(tiers):
            p = comparison[t].get("paired")
            if p:
                ci = p[f"{gap_key}_ci"]
                ax.text(
                    i,
                    max(a[i], b[i]) + 0.02,
                    f"{p[gap_key]:+.1%}\n[{ci[0]:+.1%}, {ci[1]:+.1%}]",
                    ha="center",
                    fontsize=7,
                )
        ax.set_xticks(x, tiers)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{key.capitalize()} (tier and above)")
        ax.legend(fontsize=8)
    down = [comparison[t]["moved_down"] for t in tiers]
    down_f = [comparison[t]["moved_down_failing"] for t in tiers]
    up = [comparison[t]["moved_up"] for t in tiers]
    up_f = [comparison[t]["moved_up_failing"] for t in tiers]
    axes[2].bar(x - w / 2, down, w, color="#90a4ae", label="moved below the tier")
    axes[2].bar(x - w / 2, down_f, w, color="#c62828", label="... of which failing (lost)")
    axes[2].bar(x + w / 2, up, w, color="#a5d6a7", label="moved up to the tier")
    axes[2].bar(x + w / 2, up_f, w, color="#2e7d32", label="... of which failing (gained)")
    axes[2].set_xticks(x, tiers)
    axes[2].set_title("Drives the cluster policy moved")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_policy_pr(curve: Curve, comparison: Mapping[str, Any], base_rate: float) -> plt.Figure:
    points = []
    for t in TIERS:
        c = comparison.get(t)
        if not c or c["tier_only"]["precision"] is None:
            continue
        points.append((f"{t}, tier only", c["tier_only"]["precision"], c["tier_only"]["recall"]))
        if c["cluster"]["precision"] is not None:
            points.append(
                (f"{t}, cluster policy", c["cluster"]["precision"], c["cluster"]["recall"])
            )
    return plot_precision_recall(
        [curve],
        points=points,
        base_rate=base_rate,
        title="Combined score's drive curve, and where each policy operates (Q3 test)",
    )


def plot_lead_by_cluster(
    leads: Mapping[str, Mapping[str, Sequence[float]]], names: Sequence[str]
) -> plt.Figure:
    """Days from the decision day to failure per cluster, for drives whose
    band is warn and cordon (the decision day is the policy-independent first
    day at the band's threshold)."""
    bands = [b for b in ("warn", "cordon") if b in leads]
    fig, axes = plt.subplots(
        1, len(bands), figsize=(7 * len(bands), 0.4 * len(names) + 3), squeeze=False
    )
    for ax, band in zip(axes[0], bands, strict=True):
        data = [leads[band].get(str(i), []) for i in range(len(names))]
        ax.boxplot([d if len(d) else [np.nan] for d in data], vert=False)
        ax.set_yticks(
            range(1, len(names) + 1),
            [f"C{i}: {n} ({len(data[i])})" for i, n in enumerate(names)],
            fontsize=7,
        )
        ax.set_xlabel("days from decision to failure")
        ax.set_title(f"Failing drives in the {band} band")
    fig.tight_layout()
    return fig


def plot_cluster_stability(
    names: Sequence[str],
    val_shares: Sequence[float],
    test_shares: Sequence[float],
    val_rates: Sequence[float | None],
    test_rates: Sequence[float | None],
) -> plt.Figure:
    x = np.arange(len(names))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(x - w / 2, val_shares, w, label="validation")
    axes[0].bar(x + w / 2, test_shares, w, label="Q3 test")
    axes[0].set_title("Share of suspects per cluster")
    axes[1].bar(x - w / 2, [r or 0 for r in val_rates], w, label="validation")
    axes[1].bar(x + w / 2, [r or 0 for r in test_rates], w, label="Q3 test")
    axes[1].set_title("Failure rate per cluster (all suspects)")
    for ax in axes:
        ax.set_xticks(x, [f"C{i}" for i in range(len(names))])
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_tier_thresholds(
    curve: Curve,
    points: Sequence[tuple[str, float, float]],
    targets: Mapping[str, float],
    base_rate: float,
) -> plt.Figure:
    """The validation drive curve with each tier's threshold and the lift-derived
    precision target it had to meet (a target above 100% is drawn at the top)."""
    fig = plot_precision_recall(
        [curve],
        points=points,
        base_rate=base_rate,
        title="How the tiers are set: validation drive curve and lift-derived targets",
    )
    ax = fig.axes[0]
    for name, target in targets.items():
        y = min(target, 1.0)
        ax.axhline(y, linestyle="--", linewidth=0.8, color="grey")
        ax.text(
            0.99,
            y + 0.01,
            f"{name} target {target:.1%}" + (" (unreachable)" if target > 1 else ""),
            ha="right",
            fontsize=7,
        )
    return fig


def plot_lift_targets(lifts: Mapping[str, float], rates: Mapping[str, float]) -> plt.Figure:
    """Precision target = lift x failure rate, for each base rate: the same lift
    asks for less precision where failures are rarer."""
    tiers = list(lifts)
    x = np.arange(len(tiers))
    w = 0.8 / max(len(rates), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, rate) in enumerate(rates.items()):
        vals = [lifts[t] * rate for t in tiers]
        bars = ax.bar(
            x + (i - (len(rates) - 1) / 2) * w, vals, w, label=f"{label} (failure rate {rate:.3%})"
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{t}\n(lift {lifts[t]:g})" for t in tiers])
    ax.set_ylabel("precision target")
    ax.set_title("Lift x failure rate = precision target per tier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
