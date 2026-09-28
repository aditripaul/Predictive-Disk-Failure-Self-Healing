"""Performance-metric plots rendered from the JSON reports `make train` /
`make build-labels` already write (`data/audit/data_quality_reports/`) -
no new metrics are computed here, this is purely a visualization layer.

Uses matplotlib's headless "Agg" backend (set before `pyplot` is
imported) since this runs from a script/CI, never an interactive
session with a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration_curve(calibration: dict[str, Any], *, title: str) -> plt.Figure:
    """Reliability diagram: mean predicted probability vs. observed
    fraction of positives per bin (`src/models/evaluation.py::
    compute_calibration`'s output). Empty bins (count == 0) are skipped."""
    bins = [b for b in calibration["bins"] if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    if bins:
        predicted = [b["mean_predicted_probability"] for b in bins]
        observed = [b["observed_fraction_positive"] for b in bins]
        ax.plot(predicted, observed, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ece = calibration["expected_calibration_error"]
    brier = calibration["brier_score"]
    ax.set_title(f"{title}\nECE={ece:.4f}, Brier={brier:.4f}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_shap_feature_importance(
    shap_importance: list[dict[str, Any]], *, top_n: int = 20
) -> plt.Figure:
    """Horizontal bar chart of the top-N features by global mean |SHAP
    value| (`src/models/explainability.py::global_feature_importance`'s
    output, sorted descending)."""
    top = list(shap_importance[:top_n])
    features = [e["feature"] for e in top][::-1]
    importance = [e["mean_abs_shap"] for e in top][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.3)))
    ax.barh(features, importance)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {len(top)} features by global SHAP importance")
    fig.tight_layout()
    return fig


def plot_class_imbalance(class_distribution: list[dict[str, Any]]) -> plt.Figure:
    """Grouped bar chart of failure rate (%) per split, one group per
    horizon (`src/labels/imbalance.py::class_distribution_report`'s
    output)."""
    horizons = sorted({r["horizon_days"] for r in class_distribution})
    splits = sorted({r["split"] for r in class_distribution}, key=_split_sort_key)

    fig, ax = plt.subplots(figsize=(7, 5))
    n = len(horizons)
    width = 0.8 / max(n, 1)
    x = range(len(splits))
    for i, horizon in enumerate(horizons):
        rates = []
        for split in splits:
            match = next(
                (
                    r
                    for r in class_distribution
                    if r["horizon_days"] == horizon and r["split"] == split
                ),
                None,
            )
            rates.append(match["failure_rate"] * 100 if match else 0.0)
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, rates, width=width, label=f"{horizon}d horizon")

    ax.set_xticks([xi + width * (n - 1) / 2 for xi in x])
    ax.set_xticklabels(splits)
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Class imbalance by split and horizon")
    ax.legend()
    fig.tight_layout()
    return fig


def _split_sort_key(split: str) -> tuple[int, str]:
    order = {"train": 0, "validation": 1, "test": 2, "external_smartz": 3}
    return (order.get(split, 99), split)


def plot_metric_comparison(
    validation_metrics: dict[str, Any], test_metrics: dict[str, Any]
) -> plt.Figure:
    """Side-by-side validation vs. test bars for AUPRC/precision/recall/
    FPR/FNR - the headline numbers from `evaluate_at_threshold`."""
    metric_keys = ["auprc", "precision", "recall", "false_positive_rate", "false_negative_rate"]
    labels = ["AUPRC", "Precision", "Recall", "FPR", "FNR"]
    val_values = [float(validation_metrics[k]) for k in metric_keys]
    test_values = [float(test_metrics[k]) for k in metric_keys]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([xi - width / 2 for xi in x], val_values, width=width, label="Validation")
    ax.bar([xi + width / 2 for xi in x], test_values, width=width, label="Test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation vs. test metrics")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_at_k(metrics: dict[str, Any], *, title: str) -> plt.Figure:
    """Bar chart of `precision_at_top_{K}pct` entries
    (`src/models/evaluation.py::precision_at_k_fractions`'s output),
    sorted by K ascending."""
    entries = [(k, v) for k, v in metrics.items() if k.startswith("precision_at_top_")]
    entries.sort(key=lambda kv: float(kv[0].removeprefix("precision_at_top_").removesuffix("pct")))
    labels = [k.removeprefix("precision_at_top_") for k, _ in entries]
    values = [v for _, v in entries]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision at top-K ({title})")
    fig.tight_layout()
    return fig
