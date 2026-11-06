"""Precision-recall curves and confusion matrices, written by the pipelines
that train or evaluate a model (`make train`, the cascade experiment,
`scripts/compare_horizons.py`) for every split they score.

Every number is per DRIVE, built with `src/models/evaluation.py::
drive_level_table` - the same drive scoring every table in the runbook uses -
so a confusion matrix here equals the run's `drive_level_metrics` at the same
threshold. The row-level curve is drawn beside it (dashed) because it is what
an AUPRC on drive-days measures, and the gap between the two is the
over-counting a drive's ~horizon near-identical rows causes.

Plot functions take precomputed arrays and return a Figure; the numbers they
draw are returned by the `*_data` helpers, so tests check data, not pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import drive_level_table, top_k_from_drive_table
from src.reporting.plots import plt, save_figure

#: Curves are thinned to about this many points before drawing; a 20M-row
#: curve otherwise makes a multi-megabyte PNG with no visible difference.
MAX_CURVE_POINTS = 2000


@dataclass(frozen=True)
class Curve:
    label: str
    precision: np.ndarray
    recall: np.ndarray
    linestyle: str = "-"


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    threshold: float
    precision: float
    recall: float
    caught: int
    false_alarms: int
    missed: int
    true_negatives: int


def thin_curve(
    precision: np.ndarray, recall: np.ndarray, max_points: int = MAX_CURVE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """At most `max_points` evenly spaced points, always keeping both ends."""
    n = len(precision)
    if n <= max_points:
        return np.asarray(precision), np.asarray(recall)
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return np.asarray(precision)[idx], np.asarray(recall)[idx]


def pr_curve(label: str, y_true: np.ndarray, scores: np.ndarray, linestyle: str = "-") -> Curve:
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(scores))
    precision, recall = thin_curve(precision, recall)
    return Curve(label, precision, recall, linestyle)


def operating_point(
    name: str, drive_labels: np.ndarray, drive_scores: np.ndarray, threshold: float
) -> OperatingPoint:
    """Counts at `threshold` on a drive table; flagged = score >= threshold,
    exactly as `drive_level_metrics`."""
    labels = np.asarray(drive_labels).astype(bool)
    flagged = np.asarray(drive_scores) >= threshold
    caught = int((flagged & labels).sum())
    false_alarms = int((flagged & ~labels).sum())
    missed = int((~flagged & labels).sum())
    true_negatives = int((~flagged & ~labels).sum())
    return OperatingPoint(
        name=name,
        threshold=float(threshold),
        precision=caught / (caught + false_alarms) if caught + false_alarms else 0.0,
        recall=caught / (caught + missed) if caught + missed else 0.0,
        caught=caught,
        false_alarms=false_alarms,
        missed=missed,
        true_negatives=true_negatives,
    )


def plot_precision_recall(
    curves: Sequence[Curve],
    *,
    points: Sequence[tuple[str, float, float]] = (),
    base_rate: float | None = None,
    title: str,
) -> plt.Figure:
    """One curve per entry; `points` are (name, precision, recall) markers;
    `base_rate` draws the precision a random alert would get."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for curve in curves:
        ax.plot(curve.recall, curve.precision, linestyle=curve.linestyle, label=curve.label)
    markers = "oDs^v<>Pp*h"
    for i, (name, precision, recall) in enumerate(points):
        ax.plot(
            recall,
            precision,
            marker=markers[i % len(markers)],
            linestyle="none",
            markersize=7,
            label=f"{name} (P {precision:.1%}, R {recall:.1%})",
        )
    if base_rate is not None:
        ax.axhline(base_rate, color="grey", linestyle=":", label=f"base rate {base_rate:.3%}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(points: Sequence[OperatingPoint], *, title: str) -> plt.Figure:
    """One 2x2 panel per operating point, drive counts with row-normalised
    rates (share of failing / healthy drives in each cell)."""
    n = max(1, len(points))
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, point in zip(axes.flat, points, strict=False):
        counts = np.array(
            [[point.caught, point.missed], [point.false_alarms, point.true_negatives]]
        )
        rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]:,}\n{rates[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="white" if rates[i, j] > 0.6 else "black",
                    fontsize=9,
                )
        ax.set_xticks([0, 1], ["flagged", "not flagged"])
        ax.set_yticks([0, 1], ["failing", "healthy"])
        ax.set_title(
            f"{point.name}\nthr {point.threshold:.4g}, P {point.precision:.1%}, "
            f"R {point.recall:.1%}",
            fontsize=9,
        )
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def evaluation_figure_data(
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    top_k: Sequence[int] = (),
    row_level: bool = True,
) -> dict[str, Any]:
    """Everything the two figures of one split draw."""
    drive_labels, drive_scores = drive_level_table(drive_ids, y_true, scores)
    curves = [pr_curve("drives", drive_labels, drive_scores)]
    if row_level:
        curves.append(pr_curve("drive-days", y_true, scores, linestyle="--"))
    points = [
        operating_point(name, drive_labels, drive_scores, threshold)
        for name, threshold in thresholds.items()
        if threshold is not None
    ]
    top = top_k_from_drive_table(drive_labels, drive_scores, top_k) if len(top_k) else []
    return {
        "curves": curves,
        "operating_points": points,
        "top_k": top,
        "base_rate": float(np.mean(drive_labels)) if len(drive_labels) else 0.0,
        "failing_drives": int(np.sum(drive_labels)),
        "drives": len(drive_labels),
    }


def write_evaluation_figures(
    out_dir: Path,
    split: str,
    drive_ids: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    title_prefix: str = "",
    in_sample: bool = False,
    top_k: Sequence[int] = (),
    row_level: bool = True,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Writes `{split}_precision_recall.png` and `{split}_confusion.png` to
    `out_dir` and, inside an active MLflow run, logs them under `figures/`.
    `in_sample` marks scores on the rows the model was fitted on, which are
    optimistic by construction. Returns the drawn data and the file paths."""
    data = evaluation_figure_data(
        drive_ids, y_true, scores, thresholds, top_k=top_k, row_level=row_level
    )
    suffix = " (in-sample, optimistic)" if in_sample else ""
    title = (
        f"{title_prefix}{split}{suffix}: {data['failing_drives']:,} failing of "
        f"{data['drives']:,} drives"
    )
    markers = [(p.name, p.precision, p.recall) for p in data["operating_points"]]
    markers += [(f"top {row['k']:,}", row["precision"], row["recall"]) for row in data["top_k"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        save_figure(
            plot_precision_recall(
                data["curves"], points=markers, base_rate=data["base_rate"], title=title
            ),
            out_dir / f"{split}_precision_recall.png",
        )
    ]
    if data["operating_points"]:
        paths.append(
            save_figure(
                plot_confusion_matrices(data["operating_points"], title=title),
                out_dir / f"{split}_confusion.png",
            )
        )
    if log_to_mlflow:
        import mlflow

        if mlflow.active_run() is not None:
            for path in paths:
                mlflow.log_artifact(str(path), artifact_path="figures")
    data["paths"] = paths
    data["title"] = title
    return data
