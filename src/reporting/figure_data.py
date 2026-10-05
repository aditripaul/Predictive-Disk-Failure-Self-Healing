"""Data behind the figures from the paper (Amram et al., 2021) that apply to
this pipeline. Pure functions, so the numbers are testable without drawing.

- `roc_points`: ROC curve points, downsampled for storage in the evaluation
  report (the paper's Figure 11 compares ROC curves of three models).
- `gain_importance`: LightGBM's gain-based feature importance, normalized
  (the paper's variable importance Figures 5, 7, 10 and 13).
- `failure_counts_by_family`: drives and failed drives per drive family
  (the paper's Figure 2, failures per drive model).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve


def roc_points(
    y_true: np.ndarray, y_scores: np.ndarray, *, max_points: int = 400
) -> dict[str, list[float]]:
    """False-positive and true-positive rates along the ROC curve, thinned to
    at most `max_points` evenly spaced thresholds. Degenerate inputs (only one
    class) return empty lists rather than failing."""
    empty: dict[str, Any] = {"fpr": [], "tpr": [], "auc": None}
    if len(np.unique(y_true)) < 2:
        return empty
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    keep = np.unique(np.linspace(0, len(fpr) - 1, num=min(max_points, len(fpr))).astype(int))
    result: dict[str, Any] = {
        "fpr": [round(float(v), 6) for v in fpr[keep]],
        "tpr": [round(float(v), 6) for v in tpr[keep]],
        "auc": float(roc_auc_score(y_true, y_scores)),
    }
    return result


def gain_importance(
    feature_names: list[str], gains: np.ndarray, *, top_n: int = 20
) -> list[dict[str, Any]]:
    """Top features by share of total gain, sorted descending."""
    total = float(np.sum(gains)) or 1.0
    ranked = sorted(
        zip(feature_names, np.asarray(gains, dtype=float), strict=True), key=lambda p: -p[1]
    )
    return [
        {"feature": name, "importance": round(float(g) / total, 6)} for name, g in ranked[:top_n]
    ]


def failure_counts_by_family(metadata: pl.DataFrame) -> list[dict[str, Any]]:
    """One row per drive family: drives seen, drives that failed, and the
    failure share. Expects `model_family` and `failure_date` (nullable), as
    written by `src/preprocess/feature_maturity.py::build_drive_metadata`."""
    rows = (
        metadata.with_columns(pl.col("failure_date").is_not_null().alias("failed"))
        .group_by("model_family")
        .agg(
            pl.len().alias("drives"),
            pl.col("failed").sum().alias("failed_drives"),
        )
        .sort("failed_drives", descending=True)
    )
    return [
        {
            "model_family": r["model_family"],
            "drives": int(r["drives"]),
            "failed_drives": int(r["failed_drives"]),
            "failure_share": round(r["failed_drives"] / r["drives"], 6) if r["drives"] else 0.0,
        }
        for r in rows.iter_rows(named=True)
    ]
