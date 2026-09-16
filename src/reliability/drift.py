"""Feature and prediction drift monitoring (docs/dataset_strategy.md section
19, docs/design_goal.md section 16 "Drift / retrain signal"). PSI is the
primary metric: it needs no distributional assumptions and has widely-used
rule-of-thumb thresholds, making it a practical first drift signal ahead of
a full statistical test suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: conventional rule-of-thumb PSI thresholds
PSI_NO_SIGNIFICANT_SHIFT = 0.1
PSI_MODERATE_SHIFT = 0.25


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10
) -> float:
    """PSI between a reference distribution (`expected`, e.g. training-time
    feature or score values) and a current one (`actual`, e.g. this week's).
    Bin edges are quantiles of `expected`, so each reference bin starts with
    ~equal mass; empty bins are floored to avoid a divide-by-zero/log(0)."""
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("Both `expected` and `actual` must be non-empty.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(expected, quantiles))
    if len(bin_edges) < 3:
        # Degenerate reference distribution (e.g. near-constant); PSI is
        # not meaningful with fewer than 2 usable bins.
        return 0.0
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = np.maximum(expected_counts / len(expected), 1e-6)
    actual_pct = np.maximum(actual_counts / len(actual), 1e-6)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def classify_psi(psi: float) -> str:
    if psi < PSI_NO_SIGNIFICANT_SHIFT:
        return "no_significant_shift"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate_shift"
    return "significant_shift"


def build_drift_report(
    reference_by_feature: dict[str, np.ndarray],
    current_by_feature: dict[str, np.ndarray],
) -> dict[str, Any]:
    """One PSI + classification per feature present in both snapshots, plus
    a `retrain_recommended` flag if any feature shows a significant shift -
    the "drift / retrain signal" the Reliability Checker feeds to MLOps."""
    report: dict[str, dict] = {}
    for feature_name, reference_values in reference_by_feature.items():
        current_values = current_by_feature.get(feature_name)
        if current_values is None or len(current_values) == 0:
            continue
        psi = population_stability_index(reference_values, current_values)
        report[feature_name] = {"psi": psi, "classification": classify_psi(psi)}

    retrain_recommended = any(r["classification"] == "significant_shift" for r in report.values())
    return {"features": report, "retrain_recommended": retrain_recommended}
