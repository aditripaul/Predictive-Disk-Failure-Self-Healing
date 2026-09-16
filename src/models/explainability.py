"""Model explainability via SHAP (SHapley Additive exPlanations)
(docs/design_goal.md section 5 "Explainability by default";
docs/dataset_strategy.md section 21 "Integration with the Reliability
Checker" — every high-risk prediction must be able to answer "which
features contributed most to this prediction?").

Scope: this module computes SHAP values for a trained tree model
(TreeExplainer is exact and fast for LightGBM/XGBoost) and turns them into
two audit artifacts: a global feature-importance ranking (logged once per
training run) and a per-row top-contributing-features explanation (usable
wherever a specific prediction needs to be explained, e.g. a future
serving-time integration with `PredictionOutput.top_contributing_features`).
Wiring per-prediction SHAP into the live agent loop is not done here — the
agent's lean-state design (docs/design_goal.md section 5.6) deliberately
keeps bulk data out of AgentState, so serving-time explanations would need
their own lookup path, not a field on the checkpointed state.
"""

from __future__ import annotations

import numpy as np
import shap


def build_explainer(model, background: np.ndarray) -> shap.TreeExplainer:
    """`background` is a representative sample of rows (e.g. a subsample of
    the training set) used for SHAP's baseline/expected-value calibration —
    it is not the data being explained."""
    return shap.TreeExplainer(model, background)


def compute_shap_values(explainer: shap.TreeExplainer, x: np.ndarray) -> np.ndarray:
    """Returns one row of per-feature SHAP values per input row. For a
    binary classifier, `shap_values` may return either a single
    (n_samples, n_features) array (contribution to the positive class) or a
    list of two such arrays (one per class); this normalizes to the former."""
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]  # positive-class contributions
    return np.asarray(values)


def global_feature_importance(
    shap_values: np.ndarray, feature_names: list[str]
) -> list[dict[str, float | str]]:
    """Mean absolute SHAP value per feature, sorted descending — the global
    importance ranking logged alongside a trained model."""
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} feature columns but "
            f"{len(feature_names)} feature_names were given."
        )
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs, strict=True), key=lambda pair: -pair[1])
    return [{"feature": name, "mean_abs_shap": float(value)} for name, value in ranked]


def top_contributing_features(
    shap_values_row: np.ndarray, feature_names: list[str], *, k: int = 5
) -> list[str]:
    """Per-prediction explanation: the k features with the largest absolute
    SHAP contribution for one row, signed so the direction (pushed the
    prediction up vs. down) is visible in the string itself."""
    k = min(k, len(feature_names))
    order = np.argsort(-np.abs(shap_values_row))[:k]
    return [
        f"{feature_names[i]} ({'+' if shap_values_row[i] >= 0 else ''}{shap_values_row[i]:.3f})"
        for i in order
    ]
