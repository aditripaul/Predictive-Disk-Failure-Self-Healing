import numpy as np

from src.models.explainability import (
    build_explainer,
    compute_shap_values,
    global_feature_importance,
    top_contributing_features,
)
from src.models.training import train_lightgbm

FEATURE_NAMES = ["informative", "noise_1", "noise_2"]


def _toy_model_and_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    model = train_lightgbm(x, y, params={"n_estimators": 30})
    return model, x, y


def test_compute_shap_values_shape_matches_input():
    model, x, _ = _toy_model_and_data()
    explainer = build_explainer(model, background=x[:50])
    shap_values = compute_shap_values(explainer, x)
    assert shap_values.shape == x.shape


def test_global_feature_importance_ranks_the_informative_feature_first():
    model, x, _ = _toy_model_and_data()
    explainer = build_explainer(model, background=x[:50])
    shap_values = compute_shap_values(explainer, x)

    ranking = global_feature_importance(shap_values, FEATURE_NAMES)

    assert len(ranking) == 3
    assert ranking[0]["feature"] == "informative"
    assert ranking[0]["mean_abs_shap"] > ranking[1]["mean_abs_shap"]
    assert ranking[0]["mean_abs_shap"] > ranking[2]["mean_abs_shap"]


def test_global_feature_importance_rejects_mismatched_feature_names():
    model, x, _ = _toy_model_and_data()
    explainer = build_explainer(model, background=x[:50])
    shap_values = compute_shap_values(explainer, x)

    try:
        global_feature_importance(shap_values, ["only_one_name"])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "feature_names" in str(exc)


def test_top_contributing_features_includes_the_informative_feature():
    model, x, _ = _toy_model_and_data()
    explainer = build_explainer(model, background=x[:50])
    shap_values = compute_shap_values(explainer, x)

    explanation = top_contributing_features(shap_values[0], FEATURE_NAMES, k=2)

    assert len(explanation) == 2
    assert any(line.startswith("informative") for line in explanation)


def test_top_contributing_features_caps_k_at_feature_count():
    model, x, _ = _toy_model_and_data()
    explainer = build_explainer(model, background=x[:50])
    shap_values = compute_shap_values(explainer, x)

    explanation = top_contributing_features(shap_values[0], FEATURE_NAMES, k=10)
    assert len(explanation) == 3
