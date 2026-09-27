import numpy as np

from src.models.evaluation import compute_auprc
from src.models.logistic_regression_baseline import train_logistic_regression_baseline
from src.models.training import predict_proba_positive


def _toy_classification_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    return x, y


def test_train_logistic_regression_baseline_and_predict_proba():
    x, y = _toy_classification_data()
    model = train_logistic_regression_baseline(x, y)
    scores = predict_proba_positive(model, x)
    assert scores.shape == (300,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert compute_auprc(y, scores) > 0.5


def test_train_logistic_regression_baseline_handles_unscaled_features():
    # feature 0 spans thousands, feature 1 spans 0-1 - a badly-scaled input
    # that would mislead an unscaled logistic regression's coefficients.
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(scale=1000, size=300), rng.normal(scale=0.01, size=300)])
    y = (x[:, 0] > 0).astype(int)
    model = train_logistic_regression_baseline(x, y)
    scores = predict_proba_positive(model, x)
    assert compute_auprc(y, scores) > 0.9


def test_train_logistic_regression_baseline_is_a_fitted_pipeline():
    x, y = _toy_classification_data()
    model = train_logistic_regression_baseline(x, y)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps
    assert model.named_steps["classifier"].get_params()["class_weight"] == "balanced"
