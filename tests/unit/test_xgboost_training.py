import numpy as np

from src.models.evaluation import compute_auprc
from src.models.training import predict_proba_positive
from src.models.xgboost_training import train_xgboost


def _toy_classification_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    return x, y


def test_train_xgboost_and_predict_proba():
    x, y = _toy_classification_data()
    model = train_xgboost(x, y, params={"n_estimators": 20})
    scores = predict_proba_positive(model, x)
    assert scores.shape == (200,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert compute_auprc(y, scores) > 0.5


def test_train_xgboost_computes_scale_pos_weight_from_class_counts():
    y = np.array([1, 0, 0, 0, 0])  # 1 positive, 4 negative -> scale_pos_weight = 4.0
    x = np.random.default_rng(0).normal(size=(5, 2))
    model = train_xgboost(x, y, params={"n_estimators": 5})
    assert model.get_params()["scale_pos_weight"] == 4.0


def test_train_xgboost_respects_explicit_param_overrides():
    x, y = _toy_classification_data()
    model = train_xgboost(x, y, params={"n_estimators": 7, "max_depth": 2})
    assert model.get_params()["n_estimators"] == 7
    assert model.get_params()["max_depth"] == 2
