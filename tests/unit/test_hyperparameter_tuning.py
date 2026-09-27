import numpy as np

from src.models.evaluation import compute_auprc
from src.models.hyperparameter_tuning import tune_lightgbm_hyperparameters
from src.models.training import predict_proba_positive


def _toy_classification_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    return x, y


def test_tune_lightgbm_hyperparameters_returns_best_params_and_trials():
    x, y = _toy_classification_data()
    x_train, y_train = x[:200], y[:200]
    x_val, y_val = x[200:], y[200:]

    result = tune_lightgbm_hyperparameters(
        x_train,
        y_train,
        x_val,
        y_val,
        base_params={"objective": "binary", "is_unbalance": True},
        search_space={"num_leaves": [7, 15], "n_estimators": [10, 30]},
        n_trials=3,
        subsample_fraction=0.5,
        seed=0,
    )

    assert result["n_trials"] == 3
    assert len(result["trials"]) == 3
    assert "num_leaves" in result["best_params"]
    assert "n_estimators" in result["best_params"]
    assert result["best_params"]["objective"] == "binary"  # base_params carried through
    assert result["subsample_size"] == 100


def test_tune_lightgbm_hyperparameters_is_deterministic_given_a_seed():
    x, y = _toy_classification_data()
    x_train, y_train = x[:200], y[:200]
    x_val, y_val = x[200:], y[200:]

    kwargs = dict(
        base_params={"objective": "binary"},
        search_space={"num_leaves": [7, 31]},
        n_trials=3,
        subsample_fraction=0.5,
        seed=42,
    )
    result_a = tune_lightgbm_hyperparameters(x_train, y_train, x_val, y_val, **kwargs)
    result_b = tune_lightgbm_hyperparameters(x_train, y_train, x_val, y_val, **kwargs)
    assert result_a["best_params"] == result_b["best_params"]


def test_tune_lightgbm_hyperparameters_produces_a_usable_model():
    x, y = _toy_classification_data()
    x_train, y_train = x[:200], y[:200]
    x_val, y_val = x[200:], y[200:]

    result = tune_lightgbm_hyperparameters(
        x_train,
        y_train,
        x_val,
        y_val,
        base_params={"objective": "binary"},
        search_space={"num_leaves": [7, 15], "learning_rate": [0.05, 0.2]},
        n_trials=2,
        subsample_fraction=1.0,
        seed=0,
    )
    from src.models.training import train_lightgbm

    model = train_lightgbm(x_train, y_train, params=result["best_params"])
    scores = predict_proba_positive(model, x_val)
    assert compute_auprc(y_val, scores) > 0.5
