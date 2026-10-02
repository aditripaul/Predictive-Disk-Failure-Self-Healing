"""LightGBM training for disk-failure prediction (docs/dataset_strategy.md
section 16.1). Uses `is_unbalance` class weighting rather than SMOTE by
default, per the imbalance-mitigation preference order in section 15.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np


def early_stopping_subset(
    y: np.ndarray, *, max_negatives: int | None = 1_500_000, seed: int = 0
) -> np.ndarray:
    """Sorted row indices of a validation subsample for early stopping:
    every positive row plus at most `max_negatives` random negatives.
    Average precision on this subsample ranks models the same way as on the
    full split (the dropped rows are all negatives), at a fraction of the
    scoring cost and without a second multi-GB copy of the matrix."""
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if max_negatives is not None and len(negatives) > max_negatives:
        negatives = np.random.default_rng(seed).choice(
            negatives, size=max_negatives, replace=False
        )
    return np.sort(np.concatenate([positives, negatives]))


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    positive_weight_power: float | None = None,
    eval_x: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> lgb.LGBMClassifier:
    """`positive_weight_power`: weight positives by `(n_neg / n_pos) **
    power` (`scale_pos_weight`) instead of LightGBM's `is_unbalance`
    (power 1.0, the full ratio). On real fleet data (~0.1% positives, each
    failing drive contributing ~14 near-identical rows) the full ratio
    makes every tree chase a handful of drives - the model collapsed to a
    single tree with leaf values ~1e7 - while the square root (0.5) trained
    normally and ranked failures ~50-100x better by AUPRC. `None` keeps
    the old `is_unbalance` behavior. An explicit `scale_pos_weight` or
    `is_unbalance` in `params` always wins.

    `eval_x`/`eval_y` + `early_stopping_rounds`: stop adding trees once
    average precision on that held-out set stops improving, and keep the
    best iteration."""
    default_params: dict[str, Any] = {
        "objective": "binary",
        "is_unbalance": True,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "verbosity": -1,
    }
    if positive_weight_power is not None:
        default_params.pop("is_unbalance")
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            default_params["scale_pos_weight"] = (n_neg / n_pos) ** positive_weight_power
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    if eval_x is not None and eval_y is not None and early_stopping_rounds:
        model.fit(
            x_train,
            y_train,
            eval_X=eval_x,
            eval_y=eval_y,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(x_train, y_train)
    return model


def predict_proba_positive(model: Any, x: np.ndarray) -> np.ndarray:
    # Works for any sklearn-style classifier with predict_proba (LightGBM,
    # XGBoost, scikit-learn's LogisticRegression, ...) - lightgbm's own
    # stubs type predict_proba's return as a plain list, but it always
    # returns an (n_samples, n_classes) ndarray at runtime.
    probabilities = np.asarray(model.predict_proba(x))
    return probabilities[:, 1]
