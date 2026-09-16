"""LightGBM training for disk-failure prediction (docs/dataset_strategy.md
section 16.1). Uses `is_unbalance` class weighting rather than SMOTE by
default, per the imbalance-mitigation preference order in section 15.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np


def train_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    default_params: dict[str, Any] = {
        "objective": "binary",
        "is_unbalance": True,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(x_train, y_train)
    return model


def predict_proba_positive(model: lgb.LGBMClassifier, x: np.ndarray) -> np.ndarray:
    # lightgbm's stubs type predict_proba's return as a plain list, but it
    # always returns an (n_samples, n_classes) ndarray at runtime.
    probabilities = np.asarray(model.predict_proba(x))
    return probabilities[:, 1]
