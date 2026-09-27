"""XGBoost training — the alternative tree-based model to LightGBM
(docs/design_goal.md / docs/project_plan.md: "ML models | XGBoost /
LightGBM | Primary failure prediction models"; "XGBoost | Strong
alternative; external-memory support"). Uses `scale_pos_weight =
negative_count / positive_count` for class imbalance
(docs/dataset_strategy.md section 15 Imbalance Mitigations #1), the
XGBoost-specific counterpart to LightGBM's `is_unbalance`
(`src/models/training.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    default_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "eval_metric": "aucpr",
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(x_train, y_train)
    return model
