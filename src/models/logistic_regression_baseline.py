"""Logistic Regression baseline — an interpretable sanity check
(docs/project_plan.md Phase 6 Model Candidates: "Logistic Regression |
Interpretable sanity baseline"). Not intended for production decisions;
trained alongside the primary model so a training run can confirm the
primary (tree-based) model is actually adding value over a simple linear
baseline, rather than assuming it without checking.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression_baseline(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Standardizes features first — unlike the tree-based models,
    logistic regression is scale-sensitive and the gold feature columns
    span very different raw ranges (counts vs. ratios vs. day counts).
    `class_weight="balanced"` is sklearn's equivalent of LightGBM's
    `is_unbalance`/XGBoost's `scale_pos_weight` for the same rare-failure
    imbalance (docs/dataset_strategy.md section 15)."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline
