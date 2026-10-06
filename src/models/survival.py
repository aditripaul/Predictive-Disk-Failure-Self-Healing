"""Survival (time-to-failure) model for the experiment loop: XGBoost's
accelerated failure time objective (`survival:aft`).

The binary label asks "does this drive fail within H days?", so a drive that
fails on day H+5 is taught, and scored, as healthy. A survival model learns
the time to failure itself and treats a drive that has not failed yet as
"not failed up to day t" (right-censored) instead of as a negative.

Used only by `pipelines/experiment_model.py`; not part of `make train`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.labels.event_types import FAILURE_EVENT_TYPES

#: Smallest time to failure handed to the AFT objective, which works on
#: log(time) and needs it strictly positive.
MIN_SURVIVAL_DAYS = 0.5

AFT_PARAMS: dict[str, Any] = {
    "objective": "survival:aft",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 20,
    "reg_lambda": 10.0,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "verbosity": 0,
    # Early stopping watches drive-failure ranking (average precision), the
    # same criterion as the classifiers, not the AFT likelihood.
    "disable_default_eval_metric": 1,
}


def survival_bounds(
    dates: np.ndarray,
    event_types: np.ndarray,
    days_to_event: np.ndarray,
    *,
    cutoff_date: np.datetime64,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) time-to-failure bounds in days for each row.

    Outcomes are only used up to `cutoff_date + horizon_days`: the same
    information the H-day binary label of the last row before the cutoff
    uses, so the survival model is not shown more of the future than the
    classifier it is compared with. A row's follow-up is therefore
    `(cutoff_date - date) + horizon_days` days.

    - Failure seen within the follow-up: exact time, lower == upper.
    - Anything else (still running, removed, or failing after the
      follow-up): right-censored at the follow-up, upper = +inf.
    """
    follow_up = (cutoff_date - dates.astype("datetime64[D]")).astype("timedelta64[D]").astype(
        np.float64
    ) + float(horizon_days)
    follow_up = np.maximum(follow_up, MIN_SURVIVAL_DAYS)
    days = np.asarray(days_to_event, dtype=np.float64)
    is_failure = np.isin(event_types, list(FAILURE_EVENT_TYPES))
    observed = is_failure & np.isfinite(days) & (days <= follow_up)

    lower = follow_up.copy()
    upper = np.full(len(lower), np.inf)
    lower[observed] = np.maximum(days[observed], MIN_SURVIVAL_DAYS)
    upper[observed] = lower[observed]
    return lower.astype(np.float32), upper.astype(np.float32)


def risk_from_survival_time(predicted_days: np.ndarray, horizon_days: int) -> np.ndarray:
    """Maps a predicted time to failure to a risk score in (0, 1), higher =
    sooner: 0.5 when the prediction equals the horizon. Monotone, so it
    ranks drives exactly as the predicted time does; the 0-1 range is what
    the threshold and reporting code expects."""
    predicted = np.maximum(np.asarray(predicted_days, dtype=np.float64), 0.0)
    return horizon_days / (horizon_days + predicted)


class AftModel:
    """A trained AFT booster with the classifier interface the scoring code
    uses (`predict_proba(x)[:, 1]` = risk)."""

    def __init__(self, booster: xgb.Booster, horizon_days: int) -> None:
        self.booster = booster
        self.horizon_days = horizon_days

    @property
    def tree_count(self) -> int:
        return int(self.booster.best_iteration) + 1

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        predicted = self.booster.predict(xgb.DMatrix(x), iteration_range=(0, self.tree_count))
        risk = risk_from_survival_time(predicted, self.horizon_days)
        return np.column_stack([1.0 - risk, risk])

    def gain_by_feature(self, feature_count: int) -> np.ndarray:
        gains = np.zeros(feature_count, dtype=np.float64)
        for name, gain in self.booster.get_score(importance_type="total_gain").items():
            gains[int(name[1:])] = gain  # unnamed features are "f<index>"
        return gains


def train_aft(
    x_train: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    horizon_days: int,
    sample_weight: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> AftModel:
    """Trains on time-to-failure bounds; early-stops on the average
    precision of the risk score against the binary H-day label `eval_y`."""
    train_matrix = xgb.QuantileDMatrix(
        x_train, label_lower_bound=lower, label_upper_bound=upper, weight=sample_weight
    )
    eval_matrix = xgb.QuantileDMatrix(eval_x, ref=train_matrix)
    eval_labels = np.asarray(eval_y)

    def average_precision(predicted: np.ndarray, _: xgb.DMatrix) -> tuple[str, float]:
        risk = risk_from_survival_time(predicted, horizon_days)
        return "average_precision", float(average_precision_score(eval_labels, risk))

    booster = xgb.train(
        {**AFT_PARAMS, **(params or {})},
        train_matrix,
        num_boost_round=num_boost_round,
        evals=[(eval_matrix, "validation")],
        custom_metric=average_precision,
        maximize=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return AftModel(booster, horizon_days)
