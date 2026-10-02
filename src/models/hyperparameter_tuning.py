"""Optuna-based LightGBM hyperparameter search (docs/project_plan.md
Phase 6 "Threshold tuning: Use Optuna. Subsample for search. Retrain best
configuration on full training partition."; RAM Practices "Use subsampled
Optuna search."; docs/project_plan.md line 114: "Hyperparameter tuning |
Optuna | Threshold and model tuning | Subsampled search").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from src.models.evaluation import compute_auprc
from src.models.training import predict_proba_positive, train_lightgbm

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _subsample(
    x: np.ndarray, y: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return x, y
    rng = np.random.default_rng(seed)
    n = max(1, round(len(y) * fraction))
    idx = rng.choice(len(y), size=n, replace=False)
    return x[idx], y[idx]


def tune_lightgbm_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    base_params: dict[str, Any],
    search_space: dict[str, list[float]],
    n_trials: int = 20,
    subsample_fraction: float = 0.3,
    seed: int = 0,
    positive_weight_power: float | None = None,
) -> dict[str, Any]:
    """Runs a subsampled Optuna search over `search_space` (each key mapped
    to a `[low, high]` range - an all-int range samples an int, otherwise a
    float), training each trial on a random subsample of
    `(x_train, y_train)` and scoring AUPRC on the full, never-subsampled
    `(x_val, y_val)` so the objective reflects real generalization, not an
    artifact of the subsample. Returns the best hyperparameters (merged
    over `base_params`) and the trial history for auditability. The
    caller is expected to retrain a final model on the FULL training
    partition with the returned params (docs/project_plan.md: "Retrain
    best configuration on full training partition") - this function never
    does that itself."""
    x_sub, y_sub = _subsample(x_train, y_train, fraction=subsample_fraction, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        sampled = {
            name: (
                trial.suggest_int(name, int(low), int(high))
                if isinstance(low, int) and isinstance(high, int)
                else trial.suggest_float(name, float(low), float(high))
            )
            for name, (low, high) in search_space.items()
        }
        params = {**base_params, **sampled}
        model = train_lightgbm(
            x_sub, y_sub, params=params, positive_weight_power=positive_weight_power
        )
        scores = predict_proba_positive(model, x_val)
        return compute_auprc(y_val, scores)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": {**base_params, **study.best_params},
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "subsample_fraction": subsample_fraction,
        "subsample_size": len(y_sub),
        "trials": [
            {"number": t.number, "params": t.params, "value": t.value} for t in study.trials
        ],
    }
