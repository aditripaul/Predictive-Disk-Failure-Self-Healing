"""Two-stage scoring: a second model re-ranks the drive-days the first model
already finds suspicious.

Stage 1 is the pipeline's ordinary model. Its *candidates* are the rows
scoring at or above the validation threshold that still catches
`candidate_recall` of the failing drives. Stage 2 is trained only on train
rows that are candidates, with the Stage 1 score as an extra feature, so it
spends all its capacity on telling failing drives from the healthy drives
that look like them.

The final score keeps one scale for every action tier:

- a non-candidate row keeps its Stage 1 score (always below the candidate
  threshold), so lenient tiers behave exactly as with Stage 1 alone;
- a candidate row scores `threshold + (1 - threshold) * stage2_score`, so
  candidates always rank above non-candidates and are ordered by Stage 2.

Stage 2 must be trained on Stage 1 scores the rows would really get in use.
In-sample Stage 1 scores are far more confident than anything Stage 2 meets
later, so the train rows are scored out of fold: each drive by a Stage 1
model that never saw it (`out_of_fold_scores`).

A cascade cannot raise precision by filtering alone; it helps only if Stage
2 separates the look-alike drives better than Stage 1 did. On the 30-day
label it did so at the strictest operating points in two runs; see
docs/model_status_and_runbook.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import drive_level_table
from src.models.training import PREDICT_CHUNK_ROWS, predict_proba_positive, train_lightgbm

#: Fits one LightGBM. `eval_x`/`eval_y` switch on early stopping;
#: `n_estimators` fixes the tree count instead; `extra_params` override the
#: pipeline's parameters for this fit.
Fitter = Callable[..., lgb.LGBMClassifier]

#: Stage 2 sees thousands of rows, not millions: it needs smaller leaves
#: than the pipeline's `min_child_samples`.
STAGE2_PARAMS = {"min_child_samples": 50}


def lightgbm_fitter(
    params: dict[str, Any],
    *,
    positive_weight_power: float | None,
    early_stopping_rounds: int | None,
    seed: int = 0,
) -> Fitter:
    """A `Fitter` using the pipeline's own training function, class
    weighting and early stopping (src/models/training.py::train_lightgbm)."""

    def fit(
        x: np.ndarray,
        y: np.ndarray,
        *,
        eval_x: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
        n_estimators: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> lgb.LGBMClassifier:
        fit_params = {**params, "random_state": seed, "verbosity": -1, **(extra_params or {})}
        if n_estimators is not None:
            fit_params["n_estimators"] = n_estimators
        return train_lightgbm(
            x,
            y,
            params=fit_params,
            positive_weight_power=positive_weight_power,
            eval_x=eval_x,
            eval_y=eval_y,
            early_stopping_rounds=early_stopping_rounds if eval_x is not None else None,
        )

    return fit


def tree_count(model: lgb.LGBMClassifier) -> int:
    return int(model.best_iteration_ or model.n_estimators)


def drive_folds(drive_ids: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per row, by hash of the drive: all of a drive's rows share
    a fold, so an out-of-fold score comes from a model that never saw that
    drive."""
    return (pl.Series("d", drive_ids).hash(seed=seed) % n_folds).to_numpy().astype(np.int64)


def out_of_fold_scores(
    fit: Fitter,
    x: np.ndarray,
    y: np.ndarray,
    drive_ids: np.ndarray,
    *,
    n_estimators: int,
    n_folds: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Stage 1 scores for the train rows, each from a model trained on the
    other folds' drives with the same tree count as the real Stage 1."""
    folds = drive_folds(drive_ids, n_folds, seed)
    scores = np.zeros(len(y), dtype=np.float64)
    for fold in range(n_folds):
        held_out = folds == fold
        if not held_out.any():
            continue
        model = fit(x[~held_out], y[~held_out], n_estimators=n_estimators)
        scores[held_out] = predict_proba_positive(model, x[held_out])
        del model
    return scores


def threshold_for_drive_recall(
    drive_ids: np.ndarray, y: np.ndarray, scores: np.ndarray, recall: float
) -> float | None:
    """The highest-precision drive-level threshold that still catches at
    least `recall` of the failing drives; None if that is not reachable."""
    labels, drive_scores = drive_level_table(drive_ids, y, scores)
    if labels.sum() == 0:
        return None
    precision, achieved, thresholds = precision_recall_curve(labels, drive_scores)
    precision, achieved = precision[:-1], achieved[:-1]
    ok = np.flatnonzero(achieved >= recall)
    if not len(ok):
        return None
    return float(thresholds[ok[np.argmax(precision[ok])]])


def with_stage1_score(x: np.ndarray, stage1_scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Stage 2's input: the `rows` of `x` plus their Stage 1 score as a last
    column."""
    return np.column_stack([np.asarray(x[rows]), stage1_scores[rows]])


class TwoStageModel:
    """Stage 1, Stage 2 and the candidate threshold, with the classifier
    interface the rest of the pipeline scores through (`predict_proba`)."""

    def __init__(self, stage1: Any, stage2: Any, candidate_threshold: float) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.candidate_threshold = float(candidate_threshold)

    def final_scores(self, x: np.ndarray, stage1_scores: np.ndarray) -> np.ndarray:
        """The combined score for rows whose Stage 1 scores are already
        known. `x` may be a memory map; it is read one chunk at a time."""
        threshold = self.candidate_threshold
        final = np.asarray(stage1_scores, dtype=np.float64).copy()
        for start in range(0, len(final), PREDICT_CHUNK_ROWS):
            stop = min(start + PREDICT_CHUNK_ROWS, len(final))
            rows = start + np.flatnonzero(final[start:stop] >= threshold)
            if not len(rows):
                continue
            stage2_scores = predict_proba_positive(
                self.stage2, with_stage1_score(x, stage1_scores, rows)
            )
            final[rows] = threshold + (1.0 - threshold) * stage2_scores
        return final

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        stage1_scores = np.asarray(self.stage1.predict_proba(x))[:, 1]
        final = self.final_scores(x, stage1_scores)
        return np.column_stack([1.0 - final, final])


def fit_second_stage(
    fit: Fitter,
    stage1: lgb.LGBMClassifier,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_drives: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_drives: np.ndarray,
    val_stage1_scores: np.ndarray,
    candidate_recall: float = 0.5,
    n_folds: int = 3,
    seed: int = 0,
    min_candidate_positives: int = 20,
) -> tuple[TwoStageModel | None, dict[str, Any]]:
    """Trains Stage 2 for an already-trained `stage1`.

    Returns the combined model and a report; the model is None (and
    `report["status"]` says why) when validation cannot reach
    `candidate_recall` or too few failing rows are candidates to train on."""
    report: dict[str, Any] = {
        "stage1_trees": tree_count(stage1),
        "candidate_recall": candidate_recall,
        "n_folds": n_folds,
        "seed": seed,
    }
    threshold = threshold_for_drive_recall(val_drives, y_val, val_stage1_scores, candidate_recall)
    report["candidate_threshold"] = threshold
    if threshold is None:
        report["status"] = "candidate recall not reachable on validation"
        return None, report

    train_stage1_scores = out_of_fold_scores(
        fit,
        x_train,
        y_train,
        train_drives,
        n_estimators=report["stage1_trees"],
        n_folds=n_folds,
        seed=seed,
    )
    train_rows = np.flatnonzero(train_stage1_scores >= threshold)
    val_rows = np.flatnonzero(val_stage1_scores >= threshold)
    report["candidates"] = {
        "train_rows": len(train_rows),
        "train_positive_rows": int(y_train[train_rows].sum()),
        "validation_rows": len(val_rows),
    }
    if (
        report["candidates"]["train_positive_rows"] < min_candidate_positives
        or y_val[val_rows].sum() < 5
    ):
        report["status"] = "too few candidate failures to train a second stage"
        return None, report

    stage2 = fit(
        with_stage1_score(x_train, train_stage1_scores, train_rows),
        y_train[train_rows],
        eval_x=with_stage1_score(x_val, val_stage1_scores, val_rows),
        eval_y=y_val[val_rows],
        extra_params=STAGE2_PARAMS,
    )
    report["stage2_trees"] = tree_count(stage2)
    report["status"] = "ok"
    return TwoStageModel(stage1, stage2, threshold), report
