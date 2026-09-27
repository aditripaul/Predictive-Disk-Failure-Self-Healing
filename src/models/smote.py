"""Subsampled SMOTE (docs/dataset_strategy.md section 15 Imbalance
Mitigations #3): "SMOTE may be used, but only on training data and only
if it improves AUPRC... apply SMOTE only to a stratified training sample;
never apply SMOTE to validation or test data; compare against class
weighting; prefer class weighting if SMOTE does not clearly improve
validation performance." `imbalanced-learn` was previously not even a
declared dependency - this is a comparison tool for `pipelines/
train_model.py`, not a replacement for the class-weighting the primary
model already uses (`is_unbalance`/`scale_pos_weight`).
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def apply_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    subsample_fraction: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified-subsamples `(x_train, y_train)` to `subsample_fraction`
    (SMOTE's nearest-neighbor search is the expensive part on a large
    training set) and then applies SMOTE to that subsample. Never touches
    validation/test data - callers must not pass it here."""
    if subsample_fraction < 1.0:
        x_sub, _, y_sub, _ = train_test_split(
            x_train,
            y_train,
            train_size=subsample_fraction,
            stratify=y_train,
            random_state=seed,
        )
    else:
        x_sub, y_sub = x_train, y_train

    smote = SMOTE(random_state=seed)
    x_resampled, y_resampled = smote.fit_resample(x_sub, y_sub)
    return np.asarray(x_resampled), np.asarray(y_resampled)
