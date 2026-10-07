"""Monotone probability calibration, fitted on the validation split.

The training objective class-weights positives (`positive_weight_power`), so
raw model scores are not failure probabilities: a score of 0.9 does not mean
90% of such drive-days fail. An isotonic map fitted on validation turns scores
into calibrated probabilities. Because the map is non-decreasing, it never
changes the ranking, so precision at a given rank cut-off is unchanged; what
changes is that the number reported as "probability" means what it says.

Decisions (action tiers) are still set by `src/models/threshold.py` on the
raw ranking. Calibrated metrics are reported beside the raw ones, not in place
of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Non-decreasing map from raw score to calibrated probability in [0, 1]."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if scores.shape != y_true.shape:
            raise ValueError("scores and labels must have the same shape")
        if len(scores) == 0:
            raise ValueError("cannot calibrate on an empty split")
        if np.unique(y_true).size < 2:
            raise ValueError("calibration needs both classes in the fitting split")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(scores, y_true)
        self._x = np.asarray(iso.X_thresholds_, dtype=float)
        self._y = np.asarray(iso.y_thresholds_, dtype=float)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._x is None or self._y is None:
            raise RuntimeError("calibrator is not fitted")
        # sklearn's isotonic predict is linear interpolation between these
        # knots, clipped outside them; np.interp reproduces it exactly.
        return np.interp(np.asarray(scores, dtype=float), self._x, self._y)

    def to_dict(self) -> dict[str, Any]:
        if self._x is None or self._y is None:
            raise RuntimeError("calibrator is not fitted")
        return {"x": self._x.tolist(), "y": self._y.tolist()}

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> IsotonicCalibrator:
        calibrator = cls()
        calibrator._x = np.asarray(settings["x"], dtype=float)
        calibrator._y = np.asarray(settings["y"], dtype=float)
        return calibrator
