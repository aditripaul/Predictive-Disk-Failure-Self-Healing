import json

import numpy as np
import pytest

import pipelines.evaluate_frozen as frozen
from src.models.calibration import IsotonicCalibrator


class _FakeModel:
    """Returns the scores it was built with, row for row."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        s = self._scores[: len(x)]
        return np.column_stack([1.0 - s, s])


def _spec(scores: np.ndarray, y: np.ndarray) -> dict:
    calibrator = IsotonicCalibrator().fit(scores, y)
    return {
        "drive_threshold": 0.5,
        "action_tier_thresholds": {"warn": 0.3, "drain": None},
        "calibrator": calibrator.to_dict(),
    }


def _data(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    drives = np.repeat([f"d{i}" for i in range(n // 6)], 6)
    y = np.zeros(n, dtype=int)
    y[::50] = 1
    scores = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, n), 0, 1)
    return drives, y, scores


def test_score_frozen_uses_only_the_frozen_thresholds():
    drives, y, scores = _data()
    spec = _spec(scores, y)
    x = np.zeros((len(y), 2), dtype=np.float32)
    out = frozen.score_frozen(
        spec=spec,
        model=_FakeModel(scores),
        two_stage=None,
        x=x,
        y=y,
        drive_ids=drives,
        n_boot=50,
    )
    assert out["drive_threshold"] == 0.5
    assert out["action_tiers"]["drain"] is None
    assert out["action_tiers"]["warn"]["threshold"] == 0.3
    assert out["rows"] == len(y)
    assert out["positives"] == int(y.sum())
    lo, hi = out["drive_bootstrap_ci"]["precision_ci"]
    assert lo <= out["drive_bootstrap_ci"]["precision"] <= hi


def test_score_frozen_rejects_misaligned_inputs():
    drives, y, scores = _data()
    x = np.zeros((len(y) - 1, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        frozen.score_frozen(
            spec=_spec(scores, y),
            model=_FakeModel(scores),
            two_stage=None,
            x=x,
            y=y,
            drive_ids=drives,
            n_boot=10,
        )


def test_second_evaluation_of_the_same_split_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "RESULTS_DIR", tmp_path)
    (tmp_path / "sealed__run123.json").write_text(json.dumps({"split": "sealed"}))
    with pytest.raises(SystemExit, match="single-shot"):
        frozen.evaluate("run123", "sealed", tmp_path / "frame.parquet", n_boot=10)


def test_only_sealed_or_external_splits_can_be_evaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "RESULTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="split must be one of"):
        frozen.evaluate("run123", "test", tmp_path / "frame.parquet", n_boot=10)
