import numpy as np
import pytest

from src.models.calibration import IsotonicCalibrator
from src.models.evaluation import compute_calibration


def _miscalibrated(seed: int, n: int = 20000):
    """True failure probability p; the model reports a class-weighted score
    s = p / (p + (1 - p) * 0.1), which overstates risk by about 10x at low p."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 0.2, size=n)
    y = (rng.random(n) < p).astype(float)
    raw = p / (p + (1 - p) * 0.1)
    return raw, y


def test_calibration_improves_expected_calibration_error():
    raw, y = _miscalibrated(0)
    before = compute_calibration(y, raw)["expected_calibration_error"]
    cal = IsotonicCalibrator().fit(raw, y)
    after = compute_calibration(y, cal.transform(raw))["expected_calibration_error"]
    assert before > 0.1  # the fixture really is miscalibrated
    assert after < 0.02
    assert after < before / 5


def test_calibration_preserves_ranking():
    raw, y = _miscalibrated(1)
    cal = IsotonicCalibrator().fit(raw, y)
    probe = np.linspace(raw.min(), raw.max(), 500)
    mapped = cal.transform(probe)
    assert np.all(np.diff(mapped) >= 0)  # non-decreasing map


def test_calibrated_scores_are_probabilities():
    raw, y = _miscalibrated(2)
    out = IsotonicCalibrator().fit(raw, y).transform(raw)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_round_trip_is_exact():
    raw, y = _miscalibrated(3)
    cal = IsotonicCalibrator().fit(raw, y)
    restored = IsotonicCalibrator.from_dict(cal.to_dict())
    np.testing.assert_array_equal(cal.transform(raw), restored.transform(raw))


def test_out_of_range_scores_are_clipped():
    raw, y = _miscalibrated(4)
    cal = IsotonicCalibrator().fit(raw, y)
    out = cal.transform(np.array([-5.0, 5.0]))
    assert 0.0 <= out[0] <= out[1] <= 1.0


def test_rejects_single_class_and_unfitted_use():
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(np.array([0.1]))
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().to_dict()
