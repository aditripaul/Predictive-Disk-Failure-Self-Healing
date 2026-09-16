import numpy as np

from src.reliability.drift import build_drift_report, classify_psi, population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5000)
    current = rng.normal(size=5000)
    psi = population_stability_index(reference, current)
    assert psi < 0.05


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(loc=0.0, size=5000)
    current = rng.normal(loc=3.0, size=5000)
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_psi_raises_on_empty_input():
    try:
        population_stability_index(np.array([]), np.array([1.0]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "no_significant_shift"
    assert classify_psi(0.15) == "moderate_shift"
    assert classify_psi(0.30) == "significant_shift"


def test_build_drift_report_flags_retrain_when_any_feature_shifts():
    rng = np.random.default_rng(0)
    reference = {
        "stable_feature": rng.normal(size=2000),
        "shifted_feature": rng.normal(size=2000),
    }
    current = {
        "stable_feature": rng.normal(size=2000),
        "shifted_feature": rng.normal(loc=5.0, size=2000),
    }

    report = build_drift_report(reference, current)

    assert report["features"]["stable_feature"]["classification"] == "no_significant_shift"
    assert report["features"]["shifted_feature"]["classification"] == "significant_shift"
    assert report["retrain_recommended"] is True


def test_build_drift_report_skips_features_missing_from_current():
    reference = {"only_in_reference": np.array([1.0, 2.0, 3.0])}
    report = build_drift_report(reference, {})
    assert report["features"] == {}
    assert report["retrain_recommended"] is False
