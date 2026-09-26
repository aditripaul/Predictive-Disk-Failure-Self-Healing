import datetime as dt

import numpy as np
import polars as pl

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.models.action_tiers import determine_action_tier
from src.models.evaluation import (
    compute_auprc,
    compute_calibration,
    compute_warning_lead_time_days,
    evaluate_at_threshold,
    precision_at_k,
    precision_at_k_fractions,
)
from src.models.features import assemble_training_frame, select_feature_columns
from src.models.threshold import tune_threshold_for_precision
from src.models.training import predict_proba_positive, train_lightgbm

ACTION_THRESHOLDS = {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}


def test_assemble_training_frame_and_select_feature_columns():
    gold = pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
            "model_family": ["Seagate HDD"] * 3,
            "reallocated_sector_count_7d_mean": [0.0, 1.0, 5.0],
        }
    )
    labels = pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
            "horizon_days": [14, 14, 14],
            "label": [0, 1, None],
            "split": ["train", "train", "train"],
        }
    )
    frame = assemble_training_frame(gold, labels, horizon_days=14)
    assert frame.height == 2  # the null-label row for B is dropped

    feature_columns = select_feature_columns(frame)
    assert feature_columns == ["reallocated_sector_count_7d_mean"]
    assert "model_family" not in feature_columns
    assert "label" not in feature_columns


def _toy_classification_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    return x, y


def test_train_lightgbm_and_predict_proba():
    x, y = _toy_classification_data()
    model = train_lightgbm(x, y, params={"n_estimators": 20})
    scores = predict_proba_positive(model, x)
    assert scores.shape == (200,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert compute_auprc(y, scores) > 0.5


def test_tune_threshold_for_precision_meets_target():
    x, y = _toy_classification_data()
    model = train_lightgbm(x, y, params={"n_estimators": 50})
    scores = predict_proba_positive(model, x)

    result = tune_threshold_for_precision(y, scores, target_precision=0.9)
    assert result["precision"] >= 0.9 or not result["target_met"]

    metrics = evaluate_at_threshold(y, scores, result["threshold"])
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_tune_threshold_falls_back_when_target_unreachable():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # indistinguishable scores
    result = tune_threshold_for_precision(y, scores, target_precision=0.999)
    assert result["target_met"] is False


def test_precision_at_k_picks_highest_scored_rows():
    y_true = np.array([0, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    # top-2 by score: indices 2 (0.9, label 1) and 3 (0.8, label 1) -> precision 1.0
    assert precision_at_k(y_true, y_scores, k=2) == 1.0
    # top-3 by score adds index 4 (0.3, label 0) -> 2/3
    assert abs(precision_at_k(y_true, y_scores, k=3) - 2 / 3) < 1e-9


def test_precision_at_k_caps_k_at_population_size():
    y_true = np.array([1, 0])
    y_scores = np.array([0.9, 0.1])
    assert precision_at_k(y_true, y_scores, k=1000) == 0.5


def test_precision_at_k_fractions_returns_one_entry_per_fraction():
    y_true = np.array([1] * 10 + [0] * 90)
    y_scores = np.linspace(1.0, 0.0, 100)
    result = precision_at_k_fractions(y_true, y_scores, fractions=(0.1,))
    assert result["precision_at_top_10pct"] == 1.0


def test_compute_calibration_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    y_scores = rng.uniform(0, 1, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < y_scores).astype(int)
    result = compute_calibration(y_true, y_scores, n_bins=10)
    assert result["expected_calibration_error"] < 0.1
    assert len(result["bins"]) == 10
    assert result["brier_score"] >= 0.0


def test_compute_calibration_flags_badly_miscalibrated_scores():
    y_true = np.array([0] * 100)
    y_scores = np.array([0.9] * 100)
    result = compute_calibration(y_true, y_scores, n_bins=10)
    assert result["expected_calibration_error"] > 0.5


def _lead_time_frame():
    return pl.DataFrame(
        {
            "drive_id": ["A", "A", "A", "B", "B", "C"],
            "days_to_event": [10, 5, 1, 3, 1, 2],
            "score": [0.2, 0.9, 0.95, 0.1, 0.1, 0.9],
        }
    )


def test_compute_warning_lead_time_picks_earliest_crossing_per_drive():
    df = _lead_time_frame()
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    # drive A first crosses threshold at days_to_event=5 (not the later 1)
    # drive B never crosses threshold; drive C crosses at days_to_event=2
    assert result["warned_drive_count"] == 2
    assert result["total_failed_drive_count"] == 3
    assert result["max_lead_time_days"] == 5.0
    assert abs(result["mean_lead_time_days"] - (5 + 2) / 2) < 1e-9
    assert abs(result["warning_coverage"] - 2 / 3) < 1e-9


def test_compute_warning_lead_time_handles_no_warnings():
    df = pl.DataFrame({"drive_id": ["A"], "days_to_event": [5], "score": [0.1]})
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    assert result["warned_drive_count"] == 0
    assert result["mean_lead_time_days"] is None
    assert result["warning_coverage"] == 0.0


def test_compute_warning_lead_time_handles_empty_frame():
    df = pl.DataFrame({"drive_id": [], "days_to_event": [], "score": []})
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    assert result["total_failed_drive_count"] == 0
    assert result["warning_coverage"] is None


def test_determine_action_tier_downgrades_low_confidence_destructive_action():
    tier = determine_action_tier(
        p_fail=0.95,
        feature_confidence=0.1,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.CORDON


def test_determine_action_tier_allows_destructive_action_with_high_confidence():
    tier = determine_action_tier(
        p_fail=0.95,
        feature_confidence=0.9,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.DRAIN


def test_determine_action_tier_low_risk_is_monitor():
    tier = determine_action_tier(
        p_fail=0.05,
        feature_confidence=0.99,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.MONITOR
