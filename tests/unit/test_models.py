import datetime as dt

import numpy as np
import polars as pl

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.models.action_tiers import determine_action_tier
from src.models.evaluation import compute_auprc, evaluate_at_threshold
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
