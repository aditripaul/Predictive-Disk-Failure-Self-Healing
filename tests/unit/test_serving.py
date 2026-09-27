import datetime as dt

import numpy as np
import polars as pl

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.models.serving import (
    join_feature_maturity,
    latest_row_per_drive,
    propose_actions,
    score_latest_drive_day,
)
from src.models.training import train_lightgbm

ACTION_THRESHOLDS = {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}
AS_OF = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)


def _trained_model():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] > 0).astype(int)
    return train_lightgbm(x, y, params={"n_estimators": 20})


def _latest_features(feature_columns: list[str], **overrides) -> pl.DataFrame:
    defaults = {
        "drive_id": ["D-1", "D-2"],
        "date": [dt.date(2024, 6, 1), dt.date(2024, 6, 1)],
        "telemetry_coverage_30d": [1.0, 1.0],
        "hours_since_last_telemetry": [1.0, 1.0],
        "attribute_coverage_factor": [1.0, 1.0],
        "feature_confidence": [0.95, 0.95],
        "feature_maturity": [FeatureMaturity.MATURE.value, FeatureMaturity.MATURE.value],
        feature_columns[0]: [5.0, -5.0],
        feature_columns[1]: [1.0, -1.0],
    }
    defaults.update(overrides)
    return pl.DataFrame(defaults)


def test_score_latest_drive_day_produces_one_validated_prediction_per_row():
    model = _trained_model()
    features = _latest_features(["f0", "f1"])

    predictions = score_latest_drive_day(
        features,
        model,
        ["f0", "f1"],
        model_name="lightgbm",
        model_version="1",
        horizon_days=14,
        as_of=AS_OF,
    )

    assert len(predictions) == 2
    by_drive = {p.drive_id: p for p in predictions}
    assert 0.0 <= by_drive["D-1"].p_fail <= 1.0
    assert by_drive["D-1"].feature_confidence.feature_maturity == FeatureMaturity.MATURE
    assert by_drive["D-1"].feature_confidence.drive_id == "D-1"
    assert by_drive["D-1"].as_of == AS_OF
    assert by_drive["D-1"].horizon_days == 14
    # D-1's features point toward failure, D-2's toward health
    assert by_drive["D-1"].p_fail > by_drive["D-2"].p_fail


def test_score_latest_drive_day_raises_on_missing_confidence_columns():
    model = _trained_model()
    features = _latest_features(["f0", "f1"]).drop("feature_confidence")

    try:
        score_latest_drive_day(
            features,
            model,
            ["f0", "f1"],
            model_name="lightgbm",
            model_version="1",
            horizon_days=14,
            as_of=AS_OF,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "feature_confidence" in str(exc)


def test_propose_actions_only_returns_proposals_above_monitor():
    model = _trained_model()
    features = _latest_features(["f0", "f1"])
    predictions = score_latest_drive_day(
        features,
        model,
        ["f0", "f1"],
        model_name="lightgbm",
        model_version="1",
        horizon_days=14,
        as_of=AS_OF,
    )

    proposals = propose_actions(
        predictions,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )

    proposed_drive_ids = {p.drive_id for p in proposals}
    # every proposal must actually be above MONITOR
    assert all(p.proposed_action != ActionTier.MONITOR for p in proposals)
    # D-1 is the high-p_fail drive; if it crossed any threshold it must appear
    by_drive = {pred.drive_id: pred for pred in predictions}
    if by_drive["D-1"].p_fail >= ACTION_THRESHOLDS["warn"]:
        assert "D-1" in proposed_drive_ids


def test_propose_actions_downgrades_destructive_tier_under_low_confidence():
    prediction_features = _latest_features(
        ["f0", "f1"],
        feature_confidence=[0.1, 0.1],  # below min_confidence_for_destructive_action
    )
    model = _trained_model()
    predictions = score_latest_drive_day(
        prediction_features,
        model,
        ["f0", "f1"],
        model_name="lightgbm",
        model_version="1",
        horizon_days=14,
        as_of=AS_OF,
    )
    # force a high p_fail to exercise the destructive-tier downgrade path
    predictions[0] = predictions[0].model_copy(update={"p_fail": 0.95})

    proposals = propose_actions(
        predictions,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    d1_proposal = next(p for p in proposals if p.drive_id == "D-1")
    assert d1_proposal.proposed_action == ActionTier.CORDON  # downgraded from DRAIN


def test_propose_actions_rationale_and_prediction_are_populated():
    model = _trained_model()
    features = _latest_features(["f0", "f1"])
    predictions = score_latest_drive_day(
        features,
        model,
        ["f0", "f1"],
        model_name="lightgbm",
        model_version="1",
        horizon_days=14,
        as_of=AS_OF,
    )
    predictions[0] = predictions[0].model_copy(update={"p_fail": 0.95})

    proposals = propose_actions(
        predictions,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    d1_proposal = next(p for p in proposals if p.drive_id == "D-1")
    assert d1_proposal.prediction.drive_id == "D-1"
    assert "p_fail=0.950" in d1_proposal.rationale
    assert d1_proposal.created_at == AS_OF


def test_latest_row_per_drive_picks_the_most_recent_date():
    df = pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 3), dt.date(2024, 1, 2)],
            "p_fail_input": [1.0, 2.0, 3.0],
        }
    )
    out = latest_row_per_drive(df)
    by_drive = {row["drive_id"]: row for row in out.to_dicts()}
    assert by_drive["A"]["date"] == dt.date(2024, 1, 3)
    assert by_drive["A"]["p_fail_input"] == 2.0
    assert by_drive["B"]["date"] == dt.date(2024, 1, 2)


def test_join_feature_maturity_adds_the_per_drive_column():
    """feature_maturity lives only in Silver's drive_metadata table, never
    in gold features - this is what pipelines/score_fleet.py relies on to
    satisfy score_latest_drive_day's REQUIRED_CONFIDENCE_COLUMNS check."""
    latest_features = pl.DataFrame({"drive_id": ["A", "B"], "some_feature": [1.0, 2.0]})
    drive_metadata = pl.DataFrame(
        {
            "drive_id": ["A", "B"],
            "feature_maturity": [FeatureMaturity.MATURE.value, FeatureMaturity.WARMUP.value],
            "unrelated_metadata_column": ["x", "y"],
        }
    )
    out = join_feature_maturity(latest_features, drive_metadata)
    by_drive = {row["drive_id"]: row["feature_maturity"] for row in out.to_dicts()}
    assert by_drive == {"A": "MATURE", "B": "WARMUP"}
    assert "unrelated_metadata_column" not in out.columns
