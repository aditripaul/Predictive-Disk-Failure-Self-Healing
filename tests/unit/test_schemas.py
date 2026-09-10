import datetime as dt

from data_contracts.schemas import (
    ActionTier,
    DriveMetadata,
    DriveType,
    FeatureConfidence,
    FeatureMaturity,
)


def test_drive_metadata_round_trip():
    meta = DriveMetadata(
        drive_id="Z1234567",
        first_seen_date=dt.date(2023, 1, 1),
        last_seen_date=dt.date(2023, 6, 1),
        model_family="Seagate-ST4000DM000",
        capacity_gb=4000.0,
        drive_type=DriveType.HDD,
        total_observed_days=150,
        active_history_days=150,
        survivorship_valid=True,
        feature_maturity=FeatureMaturity.MATURE,
    )
    assert meta.model_dump()["feature_maturity"] == "MATURE"


def test_feature_confidence_bounds_are_enforced():
    conf = FeatureConfidence(
        drive_id="Z1234567",
        as_of=dt.datetime(2023, 6, 1, 12, 0, 0),
        telemetry_coverage_30d=1.0,
        hours_since_last_telemetry=2.0,
        attribute_coverage_factor=1.0,
        feature_confidence=0.95,
        feature_maturity=FeatureMaturity.MATURE,
    )
    assert 0.0 <= conf.feature_confidence <= 1.0


def test_action_tier_enum_values():
    assert {t.value for t in ActionTier} == {
        "monitor",
        "warn",
        "cordon",
        "migrate",
        "drain",
        "human_review",
    }
