"""A drive must not be judged for an attribute its vendor never reports.

These are the regression tests for docs/adr/0002. The bug they pin down was
invisible to every existing test because all of them build synthetic fleets
where every drive reports every attribute: `attribute_coverage_factor` divided
by the whole priority list, so the ~68% of real drive-days without Seagate's
smart_187/188 were capped at 3/5 = 0.60 against a 0.80 destructive-action
floor - structurally unable to be cordoned, migrated or drained no matter how
complete and fresh their telemetry was.
"""

import datetime as dt
import math

import polars as pl
import pytest

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.features.confidence import add_feature_confidence
from src.features.cross_vendor import (
    expected_attribute_column,
    model_expected_attributes,
)
from src.guardrails.context import RuleContext
from src.guardrails.engine import GuardrailEngine

PRIORITY = [
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",  # Seagate only
    "command_timeout",  # Seagate only
]
SEAGATE, HGST = "ST12000NM0008", "HGST HUH721212ALN604"


def _fleet(n_days: int = 10) -> pl.DataFrame:
    """Two models with perfect telemetry. The HGST never reports the two
    Seagate attributes, exactly as the real fleet behaves."""
    rows = []
    for model in (SEAGATE, HGST):
        for day in range(n_days):
            row = {
                "drive_id": f"{model}-1",
                "drive_model": model,
                "date": dt.date(2026, 1, 1) + dt.timedelta(days=day),
                "reallocated_sector_count": 1.0,
                "current_pending_sector_count": 0.0,
                "offline_uncorrectable": 0.0,
                "reported_uncorrectable_errors": 2.0 if model == SEAGATE else None,
                "command_timeout": 0.0 if model == SEAGATE else None,
                "telemetry_coverage_30d": 1.0,
                "days_since_last_telemetry": 0,
            }
            rows.append(row)
    return pl.DataFrame(rows)


def _confidence(frame: pl.DataFrame, model: str, **kwargs) -> float:
    out = add_feature_confidence(frame, PRIORITY, **kwargs)
    return float(out.filter(pl.col("drive_model") == model)["feature_confidence"].min())


def test_vendor_specific_absence_does_not_reduce_confidence():
    fleet = _fleet()
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1)
    # The HGST reports 3 of the 5; all 3 are present, so it is fully covered.
    assert _confidence(fleet, HGST, expected_by_model=expected) == pytest.approx(1.0)
    assert _confidence(fleet, SEAGATE, expected_by_model=expected) == pytest.approx(1.0)


def test_the_global_denominator_is_what_caused_the_bug():
    """Without the model table, the HGST is capped below the 0.80 floor."""
    fleet = _fleet()
    assert _confidence(fleet, HGST) == pytest.approx(0.6)
    assert _confidence(fleet, HGST) < 0.80


def test_genuine_telemetry_loss_still_lowers_confidence():
    """A Seagate drive that stops reporting an attribute it normally reports
    must still be marked down - that is real degradation, not a vendor gap."""
    fleet = _fleet()
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1)
    degraded = fleet.with_columns(
        pl.when((pl.col("drive_model") == SEAGATE) & (pl.col("date") == dt.date(2026, 1, 5)))
        .then(None)
        .otherwise(pl.col("reported_uncorrectable_errors"))
        .alias("reported_uncorrectable_errors")
    )
    out = add_feature_confidence(degraded, PRIORITY, expected_by_model=expected)
    bad_day = out.filter(
        (pl.col("drive_model") == SEAGATE) & (pl.col("date") == dt.date(2026, 1, 5))
    )
    assert bad_day["attribute_coverage_factor"][0] == pytest.approx(4 / 5)


def test_unseen_model_is_treated_as_reporting_everything():
    fleet = _fleet()
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1)
    newcomer = fleet.head(1).with_columns(pl.lit("TOSHIBA MG08ACA16TA").alias("drive_model"))
    out = add_feature_confidence(newcomer, PRIORITY, expected_by_model=expected)
    # Conservative: an unknown model cannot earn more confidence than its data.
    assert out["attribute_coverage_factor"][0] == pytest.approx(1.0)


def test_sparse_model_keeps_every_attribute_expected():
    """Below min_drive_days there is too little evidence to conclude a model
    does not report something, so nothing is excused."""
    fleet = _fleet(n_days=2)
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1_000)
    row = expected.filter(pl.col("drive_model") == HGST)
    assert all(row[expected_attribute_column(a)][0] for a in PRIORITY), (
        "a sparsely-seen model must not have attributes excused"
    )


def test_expected_flags_do_not_leak_into_the_frame():
    """pl.Boolean counts as a feature dtype, so the scaffolding must be gone."""
    fleet = _fleet()
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1)
    out = add_feature_confidence(fleet, PRIORITY, expected_by_model=expected)
    assert not [c for c in out.columns if c.startswith("_expected_")]


def _drain_ctx(confidence: float) -> RuleContext:
    return RuleContext(
        action_id="a1",
        drive_id=f"{HGST}-1",
        proposed_action=ActionTier.DRAIN,
        p_fail=0.99,
        feature_confidence=confidence,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
    )


def test_guardrail_no_longer_blocks_a_healthy_non_seagate_drive():
    """The end of the chain: a non-Seagate drive with complete, fresh telemetry
    and a high score must not be blocked from a destructive action."""
    fleet = _fleet()
    expected = model_expected_attributes(fleet.lazy(), PRIORITY, min_drive_days=1)
    confidence = _confidence(fleet, HGST, expected_by_model=expected)

    result = GuardrailEngine().evaluate(_drain_ctx(confidence))
    assert not [v for v in result.violations if v.rule_id == "FEATURE_CONFIDENCE"]
    assert result.passed is True

    # The pre-fix value must still be blocked, so this fails if the
    # denominator ever regresses to the global one.
    blocked = GuardrailEngine().evaluate(_drain_ctx(0.60))
    assert [v for v in blocked.violations if v.rule_id == "FEATURE_CONFIDENCE"]
    assert blocked.passed is False


def _daily(days_since_last: int, coverage: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": ["A"],
            "drive_model": [SEAGATE],
            "date": [dt.date(2026, 1, 10)],
            "telemetry_coverage_30d": [coverage],
            "days_since_last_telemetry": [days_since_last],
            **{a: [1.0] for a in PRIORITY},
        }
    )


def test_healthy_daily_telemetry_clears_the_destructive_floor():
    """The second bug in ADR 0002: the normal daily cadence was charged as a
    full day of staleness, capping every drive-day at exp(-24/48) = 0.6065
    against a 0.80 floor, which no coverage could lift."""
    out = add_feature_confidence(_daily(days_since_last=1), PRIORITY)
    assert out["recency_factor"][0] == pytest.approx(1.0)
    assert out["feature_confidence"][0] >= 0.80


def test_a_real_telemetry_gap_still_lowers_confidence():
    out = add_feature_confidence(_daily(days_since_last=3), PRIORITY)
    # 3 days against a 1-day cadence is 2 days of real staleness.
    assert out["recency_factor"][0] == pytest.approx(math.exp(-48.0 / 48.0))
    assert out["feature_confidence"][0] < 0.80


def test_a_drives_first_day_is_not_charged_staleness():
    """days_since_last_telemetry is fill_null(0) on a drive's first row; it has
    no previous reading, so there is nothing to be stale against."""
    out = add_feature_confidence(_daily(days_since_last=0), PRIORITY)
    assert out["recency_factor"][0] == pytest.approx(1.0)


def test_interval_column_still_reports_the_true_gap():
    """Recency is charged on the excess, but the reported interval must stay
    the real one - src/models/serving.py surfaces it."""
    out = add_feature_confidence(_daily(days_since_last=3), PRIORITY)
    assert out["hours_since_last_telemetry"][0] == pytest.approx(72.0)


def test_the_build_refuses_a_fleet_that_can_never_act():
    """The guard that would have caught both bugs. Unit fixtures use ideal
    values, so only a check against real output catches a fleet-wide cap."""
    from pipelines.build_gold_features import _check_confidence_is_attainable

    blocked = pl.DataFrame({"feature_confidence": [0.6065, 0.5, 0.033]})
    with pytest.raises(ValueError, match="could never cordon, migrate or drain"):
        _check_confidence_is_attainable(blocked)

    # One actionable drive-day is enough to prove the floor is reachable.
    _check_confidence_is_attainable(pl.DataFrame({"feature_confidence": [0.6065, 0.95]}))
