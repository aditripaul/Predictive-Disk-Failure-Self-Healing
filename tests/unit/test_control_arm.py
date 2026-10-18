"""The control arm: drives the agent deliberately does not act on.

Acting censors the outcome, so a system that acts on everything it flags
destroys the labels it would need to retrain. A fixed, stably-chosen slice
of held-back drives keeps unbiased labels for exactly the population that
matters. These tests pin the properties that make it usable months later:
stability, reproducibility, calibration of the fraction, and that a
held-back drive is recorded with the action it would have received.
"""

import datetime as dt
from unittest.mock import patch

import pytest

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.agent.control_arm import (
    DEFAULT_CONTROL_FRACTION,
    control_bucket,
    is_control_drive,
)
from src.agent.deps import AgentDependencies, InMemoryActionLedger, default_guardrail_pass_all
from src.agent.nodes import make_plan_node

DRIVES = [f"drive-{i:05d}" for i in range(20_000)]


def test_nothing_is_held_back_by_default():
    """A control arm costs real failures, so it must be switched on
    deliberately rather than drifted into."""
    assert DEFAULT_CONTROL_FRACTION == 0.0
    assert not any(is_control_drive(d, control_fraction=0.0) for d in DRIVES[:500])


def test_membership_is_stable_across_calls():
    """A drive that flickered in and out of the control arm would be acted
    on eventually, censoring it anyway and defeating the point."""
    chosen = [is_control_drive(d, control_fraction=0.05) for d in DRIVES[:1000]]
    again = [is_control_drive(d, control_fraction=0.05) for d in DRIVES[:1000]]
    assert chosen == again


def test_membership_depends_only_on_the_id_and_salt():
    """No stored state, so the split is reconstructible from the drive id
    alone in an analysis months later."""
    assert control_bucket("drive-00042") == control_bucket("drive-00042")
    assert control_bucket("drive-00042", salt="other") != control_bucket("drive-00042")


@pytest.mark.parametrize("fraction", [0.02, 0.05, 0.10])
def test_the_held_back_share_matches_the_configured_fraction(fraction: float):
    held = sum(is_control_drive(d, control_fraction=fraction) for d in DRIVES)
    assert held / len(DRIVES) == pytest.approx(fraction, abs=0.01)


def test_a_larger_fraction_is_a_superset_of_a_smaller_one():
    """Raising the fraction must not reassign drives already held back -
    otherwise the earlier arm's drives start getting acted on, censoring the
    very outcomes they were kept for."""
    small = {d for d in DRIVES[:5000] if is_control_drive(d, control_fraction=0.02)}
    large = {d for d in DRIVES[:5000] if is_control_drive(d, control_fraction=0.10)}
    assert small and small < large


def test_the_extremes_need_no_hashing():
    assert is_control_drive("anything", control_fraction=1.0) is True
    assert is_control_drive("anything", control_fraction=-1.0) is False


def _plan_state(p_fail: float = 0.99) -> dict:
    return {
        "run_id": "run-1",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "prediction_summary": {
            "drives": [
                {
                    "drive_id": "drive-00042",
                    "p_fail": p_fail,
                    "feature_confidence": 0.95,
                    "feature_maturity": FeatureMaturity.MATURE.value,
                    "stale_telemetry": False,
                }
            ]
        },
        "drive_risk_state": {},
    }


def _deps() -> AgentDependencies:
    return AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "s", "drives": []},
        predictor=lambda s: {"drives": s["drives"]},
        guardrail_evaluator=default_guardrail_pass_all,
        executor=lambda a: {"success": True, **a},
        validator=lambda r: {},
        action_ledger=InMemoryActionLedger(),
    )


def test_a_control_drive_is_recorded_with_the_action_it_would_have_received():
    """The counterfactual is the entire value of holding the drive back, so
    the record must say what the agent would have done, not just that it
    did nothing."""
    plan = make_plan_node(
        _deps(),
        action_thresholds={"warn": 0.1, "cordon": 0.2, "migrate": 0.3, "drain": 0.4},
        hysteresis_cycles_required=1,
        control_fraction=1.0,
    )
    proposal = plan(_plan_state())["proposed_action"]
    assert proposal["control_arm"] is True
    assert proposal["proposed_action"] == ActionTier.MONITOR.value
    assert proposal["withheld_action"] == ActionTier.DRAIN.value


def test_a_normal_drive_is_unaffected_and_flagged_as_such():
    plan = make_plan_node(
        _deps(),
        action_thresholds={"warn": 0.1, "cordon": 0.2, "migrate": 0.3, "drain": 0.4},
        hysteresis_cycles_required=1,
        control_fraction=0.0,
    )
    proposal = plan(_plan_state())["proposed_action"]
    assert proposal["control_arm"] is False
    assert proposal["withheld_action"] is None
    assert proposal["proposed_action"] == ActionTier.DRAIN.value


def test_the_control_arm_is_applied_after_hysteresis_and_cooldown():
    """Order matters: applied first, the recorded counterfactual would be the
    raw tier rather than the one the agent would actually have taken."""
    plan = make_plan_node(
        _deps(),
        action_thresholds={"warn": 0.1, "cordon": 0.2, "migrate": 0.3, "drain": 0.4},
        hysteresis_cycles_required=5,  # caps escalation at WARN on cycle 1
        control_fraction=1.0,
    )
    proposal = plan(_plan_state())["proposed_action"]
    assert proposal["withheld_action"] == ActionTier.WARN.value


def test_the_plan_node_reads_the_fraction_from_config_when_not_given():
    with patch("src.agent.nodes.AgentSettings.load") as load:
        load.return_value.action_thresholds = {
            "warn": 0.1,
            "cordon": 0.2,
            "migrate": 0.3,
            "drain": 0.4,
        }
        load.return_value.min_confidence_for_destructive_action = 0.8
        load.return_value.hysteresis_cycles_required = 1
        load.return_value.cooldown_seconds = 3600
        load.return_value.control_fraction = 1.0
        load.return_value.control_salt = "control-arm-v1"
        proposal = make_plan_node(_deps())(_plan_state())["proposed_action"]
    assert proposal["control_arm"] is True
