import datetime as dt

from data_contracts.schemas import ActionTier
from src.agent.hysteresis import apply_hysteresis, is_in_cooldown, record_action_taken

NOW = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.UTC)


def test_first_escalation_is_capped_at_warn_when_hysteresis_requires_two_cycles():
    tier, state = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=2)
    assert tier == ActionTier.WARN
    assert state["D-1"]["consecutive_escalation_cycles"] == 1


def test_second_consecutive_escalation_is_let_through():
    _, state = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=2)
    tier, state = apply_hysteresis(state, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=2)
    assert tier == ActionTier.DRAIN
    assert state["D-1"]["consecutive_escalation_cycles"] == 2


def test_hysteresis_cycles_required_one_never_delays():
    tier, _ = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=1)
    assert tier == ActionTier.DRAIN


def test_non_escalation_tier_passes_through_immediately():
    tier, state = apply_hysteresis({}, "D-1", ActionTier.MONITOR, hysteresis_cycles_required=3)
    assert tier == ActionTier.MONITOR
    assert state["D-1"]["consecutive_escalation_cycles"] == 0


def test_streak_resets_when_risk_drops_below_escalation():
    _, state = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=3)
    _, state = apply_hysteresis(state, "D-1", ActionTier.MONITOR, hysteresis_cycles_required=3)
    tier, state = apply_hysteresis(state, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=3)
    # streak restarted at 1, still below the 3 required
    assert tier == ActionTier.WARN
    assert state["D-1"]["consecutive_escalation_cycles"] == 1


def test_different_drives_track_independent_streaks():
    _, state = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=2)
    tier_d2, state = apply_hysteresis(state, "D-2", ActionTier.DRAIN, hysteresis_cycles_required=2)
    assert tier_d2 == ActionTier.WARN
    assert state["D-1"]["consecutive_escalation_cycles"] == 1
    assert state["D-2"]["consecutive_escalation_cycles"] == 1


def test_is_in_cooldown_true_immediately_after_action():
    state = record_action_taken({}, "D-1", now=NOW)
    assert is_in_cooldown(state, "D-1", cooldown_seconds=3600, now=NOW) is True


def test_is_in_cooldown_false_after_window_elapses():
    state = record_action_taken({}, "D-1", now=NOW)
    later = NOW + dt.timedelta(hours=2)
    assert is_in_cooldown(state, "D-1", cooldown_seconds=3600, now=later) is False


def test_is_in_cooldown_false_for_a_drive_with_no_prior_action():
    assert is_in_cooldown({}, "D-1", cooldown_seconds=3600, now=NOW) is False


def test_record_action_taken_does_not_clobber_other_drive_state():
    state = apply_hysteresis({}, "D-1", ActionTier.DRAIN, hysteresis_cycles_required=2)[1]
    state = record_action_taken(state, "D-1", now=NOW)
    assert state["D-1"]["consecutive_escalation_cycles"] == 1
    assert state["D-1"]["last_action_at"] == NOW.isoformat()
