"""Property-based tests for hysteresis/cooldown (docs/design_goal.md
section 13, `src/agent/hysteresis.py`). `hypothesis` was a declared but
entirely unused dev dependency; these properties check the invariants
directly, against arbitrary tier sequences and timings, rather than only
the specific example sequences in `tests/unit/test_hysteresis.py`.
"""

from __future__ import annotations

import datetime as dt

from hypothesis import given
from hypothesis import strategies as st

from data_contracts.schemas import ActionTier
from src.agent.hysteresis import (
    ESCALATION_TIERS,
    apply_hysteresis,
    is_in_cooldown,
    record_action_taken,
)

TIERS = st.sampled_from(list(ActionTier))


@given(
    tiers=st.lists(TIERS, min_size=1, max_size=20),
    hysteresis_cycles_required=st.integers(min_value=1, max_value=6),
)
def test_apply_hysteresis_only_escalates_after_a_sustained_trailing_streak(
    tiers, hysteresis_cycles_required
):
    """Independently tracks the trailing run-length of escalation-tier
    proposals and asserts the module's output matches it at every step:
    an escalation tier only passes through once its trailing streak
    reaches `hysteresis_cycles_required`; otherwise it's capped at WARN.
    A non-escalation proposal always resets the streak to zero."""
    state: dict = {}
    trailing_streak = 0
    for tier in tiers:
        is_escalation = tier in ESCALATION_TIERS
        trailing_streak = trailing_streak + 1 if is_escalation else 0

        effective, state = apply_hysteresis(
            state, "D-1", tier, hysteresis_cycles_required=hysteresis_cycles_required
        )

        if is_escalation and trailing_streak < hysteresis_cycles_required:
            assert effective == ActionTier.WARN
        else:
            assert effective == tier
        assert state["D-1"]["consecutive_escalation_cycles"] == trailing_streak


@given(
    tiers_a=st.lists(TIERS, min_size=1, max_size=10),
    tiers_b=st.lists(TIERS, min_size=1, max_size=10),
    hysteresis_cycles_required=st.integers(min_value=1, max_value=4),
)
def test_apply_hysteresis_keeps_independent_state_per_drive(
    tiers_a, tiers_b, hysteresis_cycles_required
):
    """Interleaving updates for two drives must produce the same result as
    applying each drive's sequence in isolation - no cross-contamination
    of streak counters between drives."""
    state: dict = {}
    for tier in tiers_a:
        effective_a, state = apply_hysteresis(
            state, "A", tier, hysteresis_cycles_required=hysteresis_cycles_required
        )
    for tier in tiers_b:
        effective_b, state = apply_hysteresis(
            state, "B", tier, hysteresis_cycles_required=hysteresis_cycles_required
        )

    isolated_state: dict = {}
    for tier in tiers_a:
        isolated_effective_a, isolated_state = apply_hysteresis(
            isolated_state, "A", tier, hysteresis_cycles_required=hysteresis_cycles_required
        )
    for tier in tiers_b:
        isolated_effective_b, isolated_state = apply_hysteresis(
            isolated_state, "B", tier, hysteresis_cycles_required=hysteresis_cycles_required
        )

    assert effective_a == isolated_effective_a
    assert effective_b == isolated_effective_b
    assert state["A"] == isolated_state["A"]
    assert state["B"] == isolated_state["B"]


@given(
    cooldown_seconds=st.integers(min_value=0, max_value=10_000),
    delta_seconds=st.floats(min_value=0, max_value=20_000, allow_nan=False, allow_infinity=False),
)
def test_is_in_cooldown_matches_elapsed_time_exactly(cooldown_seconds, delta_seconds):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    state = record_action_taken({}, "D-1", now=now)
    later = now + dt.timedelta(seconds=delta_seconds)

    result = is_in_cooldown(state, "D-1", cooldown_seconds=cooldown_seconds, now=later)
    assert result == (delta_seconds < cooldown_seconds)
