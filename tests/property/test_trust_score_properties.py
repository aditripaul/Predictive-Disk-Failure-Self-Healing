"""Property-based tests for the trust score formula (docs/design_goal.md
section 16/18, `src/reliability/trust_score.py`): BaseScore is a convex
combination of three unit-interval inputs, and either a safety violation
or a hard guardrail violation must veto the score to exactly 0 regardless
of how good the base score is.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from data_contracts.schemas import GuardrailSeverity
from src.reliability.trust_score import compute_base_score, compute_trust_score

UNIT_FLOAT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
SEVERITIES = st.sampled_from(list(GuardrailSeverity))


@given(correctness=UNIT_FLOAT, timeliness=UNIT_FLOAT, necessity=UNIT_FLOAT)
def test_base_score_stays_within_the_unit_interval(correctness, timeliness, necessity):
    score = compute_base_score(correctness, timeliness, necessity)
    assert -1e-9 <= score <= 1.0 + 1e-9


@given(base_score=UNIT_FLOAT, safety_violation=st.booleans(), guardrail_severity=SEVERITIES)
def test_trust_score_never_exceeds_the_base_score(base_score, safety_violation, guardrail_severity):
    trust = compute_trust_score(
        base_score, safety_violation=safety_violation, guardrail_severity=guardrail_severity
    )
    assert 0.0 <= trust <= base_score + 1e-9


@given(base_score=UNIT_FLOAT, guardrail_severity=SEVERITIES)
def test_safety_violation_always_zeroes_the_trust_score(base_score, guardrail_severity):
    trust = compute_trust_score(
        base_score, safety_violation=True, guardrail_severity=guardrail_severity
    )
    assert trust == 0.0


@given(base_score=UNIT_FLOAT, safety_violation=st.booleans())
def test_hard_guardrail_severity_always_zeroes_the_trust_score(base_score, safety_violation):
    trust = compute_trust_score(
        base_score, safety_violation=safety_violation, guardrail_severity=GuardrailSeverity.HARD
    )
    assert trust == 0.0


@given(base_score=UNIT_FLOAT)
def test_no_violation_and_no_guardrail_hit_preserves_the_base_score(base_score):
    trust = compute_trust_score(
        base_score, safety_violation=False, guardrail_severity=GuardrailSeverity.NONE
    )
    assert trust == base_score
