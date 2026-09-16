from data_contracts.schemas import GuardrailSeverity
from src.reliability.audit import build_provisional_assessment
from src.reliability.checks import check_correctness, check_necessity, check_timeliness
from src.reliability.multipliers import (
    guardrail_multiplier,
    guardrail_severity_from_violations,
    safety_multiplier,
    safety_violation_from_validation,
)
from src.reliability.trust_score import (
    compute_base_score,
    compute_final_trust_score,
    compute_provisional_trust_score,
    compute_trust_score,
)


def test_check_correctness_true_positive_and_true_negative():
    assert check_correctness(label=1, action_taken=True) == 1.0
    assert check_correctness(label=0, action_taken=False) == 1.0
    assert check_correctness(label=1, action_taken=False) == 0.0
    assert check_correctness(label=0, action_taken=True) == 0.0
    assert check_correctness(label=None, action_taken=True) is None


def test_check_necessity_flags_unnecessary_action():
    assert check_necessity(label=0, action_taken=True) == 0.0  # unnecessary
    assert check_necessity(label=1, action_taken=True) == 1.0
    assert check_necessity(label=0, action_taken=False) == 1.0
    assert check_necessity(label=None, action_taken=True) is None


def test_check_timeliness_requires_lead_time():
    assert check_timeliness(label=0, days_to_event=None) == 1.0
    assert check_timeliness(label=1, days_to_event=5, min_lead_time_days=1) == 1.0
    assert check_timeliness(label=1, days_to_event=0, min_lead_time_days=1) == 0.0
    assert check_timeliness(label=1, days_to_event=None) == 0.0
    assert check_timeliness(label=None, days_to_event=5) is None


def test_safety_and_guardrail_multipliers():
    assert safety_multiplier(safety_violation=True) == 0.0
    assert safety_multiplier(safety_violation=False) == 1.0
    assert guardrail_multiplier(GuardrailSeverity.HARD) == 0.0
    assert guardrail_multiplier(GuardrailSeverity.SOFT) == 0.5
    assert guardrail_multiplier(GuardrailSeverity.NONE) == 1.0


def test_guardrail_severity_from_violations():
    assert guardrail_severity_from_violations([]) == GuardrailSeverity.NONE
    assert guardrail_severity_from_violations([{"severity": "soft"}]) == GuardrailSeverity.SOFT
    assert (
        guardrail_severity_from_violations([{"severity": "soft"}, {"severity": "hard"}])
        == GuardrailSeverity.HARD
    )


def test_safety_violation_from_validation():
    assert safety_violation_from_validation(None) is False
    assert safety_violation_from_validation({"data_integrity_ok": True, "quorum_ok": True}) is False
    assert safety_violation_from_validation({"data_integrity_ok": False, "quorum_ok": True}) is True
    assert safety_violation_from_validation({"data_integrity_ok": True, "quorum_ok": False}) is True


def test_base_score_weights_correctness_timeliness_necessity():
    score = compute_base_score(correctness=1.0, timeliness=1.0, necessity=1.0)
    assert abs(score - 1.0) < 1e-9
    score = compute_base_score(correctness=0.0, timeliness=0.0, necessity=0.0)
    assert score == 0.0


def test_trust_score_veto_on_safety_violation():
    score = compute_trust_score(
        1.0, safety_violation=True, guardrail_severity=GuardrailSeverity.NONE
    )
    assert score == 0.0


def test_trust_score_veto_on_hard_guardrail_violation():
    score = compute_trust_score(
        1.0, safety_violation=False, guardrail_severity=GuardrailSeverity.HARD
    )
    assert score == 0.0


def test_trust_score_halved_on_soft_guardrail_violation():
    score = compute_trust_score(
        1.0, safety_violation=False, guardrail_severity=GuardrailSeverity.SOFT
    )
    assert score == 0.5


def test_provisional_trust_score_uses_necessity_proxy():
    record = compute_provisional_trust_score(
        "d1",
        "a1",
        necessity_proxy=0.9,
        safety_violation=False,
        guardrail_severity=GuardrailSeverity.NONE,
    )
    assert record.trust_score_provisional == 0.9
    assert record.trust_score_final is None


def test_final_trust_score_matches_formula():
    record = compute_final_trust_score(
        "d1",
        "a1",
        correctness=1.0,
        timeliness=1.0,
        necessity=1.0,
        safety_violation=False,
        guardrail_severity=GuardrailSeverity.NONE,
        trust_score_provisional=0.9,
    )
    assert abs(record.trust_score_final - 1.0) < 1e-9


def test_build_provisional_assessment_no_action_proposed():
    result = build_provisional_assessment({"proposed_action": None})
    assert result["trust_score"] is None
    assert "no action proposed" in result["explanation"][0].lower()


DRAIN_PROPOSAL = {
    "action_id": "a1",
    "drive_id": "D-1",
    "proposed_action": "drain",
    "p_fail": 0.9,
}


def test_build_provisional_assessment_with_passing_guardrails():
    agent_result = {
        "proposed_action": DRAIN_PROPOSAL,
        "guardrail_result": {"passed": True, "violations": []},
        "human_review_required": False,
        "execution_result": {"success": True},
        "validation_result": {"data_integrity_ok": True, "quorum_ok": True},
    }
    result = build_provisional_assessment(agent_result)
    assert result["trust_score"].trust_score_provisional == 0.9
    assert any("executed successfully" in line for line in result["explanation"])


def test_build_provisional_assessment_vetoes_on_safety_violation():
    agent_result = {
        "proposed_action": DRAIN_PROPOSAL,
        "guardrail_result": {"passed": True, "violations": []},
        "human_review_required": False,
        "execution_result": {"success": True},
        "validation_result": {"data_integrity_ok": False, "quorum_ok": True},
    }
    result = build_provisional_assessment(agent_result)
    assert result["trust_score"].trust_score_provisional == 0.0
    assert any("SAFETY VIOLATION" in line for line in result["explanation"])
