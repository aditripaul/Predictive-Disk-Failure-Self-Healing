import datetime as dt

from src.reliability.batch import resolve_final_trust_scores

DATE = dt.date(2024, 1, 1)


def _decision(**overrides) -> dict:
    base = {
        "decision_id": "d1",
        "action_id": "a1",
        "drive_id": "D-1",
        "date": DATE,
        "horizon_days": 14,
        "proposed_action": "drain",
        "trust_score_provisional": 0.9,
        "safety_violation": False,
        "guardrail_severity": "none",
    }
    base.update(overrides)
    return base


def test_resolves_true_positive_to_full_trust():
    decisions = [_decision()]
    labels = {("D-1", DATE, 14): {"label": 1, "days_to_event": 5}}

    resolved = resolve_final_trust_scores(decisions, labels)

    assert len(resolved) == 1
    assert resolved[0]["correctness"] == 1.0
    assert resolved[0]["necessity"] == 1.0
    assert resolved[0]["timeliness"] == 1.0
    assert resolved[0]["trust_score_final"] == 1.0


def test_resolves_unnecessary_action_to_zero_necessity():
    decisions = [_decision()]
    labels = {("D-1", DATE, 14): {"label": 0, "days_to_event": None}}

    resolved = resolve_final_trust_scores(decisions, labels)

    assert resolved[0]["correctness"] == 0.0  # predicted failure, drive was healthy
    assert resolved[0]["necessity"] == 0.0
    assert resolved[0]["timeliness"] == 1.0  # nothing to be timely about; drive stayed healthy
    # BaseScore = 0.4*0 (correctness) + 0.3*1 (timeliness) + 0.3*0 (necessity)
    assert resolved[0]["trust_score_final"] == 0.3


def test_monitor_tier_healthy_drive_is_a_true_negative():
    decisions = [_decision(proposed_action="monitor", trust_score_provisional=0.05)]
    labels = {("D-1", DATE, 14): {"label": 0, "days_to_event": None}}

    resolved = resolve_final_trust_scores(decisions, labels)

    assert resolved[0]["correctness"] == 1.0
    assert resolved[0]["necessity"] == 1.0
    assert resolved[0]["timeliness"] == 1.0
    assert resolved[0]["trust_score_final"] == 1.0


def test_safety_violation_vetoes_final_trust_score():
    decisions = [_decision(safety_violation=True)]
    labels = {("D-1", DATE, 14): {"label": 1, "days_to_event": 5}}

    resolved = resolve_final_trust_scores(decisions, labels)

    assert resolved[0]["trust_score_final"] == 0.0


def test_hard_guardrail_violation_vetoes_final_trust_score():
    decisions = [_decision(guardrail_severity="hard")]
    labels = {("D-1", DATE, 14): {"label": 1, "days_to_event": 5}}

    resolved = resolve_final_trust_scores(decisions, labels)

    assert resolved[0]["trust_score_final"] == 0.0


def test_no_matching_label_leaves_decision_unresolved():
    decisions = [_decision()]
    resolved = resolve_final_trust_scores(decisions, labels_by_drive_date_horizon={})
    assert resolved == []


def test_censored_label_leaves_decision_unresolved():
    decisions = [_decision()]
    labels = {("D-1", DATE, 14): {"label": None, "days_to_event": None}}
    resolved = resolve_final_trust_scores(decisions, labels)
    assert resolved == []
