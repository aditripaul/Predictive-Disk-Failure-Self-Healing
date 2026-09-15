"""Trust score computation (docs/design_goal.md section 16, 18).

BaseScore = 0.40 x Correctness + 0.30 x Timeliness + 0.30 x Necessity
TrustScore = BaseScore x SafetyMultiplier x GuardrailMultiplier

Correctness and Timeliness depend on the failure label, which is only
observable once the prediction horizon completes (docs section 18):

  Provisional trust score - produced immediately after execution, using
  safety, guardrails, and a necessity *proxy* (the model's own confidence,
  since true necessity isn't knowable until the label resolves).

  Final trust score - produced once the label horizon completes, using the
  real Correctness/Necessity/Timeliness checks.
"""

from __future__ import annotations

from data_contracts.schemas import GuardrailSeverity, TrustScoreRecord
from src.reliability.multipliers import guardrail_multiplier, safety_multiplier

CORRECTNESS_WEIGHT = 0.40
TIMELINESS_WEIGHT = 0.30
NECESSITY_WEIGHT = 0.30


def compute_base_score(correctness: float, timeliness: float, necessity: float) -> float:
    return (
        CORRECTNESS_WEIGHT * correctness
        + TIMELINESS_WEIGHT * timeliness
        + NECESSITY_WEIGHT * necessity
    )


def compute_trust_score(
    base_score: float, *, safety_violation: bool, guardrail_severity: GuardrailSeverity
) -> float:
    return (
        base_score
        * safety_multiplier(safety_violation)
        * guardrail_multiplier(guardrail_severity)
    )


def compute_provisional_trust_score(
    decision_id: str,
    action_id: str,
    *,
    necessity_proxy: float,
    safety_violation: bool,
    guardrail_severity: GuardrailSeverity,
) -> TrustScoreRecord:
    """`necessity_proxy` (e.g. the model's own p_fail) stands in for the
    real necessity/correctness/timeliness checks, which require the label
    horizon to complete."""
    provisional = compute_trust_score(
        necessity_proxy, safety_violation=safety_violation, guardrail_severity=guardrail_severity
    )
    return TrustScoreRecord(
        decision_id=decision_id,
        action_id=action_id,
        necessity=necessity_proxy,
        safety_violation=safety_violation,
        guardrail_severity=guardrail_severity,
        trust_score_provisional=provisional,
    )


def compute_final_trust_score(
    decision_id: str,
    action_id: str,
    *,
    correctness: float,
    timeliness: float,
    necessity: float,
    safety_violation: bool,
    guardrail_severity: GuardrailSeverity,
    trust_score_provisional: float,
) -> TrustScoreRecord:
    base_score = compute_base_score(correctness, timeliness, necessity)
    final = compute_trust_score(
        base_score, safety_violation=safety_violation, guardrail_severity=guardrail_severity
    )
    return TrustScoreRecord(
        decision_id=decision_id,
        action_id=action_id,
        correctness=correctness,
        timeliness=timeliness,
        necessity=necessity,
        safety_violation=safety_violation,
        guardrail_severity=guardrail_severity,
        trust_score_provisional=trust_score_provisional,
        trust_score_final=final,
    )
