"""Safety and guardrail multipliers for the trust score
(docs/design_goal.md section 16). Any safety violation or hard-guardrail
violation vetoes the decision outright (TrustScore = 0).
"""

from __future__ import annotations

from data_contracts.schemas import GuardrailSeverity


def safety_multiplier(safety_violation: bool) -> float:
    return 0.0 if safety_violation else 1.0


def guardrail_multiplier(severity: GuardrailSeverity) -> float:
    return {
        GuardrailSeverity.HARD: 0.0,
        GuardrailSeverity.SOFT: 0.5,
        GuardrailSeverity.NONE: 1.0,
    }[severity]


def guardrail_severity_from_violations(violations: list[dict]) -> GuardrailSeverity:
    severities = {v["severity"] for v in violations}
    if "hard" in severities:
        return GuardrailSeverity.HARD
    if "soft" in severities:
        return GuardrailSeverity.SOFT
    return GuardrailSeverity.NONE


def safety_violation_from_validation(validation_result: dict | None) -> bool:
    """A safety violation is data loss or a quorum breach
    (docs/design_goal.md section 6 invariants)."""
    if validation_result is None:
        return False
    return not (
        validation_result.get("data_integrity_ok", True)
        and validation_result.get("quorum_ok", True)
    )
