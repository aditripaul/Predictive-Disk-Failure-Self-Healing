"""Assembles a provisional reliability assessment directly from one MAPE-K
cycle's agent state (docs/design_goal.md section 16, 19).
"""

from __future__ import annotations

import uuid

from data_contracts.schemas import ActionTier
from src.reliability.multipliers import (
    guardrail_severity_from_violations,
    safety_violation_from_validation,
)
from src.reliability.trust_score import compute_provisional_trust_score


def build_provisional_assessment(agent_result: dict) -> dict:
    """`agent_result` is the dict returned by one `compiled_agent(...).invoke()`
    call (see src/agent/graph.py). Returns the provisional TrustScoreRecord
    plus a human-readable explanation, ready to persist as part of a
    DecisionAuditRecord once Phase 10's audit store exists."""
    proposal = agent_result.get("proposed_action")
    guardrail_result = agent_result.get("guardrail_result")
    post_action_guardrail_result = agent_result.get("post_action_guardrail_result")
    validation_result = agent_result.get("validation_result")

    decision_id = str(uuid.uuid4())

    if proposal is None:
        explanation = ["No drive exceeded the risk threshold this cycle; no action proposed."]
        return {"decision_id": decision_id, "trust_score": None, "explanation": explanation}

    # Merge pre-action (guardrail-blocked-before-execution) and post-action
    # (POST_DATA_INTEGRITY/POST_SERVICE_CONTINUITY, evaluated in Validate)
    # violations: either can independently justify a guardrail-multiplier
    # veto in the trust score.
    all_violations = list(guardrail_result["violations"]) if guardrail_result else []
    if post_action_guardrail_result:
        all_violations += post_action_guardrail_result["violations"]
    guardrail_severity = guardrail_severity_from_violations(all_violations)
    safety_violation = safety_violation_from_validation(validation_result)

    trust_score = compute_provisional_trust_score(
        decision_id,
        proposal["action_id"],
        necessity_proxy=proposal["p_fail"],
        safety_violation=safety_violation,
        guardrail_severity=guardrail_severity,
    )

    explanation = generate_explanation(agent_result, guardrail_severity, safety_violation)

    return {
        "decision_id": decision_id,
        "trust_score": trust_score,
        "explanation": explanation,
    }


def generate_explanation(
    agent_result: dict, guardrail_severity, safety_violation: bool
) -> list[str]:
    proposal = agent_result["proposed_action"]
    lines = [
        f"Drive {proposal['drive_id']} was flagged with p_fail={proposal['p_fail']:.2f}.",
        f"Proposed action: {ActionTier(proposal['proposed_action']).value}.",
    ]

    guardrail_result = agent_result.get("guardrail_result") or {}
    if guardrail_result.get("passed"):
        lines.append("All guardrails passed.")
    else:
        rule_ids = [v["rule_id"] for v in guardrail_result.get("violations", [])]
        lines.append(f"Guardrails blocked: {', '.join(rule_ids)}.")

    if agent_result.get("human_review_required"):
        lines.append("Routed to human review.")
    elif agent_result.get("execution_result"):
        success = agent_result["execution_result"].get("success")
        lines.append("Action executed successfully." if success else "Action execution failed.")

    post_action_guardrail_result = agent_result.get("post_action_guardrail_result")
    if post_action_guardrail_result and not post_action_guardrail_result["passed"]:
        rule_ids = [v["rule_id"] for v in post_action_guardrail_result["violations"]]
        lines.append(f"Post-action checks failed: {', '.join(rule_ids)}.")

    if safety_violation:
        lines.append("SAFETY VIOLATION: data integrity or quorum check failed post-action.")

    return lines
