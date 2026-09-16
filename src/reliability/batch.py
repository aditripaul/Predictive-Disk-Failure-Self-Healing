"""Resolves provisional trust scores to final ones once the label horizon
completes (docs/design_goal.md section 18 "Provisional and Final Trust
Scores"): joins recorded decisions to the resolved failure-label table and
recomputes correctness/necessity/timeliness now that the true outcome is
known. Intended to run periodically (e.g. daily) against whatever decisions
the audit store has accumulated.
"""

from __future__ import annotations

import datetime as dt

from data_contracts.schemas import GuardrailSeverity
from src.reliability.checks import (
    ACTIVE_TIERS,
    check_correctness,
    check_necessity,
    check_timeliness,
)
from src.reliability.trust_score import compute_final_trust_score


def resolve_final_trust_scores(
    decisions: list[dict],
    labels_by_drive_date_horizon: dict[tuple[str, dt.date, int], dict],
) -> list[dict]:
    """`decisions` are audit records as produced by
    src.agent.orchestrator.AgentOrchestrator (each needs `decision_id`,
    `action_id`, `drive_id`, `date`, `horizon_days`, `proposed_action`,
    `trust_score_provisional`, `safety_violation`, `guardrail_severity`).
    `labels_by_drive_date_horizon` maps (drive_id, date, horizon_days) to a
    label-table row (needs `label`, `days_to_event`) - typically built once
    per run via `{(r["drive_id"], r["date"], r["horizon_days"]): r for r in
    labels_df.to_dicts()}`.

    Decisions with no matching label yet (horizon not yet observable, or
    censored) are skipped - they remain provisional-only until a later run
    finds a resolved label for them.
    """
    resolved: list[dict] = []
    for decision in decisions:
        key = (decision["drive_id"], decision["date"], decision["horizon_days"])
        label_row = labels_by_drive_date_horizon.get(key)
        if label_row is None or label_row.get("label") is None:
            continue

        label = label_row["label"]
        action_taken = decision["proposed_action"] in {t.value for t in ACTIVE_TIERS}

        correctness = check_correctness(label, action_taken)
        necessity = check_necessity(label, action_taken)
        timeliness = check_timeliness(label, label_row.get("days_to_event"))
        if correctness is None or necessity is None or timeliness is None:
            continue

        record = compute_final_trust_score(
            decision["decision_id"],
            decision["action_id"],
            correctness=correctness,
            timeliness=timeliness,
            necessity=necessity,
            safety_violation=decision["safety_violation"],
            guardrail_severity=GuardrailSeverity(decision["guardrail_severity"]),
            trust_score_provisional=decision["trust_score_provisional"],
        )
        resolved.append(
            {
                "decision_id": decision["decision_id"],
                "action_id": decision["action_id"],
                "trust_score_final": record.trust_score_final,
                "correctness": correctness,
                "necessity": necessity,
                "timeliness": timeliness,
            }
        )
    return resolved
