"""Lean LangGraph agent state (docs/design_goal.md section 10 / section 5.6).

Only ids, metadata, and small summaries live here. Bulk telemetry and
feature data live in DuckDB/Parquet/Redis and are looked up by id, so
checkpointing this state stays cheap and crash recovery stays fast.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    timestamp: str
    fleet_snapshot_id: str
    flagged_drive_ids: list[str]
    prediction_summary: dict[str, Any]
    proposed_action: dict[str, Any]
    guardrail_result: dict[str, Any]
    human_review_required: bool
    human_decision: str
    human_review_operator_id: str
    human_review_reason_code: str
    human_review_comment: str
    execution_result: dict[str, Any]
    validation_result: dict[str, Any]
    post_action_guardrail_result: dict[str, Any] | None
    decision_record_id: str
    #: per-drive {consecutive_escalation_cycles, last_action_at} - small
    #: counters, not telemetry, so still lean enough to checkpoint. Enforces
    #: hysteresis/cooldown (docs/design_goal.md section 13).
    drive_risk_state: dict[str, dict[str, Any]]
