"""MAPE-K node implementations (docs/design_goal.md section 9).

Each `make_*_node` closes over an `AgentDependencies` bundle and returns a
plain `state -> partial_state_update` function, the shape LangGraph expects.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from typing import Any

from langgraph.types import interrupt

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.agent.deps import AgentDependencies
from src.agent.hysteresis import apply_hysteresis, is_in_cooldown, record_action_taken
from src.agent.state import AgentState
from src.config import AgentSettings
from src.models.action_tiers import determine_action_tier


def make_monitor_node(deps: AgentDependencies):
    def monitor(state: AgentState) -> dict[str, Any]:
        snapshot = deps.fleet_state_provider()
        return {
            "run_id": state.get("run_id") or str(uuid.uuid4()),
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "fleet_snapshot_id": snapshot["fleet_snapshot_id"],
        }

    return monitor


def make_analyze_node(deps: AgentDependencies):
    def analyze(state: AgentState) -> dict[str, Any]:
        snapshot = deps.fleet_state_provider()
        prediction_summary = deps.predictor(snapshot)
        prediction_summary = _normalize_prediction_summary(prediction_summary)
        flagged = [d["drive_id"] for d in prediction_summary.get("drives", [])]
        return {"flagged_drive_ids": flagged, "prediction_summary": prediction_summary}

    return analyze


def _normalize_prediction_summary(prediction_summary: dict[str, Any]) -> dict[str, Any]:
    """LangGraph checkpoints AgentState as JSON/msgpack, so nothing in it may
    be a raw enum instance (predictors commonly return FeatureMaturity
    enums directly). Normalize every enum to its `.value` before it enters
    checkpointed state, per the lean/serializable-state design in
    docs/design_goal.md section 5.6/10."""
    normalized_drives = []
    for drive in prediction_summary.get("drives", []):
        normalized = dict(drive)
        maturity = normalized.get("feature_maturity")
        if isinstance(maturity, FeatureMaturity):
            normalized["feature_maturity"] = maturity.value
        normalized_drives.append(normalized)
    return {**prediction_summary, "drives": normalized_drives}


def make_plan_node(
    deps: AgentDependencies,
    *,
    action_thresholds: dict[str, float] | None = None,
    min_confidence_for_destructive_action: float | None = None,
    hysteresis_cycles_required: int | None = None,
    cooldown_seconds: int | None = None,
):
    """Thresholds default to configs/agent.yaml (via AgentSettings.load()),
    resolved lazily so tests/demos can still override them explicitly. The
    config read always happens (cheap YAML parse) rather than only when
    something is missing, so mypy can prove `defaults` is never None here -
    an `if needed: defaults = ...` pattern can't be proven safe across the
    four independent `is None` checks below."""
    defaults = AgentSettings.load()
    resolved_action_thresholds: dict[str, float] = (
        action_thresholds if action_thresholds is not None else defaults.action_thresholds
    )
    resolved_min_confidence: float = (
        min_confidence_for_destructive_action
        if min_confidence_for_destructive_action is not None
        else defaults.min_confidence_for_destructive_action
    )
    resolved_hysteresis_cycles: int = (
        hysteresis_cycles_required
        if hysteresis_cycles_required is not None
        else defaults.hysteresis_cycles_required
    )
    resolved_cooldown_seconds: int = (
        cooldown_seconds if cooldown_seconds is not None else defaults.cooldown_seconds
    )

    def plan(state: AgentState) -> dict[str, Any]:
        drives = state["prediction_summary"].get("drives", [])
        if not drives:
            return {
                "proposed_action": None,
                "guardrail_result": None,
                "human_review_required": False,
            }

        top_drive = max(drives, key=lambda d: d["p_fail"])
        drive_id = top_drive["drive_id"]
        tier = determine_action_tier(
            p_fail=top_drive["p_fail"],
            feature_confidence=top_drive["feature_confidence"],
            feature_maturity=top_drive["feature_maturity"],
            stale_telemetry=top_drive.get("stale_telemetry", False),
            action_thresholds=resolved_action_thresholds,
            min_confidence_for_destructive_action=resolved_min_confidence,
        )

        # Hysteresis (docs/design_goal.md section 13): an escalation tier
        # only takes effect once sustained for `hysteresis_cycles_required`
        # consecutive cycles - otherwise capped at WARN.
        drive_risk_state = state.get("drive_risk_state", {})
        tier, drive_risk_state = apply_hysteresis(
            drive_risk_state,
            drive_id,
            tier,
            hysteresis_cycles_required=resolved_hysteresis_cycles,
        )

        # Cooldown: a drive that was just acted on doesn't get re-escalated
        # every subsequent cycle on the same evidence ("repeated actions
        # require new evidence").
        now = dt.datetime.fromisoformat(state["timestamp"])
        if tier in {ActionTier.CORDON, ActionTier.MIGRATE, ActionTier.DRAIN} and is_in_cooldown(
            drive_risk_state, drive_id, cooldown_seconds=resolved_cooldown_seconds, now=now
        ):
            tier = ActionTier.WARN

        action_id = _deterministic_action_id(state["run_id"], drive_id, tier)
        proposal = {
            "action_id": action_id,
            "drive_id": top_drive["drive_id"],
            "proposed_action": tier.value,
            "p_fail": top_drive["p_fail"],
            "feature_confidence": top_drive["feature_confidence"],
            # Already normalized to a plain string by Analyze
            # (_normalize_prediction_summary), so state stays checkpoint-safe.
            "feature_maturity": top_drive["feature_maturity"],
            "stale_telemetry": top_drive.get("stale_telemetry", False),
            "is_last_healthy_node_in_domain": top_drive.get(
                "is_last_healthy_node_in_domain", False
            ),
            "quorum_ok_after_action": top_drive.get("quorum_ok_after_action", True),
            "high_io_period": top_drive.get("high_io_period", False),
            "maintenance_window_active": top_drive.get("maintenance_window_active", False),
        }
        guardrail_result = deps.guardrail_evaluator(proposal)

        requires_review = (not guardrail_result["passed"]) or tier == ActionTier.HUMAN_REVIEW
        return {
            "proposed_action": proposal,
            "guardrail_result": guardrail_result,
            "human_review_required": requires_review,
            "drive_risk_state": drive_risk_state,
        }

    return plan


def make_human_review_node(deps: AgentDependencies):
    """Pauses the graph via LangGraph's native interrupt mechanism
    (docs/design_goal.md section 15 "Human-in-the-Loop Flow"). Resuming with
    `Command(resume={"approved": bool, "operator_id": str, "reason_code": str,
    "comment": str | None})` continues execution from directly after the
    `interrupt()` call - this is what makes "approve/reject resumes the
    LangGraph workflow" (docs/project_plan.md Phase 10 exit criteria) real
    rather than a dead end."""

    def human_review(state: AgentState) -> dict[str, Any]:
        decision = interrupt(
            {
                "proposal": state["proposed_action"],
                "guardrail_result": state.get("guardrail_result"),
                "fleet_snapshot_id": state.get("fleet_snapshot_id"),
            }
        )
        approved = bool(decision.get("approved"))
        return {
            "human_review_required": False,
            "human_decision": "approved" if approved else "rejected",
            "human_review_operator_id": decision.get("operator_id"),
            "human_review_reason_code": decision.get("reason_code"),
            "human_review_comment": decision.get("comment"),
        }

    return human_review


def make_execute_node(deps: AgentDependencies):
    def execute(state: AgentState) -> dict[str, Any]:
        rejected_by_human = state.get("human_decision") == "rejected"
        if state.get("human_review_required") or rejected_by_human or not state.get(
            "proposed_action"
        ):
            return {"execution_result": None}

        action_id = state["proposed_action"]["action_id"]
        if deps.action_ledger.is_executed(action_id):
            # Crash-recovery replay: never re-execute a completed action.
            return {"execution_result": deps.action_ledger.get_result(action_id)}

        final_action = dict(state["proposed_action"])
        final_action["proposed_action"] = state["guardrail_result"]["final_action"]
        result = deps.executor(final_action)
        deps.action_ledger.mark_executed(action_id, result)

        update: dict[str, Any] = {"execution_result": result}
        tier = ActionTier(final_action["proposed_action"])
        if tier in {ActionTier.CORDON, ActionTier.MIGRATE, ActionTier.DRAIN}:
            now = dt.datetime.fromisoformat(state["timestamp"])
            update["drive_risk_state"] = record_action_taken(
                state.get("drive_risk_state", {}), final_action["drive_id"], now=now
            )
        return update

    return execute


def make_validate_node(deps: AgentDependencies):
    def validate(state: AgentState) -> dict[str, Any]:
        execution_result = state.get("execution_result")
        validation_result = deps.validator(execution_result or {})
        decision_record_id = str(uuid.uuid4())

        post_action_guardrail_result = None
        if execution_result is not None:
            # POST_DATA_INTEGRITY / POST_SERVICE_CONTINUITY (docs/design_goal.md
            # section 14): audited even when the simulator already rolled a
            # failed action back, so the trust score reflects the incident.
            post_action_guardrail_result = deps.post_action_guardrail_evaluator(
                execution_result, validation_result
            )

        return {
            "validation_result": validation_result,
            "post_action_guardrail_result": post_action_guardrail_result,
            "decision_record_id": decision_record_id,
        }

    return validate


def _deterministic_action_id(run_id: str, drive_id: str, tier: ActionTier) -> str:
    """Same (run_id, drive_id, tier) always yields the same action_id, so a
    crash-recovery replay of the same cycle produces the same id and hits
    the idempotency ledger instead of double-executing."""
    digest = hashlib.sha256(f"{run_id}:{drive_id}:{tier.value}".encode()).hexdigest()[:16]
    return f"action-{digest}"
