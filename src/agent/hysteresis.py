"""Hysteresis and cooldown for action-tier escalation (docs/design_goal.md
section 13 "Hysteresis and Cooldown"):

    a drive must remain high-risk for multiple cycles before escalation;
    after an action, the drive enters a cooldown period;
    repeated actions require new evidence;
    downgrades from high-risk to healthy require sustained stability.

`configs/agent.yaml`'s `loop.hysteresis_cycles_required`/`loop.cooldown_seconds`
were previously loaded into `AgentSettings` and never consulted by any
decision logic - this module is what actually enforces them. State is a
small per-drive dict (consecutive escalation-cycle count, last action
timestamp) carried in `AgentState.drive_risk_state`, which is lean enough to
checkpoint per docs/design_goal.md section 5.6.
"""

from __future__ import annotations

import datetime as dt

from data_contracts.schemas import ActionTier

#: tiers that represent an escalation requiring sustained evidence, per the
#: design doc's "Recommended Action Tiers" table (monitor/warn are always
#: safe to apply immediately since they're non-disruptive).
ESCALATION_TIERS = {ActionTier.CORDON, ActionTier.MIGRATE, ActionTier.DRAIN}

DriveRiskState = dict[str, dict]


def apply_hysteresis(
    drive_risk_state: DriveRiskState,
    drive_id: str,
    proposed_tier: ActionTier,
    *,
    hysteresis_cycles_required: int,
) -> tuple[ActionTier, DriveRiskState]:
    """Returns `(effective_tier, updated_state)`. An escalation tier
    (cordon/migrate/drain) is only let through once the drive has
    independently qualified for an escalation on `hysteresis_cycles_required`
    consecutive cycles; until then it's capped at WARN, which still surfaces
    the risk without taking a disruptive action on a single noisy reading.
    A non-escalation proposal resets the streak (sustained *stability* is
    required to walk back down, so a single healthy reading doesn't erase
    a real streak - it only resets when the tier genuinely drops out of the
    escalation set)."""
    state = dict(drive_risk_state.get(drive_id, {}))
    streak = state.get("consecutive_escalation_cycles", 0)

    is_escalation = proposed_tier in ESCALATION_TIERS
    streak = streak + 1 if is_escalation else 0

    effective_tier = proposed_tier
    if is_escalation and streak < hysteresis_cycles_required:
        effective_tier = ActionTier.WARN

    state["consecutive_escalation_cycles"] = streak
    updated = {**drive_risk_state, drive_id: state}
    return effective_tier, updated


def is_in_cooldown(
    drive_risk_state: DriveRiskState,
    drive_id: str,
    *,
    cooldown_seconds: int,
    now: dt.datetime,
) -> bool:
    """True if this drive had an action executed within the cooldown
    window - "repeated actions require new evidence" is enforced by the
    caller downgrading to WARN while this is true, rather than by silently
    re-running the same action every cycle."""
    state = drive_risk_state.get(drive_id)
    if not state or "last_action_at" not in state:
        return False
    last_action_at = dt.datetime.fromisoformat(state["last_action_at"])
    return (now - last_action_at).total_seconds() < cooldown_seconds


def record_action_taken(
    drive_risk_state: DriveRiskState, drive_id: str, *, now: dt.datetime
) -> DriveRiskState:
    state = dict(drive_risk_state.get(drive_id, {}))
    state["last_action_at"] = now.isoformat()
    return {**drive_risk_state, drive_id: state}
