"""Correctness, necessity, and timeliness checks (docs/design_goal.md
section 16). All three return `None` when the underlying label is censored
(outcome not yet observable), since an unobservable outcome cannot be
scored as either correct/incorrect or necessary/unnecessary.
"""

from __future__ import annotations

from data_contracts.schemas import ActionTier

#: tiers that represent taking a real remediation action, vs. plain monitoring
ACTIVE_TIERS = {ActionTier.CORDON, ActionTier.MIGRATE, ActionTier.DRAIN}

DEFAULT_MIN_LEAD_TIME_DAYS = 1


def took_action(final_action: ActionTier) -> bool:
    return final_action in ACTIVE_TIERS


def check_correctness(label: int | None, action_taken: bool) -> float | None:
    """1.0 for a true positive (action taken, drive failed) or true
    negative (no action, drive stayed healthy); 0.0 otherwise."""
    if label is None:
        return None
    predicted_failure = action_taken
    actual_failure = bool(label)
    return 1.0 if predicted_failure == actual_failure else 0.0


def check_necessity(label: int | None, action_taken: bool) -> float | None:
    """1.0 if no action was taken, or action was taken and the drive did
    fail; 0.0 if action was taken on a drive that would not have failed
    (an unnecessary, operationally wasteful action)."""
    if label is None:
        return None
    if not action_taken:
        return 1.0
    return 1.0 if label == 1 else 0.0


def check_timeliness(
    label: int | None,
    days_to_event: int | None,
    *,
    min_lead_time_days: int = DEFAULT_MIN_LEAD_TIME_DAYS,
) -> float | None:
    """1.0 if the drive stayed healthy (nothing to be timely about), or the
    action preceded the failure by at least `min_lead_time_days`; 0.0 if the
    action came too late to matter operationally."""
    if label is None:
        return None
    if label == 0:
        return 1.0
    if days_to_event is None:
        return 0.0
    return 1.0 if days_to_event >= min_lead_time_days else 0.0
