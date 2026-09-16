"""Human-review SLA timeout and safe-fallback logic
(docs/design_goal.md section 15 "SLA Targets" / "Human Review Requirements").

If a critical blocked action isn't reviewed within its SLA, the system must
default to the safest non-destructive action rather than stall indefinitely
or, worse, execute the original destructive action without approval.
"""

from __future__ import annotations

import datetime as dt

from data_contracts.schemas import ActionTier

DEFAULT_CRITICAL_SLA_HOURS = 1
DEFAULT_NON_CRITICAL_SLA_HOURS = 24
DEFAULT_FALLBACK_ACTION = ActionTier.CORDON


def is_review_overdue(
    created_at: dt.datetime, *, sla_hours: float, now: dt.datetime | None = None
) -> bool:
    now = now or dt.datetime.now(dt.UTC)
    return now >= created_at + dt.timedelta(hours=sla_hours)


def resolve_timeout_fallback(
    *,
    created_at: dt.datetime,
    is_critical: bool,
    now: dt.datetime | None = None,
    critical_sla_hours: float = DEFAULT_CRITICAL_SLA_HOURS,
    non_critical_sla_hours: float = DEFAULT_NON_CRITICAL_SLA_HOURS,
    fallback_action: ActionTier = DEFAULT_FALLBACK_ACTION,
) -> ActionTier | None:
    """Returns the safe fallback action tier if the review's SLA has
    elapsed, or None if it's still within SLA (still awaiting review)."""
    sla_hours = critical_sla_hours if is_critical else non_critical_sla_hours
    if is_review_overdue(created_at, sla_hours=sla_hours, now=now):
        return fallback_action
    return None
