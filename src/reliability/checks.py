"""Correctness, necessity, and timeliness checks (docs/design_goal.md
section 16). All three return `None` when the underlying label is censored
(outcome not yet observable), since an unobservable outcome cannot be
scored as either correct/incorrect or necessary/unnecessary.

**These metrics assume actions do not change outcomes. Read this before
wiring a real executor.**

`check_necessity` asks whether the drive "would not have failed" - a
counterfactual - and answers it with `label == 1`, whether the drive *did*
fail. Those are the same quantity only while the agent cannot affect the
drive. Today they are: the executor is simulated
(`src/agent/fleet.py::_recording_executor`), the labels come from recorded
history, and nothing the agent decides changes them. So the metrics are
sound as they stand.

They invert the moment an action is real. Drain a drive that was about to
fail, the drain succeeds, the drive does not fail, `label` is 0 - and
`check_necessity` scores 0.0. The successful save is recorded as an
unnecessary action and the trust score drops for doing its job.
`check_correctness` has the same shape: it reads `action_taken` as
"predicted failure" and compares it to an outcome the action itself
prevented.

Retraining compounds it. The model flags a drive, the agent removes it, no
failure is recorded, and the next model learns those features do not predict
failure - the signal that made the system work is the one the system
deletes.

A real executor therefore needs the counterfactual restored before these
numbers mean anything, which is what `src/agent/control_arm.py` is for: the
held-back drives are never acted on, so their labels stay uncensored and
these checks remain valid on them. For acted-on drives the honest options
are to score them only against the control arm's base rate, or to
propensity-weight them. Neither is implemented here.
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
    negative (no action, drive stayed healthy); 0.0 otherwise.

    Valid only while actions do not change outcomes - see this module's
    docstring. On a drive a real executor saved, the save reads as a false
    positive."""
    if label is None:
        return None
    predicted_failure = action_taken
    actual_failure = bool(label)
    return 1.0 if predicted_failure == actual_failure else 0.0


def check_necessity(label: int | None, action_taken: bool) -> float | None:
    """1.0 if no action was taken, or action was taken and the drive did
    fail; 0.0 if action was taken on a drive that would not have failed
    (an unnecessary, operationally wasteful action).

    "Would not have failed" is a counterfactual; `label` records what did
    happen. They agree only while actions do not change outcomes - see this
    module's docstring. Against a real executor this scores every successful
    intervention as unnecessary."""
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
