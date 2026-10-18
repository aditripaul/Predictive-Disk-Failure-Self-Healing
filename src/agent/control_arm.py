"""The slice of flagged drives the agent deliberately does not act on.

Acting censors the outcome. Drain a drive that was about to fail and it
never fails, so the label says 0 - and a model retrained on that data learns
that the features which triggered the drain do not predict failure. The
signal that made the system work is the signal the system deletes.

`src/reliability/checks.py` already shows the confusion in miniature:
`check_necessity`'s docstring asks whether the drive "would not have
failed" - a counterfactual - while the code tests whether it did. Those
coincide only while actions do not affect outcomes, which is true today
only because the executor is simulated.

A control arm is the standard fix and the cheap one: hold back a small,
fixed, random share of the drives the agent would have acted on. Their
outcomes stay uncensored, so they are unbiased labels for exactly the
population that matters - drives the model flags. Everything acted on is a
biased sample and needs propensity weighting to be used for training; the
control arm does not.

It costs real failures. Choosing not to save a few drives to keep the system
learnable is a deliberate trade, which is why `control_fraction` defaults to
0.0: nothing is held back until someone decides to.

Membership is a stable hash of the drive id, not a coin flip, so:

  * the same drive is always control, across cycles, processes and restarts
    - a drive that flickered in and out would be acted on eventually, which
      censors it anyway and defeats the point;
  * no state has to be stored or checkpointed to remember the assignment;
  * the split is reproducible from the drive id and the salt alone, so an
    analysis months later can reconstruct exactly which drives were held
    back.
"""

from __future__ import annotations

import hashlib

#: Changing this re-randomizes every assignment, which breaks comparability
#: with earlier data. Treat it as part of the experiment's identity.
DEFAULT_SALT = "control-arm-v1"

#: Hold nothing back until someone decides to: a control arm trades real
#: failures for unbiased labels, and that is not a default to drift into.
DEFAULT_CONTROL_FRACTION = 0.0

_HASH_SPACE = 1 << 32


def control_bucket(drive_id: str, *, salt: str = DEFAULT_SALT) -> float:
    """The drive's fixed position in [0, 1). Depends only on the id and the
    salt, so it is identical in every process and every run."""
    digest = hashlib.sha256(f"{salt}:{drive_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / _HASH_SPACE


def is_control_drive(drive_id: str, *, control_fraction: float, salt: str = DEFAULT_SALT) -> bool:
    """Whether this drive is held back from action.

    `control_fraction <= 0` holds nothing back and `>= 1` holds everything,
    both without hashing, so a disabled control arm costs nothing per
    drive."""
    if control_fraction <= 0.0:
        return False
    if control_fraction >= 1.0:
        return True
    return control_bucket(drive_id, salt=salt) < control_fraction
