"""Per-family action-tier thresholds (ADR 0002).

One fleet-wide threshold puts different drive models at very different
operating points, because their score distributions differ. These tests pin
the tuning, the fallback for families too small to tune on, and the lookup
that scoring and evaluation share.
"""

import numpy as np
import pytest

from src.models.threshold import (
    resolve_family_thresholds,
    tune_action_tiers,
    tune_action_tiers_by_family,
)

LIFT = {"warn": 2.0, "cordon": 3.0, "migrate": 4.5, "drain": 6.0}
FLEET_TARGETS = {"warn": 0.2, "cordon": 0.3, "migrate": 0.45, "drain": 0.6}


def _family(name: str, n_drives: int, failing: int, *, offset: float, seed: int = 0):
    """Drives of one model. Failing drives score higher, shifted by `offset`
    so the two families need different cut-offs for the same precision."""
    rng = np.random.default_rng(seed)
    drive_ids = np.array([f"{name}-{i:04d}" for i in range(n_drives)])
    y = np.zeros(n_drives, dtype=int)
    y[:failing] = 1
    scores = np.clip(offset + 0.35 * y + rng.normal(0, 0.05, n_drives), 0.001, 0.999)
    models = np.array([name] * n_drives)
    return drive_ids, models, y, scores


def _fleet(seed: int = 0):
    """Two models, same separation but shifted scores, plus a tiny third."""
    big = _family("BIG", 2000, 200, offset=0.30, seed=seed)
    shifted = _family("SHIFTED", 2000, 200, offset=0.55, seed=seed + 1)
    tiny = _family("TINY", 60, 4, offset=0.30, seed=seed + 2)
    return [np.concatenate(parts) for parts in zip(big, shifted, tiny, strict=True)]


def test_each_family_gets_its_own_thresholds():
    drive_ids, models, y, scores = _fleet()
    fleet = tune_action_tiers(drive_ids, y, scores, precision_targets=FLEET_TARGETS)
    plan = tune_action_tiers_by_family(
        drive_ids,
        models,
        y,
        scores,
        lift_targets=LIFT,
        fleet_thresholds=fleet,
        min_failing_drives=50,
    )
    assert set(plan["families"]) == {"BIG", "SHIFTED"}
    # The shifted family scores higher throughout, so its cut-off must be
    # higher for the same enrichment - the whole point of tuning per family.
    assert (
        plan["families"]["SHIFTED"]["thresholds"]["drain"]
        > plan["families"]["BIG"]["thresholds"]["drain"]
    )


def test_a_family_with_too_few_failures_keeps_the_fleet_thresholds():
    drive_ids, models, y, scores = _fleet()
    fleet = tune_action_tiers(drive_ids, y, scores, precision_targets=FLEET_TARGETS)
    plan = tune_action_tiers_by_family(
        drive_ids, models, y, scores, lift_targets=LIFT, fleet_thresholds=fleet
    )
    assert "TINY" in plan["skipped"]
    assert plan["skipped"]["TINY"]["failing_validation_drives"] == 4
    # And the lookup hands it the fleet threshold rather than nothing.
    assert resolve_family_thresholds(plan, "TINY", "drain") == fleet["drain"]


def test_lift_targets_convert_at_each_familys_own_failure_rate():
    """A family whose drives fail more often needs more precision for the same
    lift, so the tier keeps one meaning across a heterogeneous fleet."""
    drive_ids, models, y, scores = _fleet()
    plan = tune_action_tiers_by_family(
        drive_ids,
        models,
        y,
        scores,
        lift_targets=LIFT,
        fleet_thresholds=dict.fromkeys(FLEET_TARGETS, 0.9),
        min_failing_drives=50,
    )
    big = plan["families"]["BIG"]
    assert big["validation_failure_rate"] == pytest.approx(0.1)
    assert big["precision_targets"]["drain"] == pytest.approx(6.0 * 0.1)


def test_an_unknown_model_falls_back_to_the_fleet_threshold():
    plan = {"families": {}, "fleet_thresholds": {"drain": 0.77}}
    assert resolve_family_thresholds(plan, "NEVER-SEEN", "drain") == 0.77


def test_an_unreachable_family_tier_falls_back_rather_than_firing():
    """A tier whose target that family cannot reach must not silently become
    an always-on threshold of None."""
    plan = {
        "families": {"BIG": {"thresholds": {"warn": 0.4, "drain": None}}},
        "fleet_thresholds": {"warn": 0.5, "drain": 0.95},
    }
    assert resolve_family_thresholds(plan, "BIG", "warn") == 0.4
    assert resolve_family_thresholds(plan, "BIG", "drain") == 0.95


def test_tuning_is_deterministic():
    args = _fleet()
    kwargs = dict(
        lift_targets=LIFT,
        fleet_thresholds=dict.fromkeys(FLEET_TARGETS, 0.9),
        min_failing_drives=50,
    )
    first = tune_action_tiers_by_family(*args, **kwargs)
    second = tune_action_tiers_by_family(*args, **kwargs)
    assert first == second
