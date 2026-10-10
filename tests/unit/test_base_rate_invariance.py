"""Metrics that survive a change of label horizon.

Precision, AUPRC and lift all move with the base rate, so none of them can
compare a 30-day model against a 60-day one: doubling the positive rate
raises precision even when the model is bit-for-bit the same. These tests pin
that the invariant metrics do not move under exactly that redefinition, and
that the base-rate-dependent ones do - because the second half is the reason
the first half had to be added.
"""

import numpy as np
import pytest

from src.models.evaluation import (
    compute_auroc,
    drive_level_metrics,
    recall_at_false_alarm_rate,
    recall_at_false_alarm_table,
)


def _fleet(n_failing: int = 300, n_healthy: int = 300_000, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    drive_ids = np.array(
        [f"fail-{i:05d}" for i in range(n_failing)] + [f"ok-{i:06d}" for i in range(n_healthy)]
    )
    y = np.concatenate([np.ones(n_failing, dtype=int), np.zeros(n_healthy, dtype=int)])
    scores = np.concatenate(
        [
            rng.beta(5, 2, n_failing),  # failing drives score high
            rng.beta(2, 5, n_healthy),  # healthy drives score low
        ]
    )
    return drive_ids, y, scores


def _with_doubled_positives(drive_ids, y, scores):
    """The same fleet with every failing drive duplicated under a new id and
    its score unchanged.

    The base rate doubles while NEITHER class's score distribution moves, so
    the ROC curve is identical by construction rather than merely similar -
    which is what makes the invariance assertions below exact instead of a
    tolerance on sampling noise. This is the clean form of what a longer label
    horizon does: more drives count as positive, the telemetry is the same."""
    failing = y == 1
    return (
        np.concatenate([drive_ids, np.char.add(drive_ids[failing], "-dup")]),
        np.concatenate([y, y[failing]]),
        np.concatenate([scores, scores[failing]]),
    )


def test_auroc_is_unchanged_when_only_the_base_rate_doubles():
    """The headline claim: AUROC answers "is this model better?" where
    precision answers "is this event commoner?"."""
    rare = _fleet()
    common = _with_doubled_positives(*rare)
    assert compute_auroc(*common[1:]) == pytest.approx(compute_auroc(*rare[1:]))


def test_precision_rises_on_the_same_model_when_the_base_rate_doubles():
    """The failure mode the invariants exist for. Same score distributions,
    twice the positives, and precision goes up for free."""
    threshold = 0.9
    base = _fleet()
    rare = drive_level_metrics(*base, threshold)
    common = drive_level_metrics(*_with_doubled_positives(*base), threshold)
    assert common["fleet_failure_rate"] > 1.9 * rare["fleet_failure_rate"]
    assert common["precision"] > rare["precision"] + 0.05
    # Lift is no escape either - it falls, though nothing about the model
    # changed, because it is invariant only in the p -> 0 limit.
    assert common["lift"] < rare["lift"]


def test_the_positive_likelihood_ratio_holds_where_precision_does_not():
    """recall / false_alarm_rate is this threshold's own invariant: both rates
    are measured within one class, so neither depends on the class balance."""
    threshold = 0.9
    base = _fleet()
    rare = drive_level_metrics(*base, threshold)
    common = drive_level_metrics(*_with_doubled_positives(*base), threshold)
    assert rare["positive_likelihood_ratio"] is not None
    assert common["positive_likelihood_ratio"] == pytest.approx(rare["positive_likelihood_ratio"])


def test_recall_at_matched_false_alarm_rate_holds_too():
    """The ROC-space counterpart of precision at matched recall."""
    budget = 2e-4
    base = _fleet()
    rare = recall_at_false_alarm_rate(*base, budget)
    common = recall_at_false_alarm_rate(*_with_doubled_positives(*base), budget)
    assert rare["reachable"] and common["reachable"]
    assert common["recall"] == pytest.approx(rare["recall"])


def test_recall_at_false_alarm_rate_stays_within_its_budget():
    budget = 1e-4
    result = recall_at_false_alarm_rate(*_fleet(), budget)
    assert result["reachable"]
    assert result["false_alarm_rate"] <= budget


def test_a_tighter_budget_never_buys_more_recall():
    rows = recall_at_false_alarm_table(*_fleet())
    recalls = [r["recall"] for r in rows if r["reachable"]]
    assert recalls == sorted(recalls)


def test_an_unreachable_budget_is_reported_rather_than_guessed():
    """When the top-scoring drive is healthy, every threshold that catches a
    failing drive also alerts on it, so no operating point fits under a budget
    below 1/n_healthy. That must be reported, not answered with the
    alert-on-nothing point that `roc_curve` prepends - which meets any budget
    at zero recall and carries a threshold of inf."""
    drive_ids = np.array(["ok-top", "fail-a", "fail-b", "ok-1", "ok-2", "ok-3"])
    y = np.array([0, 1, 1, 0, 0, 0])
    scores = np.array([0.99, 0.90, 0.80, 0.30, 0.20, 0.10])
    result = recall_at_false_alarm_rate(drive_ids, y, scores, 1e-9)
    assert result["reachable"] is False
    assert "recall" not in result

    # Widen the budget to one healthy drive out of four and it becomes a real
    # operating point, with a finite threshold.
    widened = recall_at_false_alarm_rate(drive_ids, y, scores, 0.25)
    assert widened["reachable"] is True
    assert widened["recall"] == 1.0
    assert np.isfinite(widened["threshold"])


def test_auroc_of_a_single_class_split_is_undefined_not_an_error():
    ids, y, scores = _fleet(0, 100)
    assert compute_auroc(y, scores) == 0.5
    assert recall_at_false_alarm_rate(ids, y, scores, 1e-3)["reachable"] is False
