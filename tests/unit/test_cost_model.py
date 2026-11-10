"""src/models/threshold.py: cost ratios from costs, and the ratio a threshold implies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.threshold import (
    ACTION_TIERS,
    cost_ratios_from_costs,
    implied_cost_ratio,
    resolve_cost_ratios,
    tune_cost_tiers,
)

COSTS = {
    "failure_cost": 1000.0,
    "action_cost": {"warn": 1.0, "cordon": 10.0, "migrate": 50.0, "drain": 200.0},
    "p_save": {"warn": 0.1, "cordon": 0.5, "migrate": 0.8, "drain": 0.9},
}
PLACEHOLDERS = {"warn": 100, "cordon": 20, "migrate": 5, "drain": 2}


def test_ratio_is_p_save_times_failure_cost_over_action_cost_minus_one():
    ratios = cost_ratios_from_costs(**COSTS)
    assert ratios == pytest.approx({"warn": 99.0, "cordon": 49.0, "migrate": 15.0, "drain": 3.5})


def test_break_even_precision_is_where_an_action_stops_paying():
    # At precision 1 / (1 + r) a flagged set has zero net value.
    ratios = cost_ratios_from_costs(**COSTS)
    for tier in ACTION_TIERS:
        precision = 1 / (1 + ratios[tier])
        caught, false_alarms = precision * 1000, (1 - precision) * 1000
        value = (
            caught * (COSTS["p_save"][tier] * COSTS["failure_cost"] - COSTS["action_cost"][tier])
            - false_alarms * COSTS["action_cost"][tier]
        )
        assert value == pytest.approx(0.0, abs=1e-6)


def test_an_action_that_never_pays_is_rejected():
    bad = {**COSTS, "action_cost": {**COSTS["action_cost"], "drain": 900.0}}
    with pytest.raises(ValueError, match="drain never pays"):
        cost_ratios_from_costs(**bad)


def test_ratios_increasing_towards_drain_are_rejected():
    bad = {**COSTS, "action_cost": {**COSTS["action_cost"], "drain": 1.0}}
    with pytest.raises(ValueError, match="increase from warn to drain"):
        cost_ratios_from_costs(**bad)


def test_resolve_falls_back_to_placeholders_when_cost_model_is_empty():
    empty = {
        "failure_cost": None,
        "action_cost": dict.fromkeys(ACTION_TIERS),
        "p_save": dict.fromkeys(ACTION_TIERS),
    }
    for cfg in (
        {"action_cost_ratio": PLACEHOLDERS},
        {"action_cost_ratio": PLACEHOLDERS, "cost_model": empty},
    ):
        ratios, source = resolve_cost_ratios(cfg)
        assert ratios == PLACEHOLDERS and "placeholder" in source


def test_resolve_uses_a_filled_cost_model():
    ratios, source = resolve_cost_ratios({"action_cost_ratio": PLACEHOLDERS, "cost_model": COSTS})
    assert source == "cost_model"
    assert ratios == pytest.approx(cost_ratios_from_costs(**COSTS))


def test_resolve_rejects_a_partly_filled_cost_model():
    partial = {**COSTS, "p_save": {**COSTS["p_save"], "warn": None}}
    with pytest.raises(ValueError, match="partly filled"):
        resolve_cost_ratios({"action_cost_ratio": PLACEHOLDERS, "cost_model": partial})


def _drives(n: int, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < rate).astype(int)
    scores = np.round(rng.random(n) * 0.6 + labels * rng.random(n) * 0.4, 3)  # with ties
    return np.arange(n), labels, scores


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_cost_optimal_threshold_for_r_implies_a_range_containing_r(seed):
    ids, labels, scores = _drives(5000, 0.03, seed)
    for r in (2.0, 5.0, 20.0, 100.0):
        threshold = tune_cost_tiers(
            ids, labels, scores, cost_ratios=dict.fromkeys(ACTION_TIERS, r)
        )["warn"]
        imp = implied_cost_ratio(labels, scores, threshold)
        assert imp["on_hull"]
        assert imp["r_low"] - 1e-9 <= r <= imp["r_high"] + 1e-9


def test_implied_ratio_rises_as_the_threshold_falls():
    _, labels, scores = _drives(5000, 0.03, 3)
    strict = implied_cost_ratio(labels, scores, float(np.quantile(scores, 0.99)))
    loose = implied_cost_ratio(labels, scores, float(np.quantile(scores, 0.7)))
    assert loose["r_low"] >= strict["r_high"] - 1e-9


def test_flagging_everything_implies_no_upper_bound():
    labels = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    imp = implied_cost_ratio(labels, scores, 0.1)
    assert math.isinf(imp["r_high"]) and imp["marginal_precision_low"] == 0.0


def test_a_cut_inside_a_hull_segment_takes_its_slope():
    # Cuts (false alarms, caught): (0,1) (1,1) (2,1) (2,2) (3,2) (4,2); hull (0,1)-(2,2)-(4,2).
    # The cut at 0.8 sits at (1,1), under the segment of slope 1/2: optimal for no r, r = 2.
    labels = np.array([1, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    imp = implied_cost_ratio(labels, scores, 0.8)
    assert not imp["on_hull"]
    assert imp["r_low"] == pytest.approx(2.0) and imp["r_high"] == pytest.approx(2.0)


def test_a_cut_dominated_by_a_vertex_takes_that_vertex_range():
    # Hull (0,1)-(1,2)-(4,2). The cut at 0.8 is (1,1), below the vertex (1,2), which is
    # optimal for r from 1 (segment slope 1) upward without bound (the last segment is flat).
    labels = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    imp = implied_cost_ratio(labels, scores, 0.8)
    assert not imp["on_hull"]
    assert imp["r_low"] == pytest.approx(1.0) and math.isinf(imp["r_high"])
