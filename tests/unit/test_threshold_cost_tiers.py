"""src/models/threshold.py::tune_cost_tiers: each tier where its action pays for itself."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from src.agent.fleet import load_precomputed_scores
from src.models.threshold import ACTION_TIERS, tune_cost_tiers


def _one_row_per_drive(labels, scores):
    n = len(labels)
    return np.array([f"d{i}" for i in range(n)]), np.asarray(labels), np.asarray(scores, float)


def _brute_force(labels, scores, ratio):
    """Best threshold over every distinct score, by r * caught - false_alarms."""
    best, best_value = None, 0.0
    for t in sorted(set(scores)):
        flagged = scores >= t
        value = ratio * int((flagged & (labels == 1)).sum()) - int((flagged & (labels == 0)).sum())
        if value > best_value or (value == best_value and best is not None and t > best):
            best, best_value = t, value
    return best


def test_optimum_matches_a_brute_force_sweep():
    rng = np.random.default_rng(0)
    labels = (rng.random(400) < 0.08).astype(int)
    scores = np.round(rng.random(400) * 0.5 + labels * rng.random(400) * 0.5, 2)  # with ties
    ids, y, s = _one_row_per_drive(labels, scores)
    for r in (2, 5, 20, 100):
        got = tune_cost_tiers(ids, y, s, cost_ratios={t: r for t in ACTION_TIERS})["warn"]
        flagged = s >= got
        value = r * int((flagged & (y == 1)).sum()) - int((flagged & (y == 0)).sum())
        expected = _brute_force(y, s, r)
        e_flagged = s >= expected
        e_value = r * int((e_flagged & (y == 1)).sum()) - int((e_flagged & (y == 0)).sum())
        assert value == e_value  # the same net value (ties between cuts are allowed)


def test_a_cheaper_action_sits_at_a_lower_threshold_and_tiers_nest():
    rng = np.random.default_rng(1)
    labels = (rng.random(2000) < 0.05).astype(int)
    scores = rng.random(2000) * 0.6 + labels * 0.4 * rng.random(2000)
    ids, y, s = _one_row_per_drive(labels, scores)
    tiers = tune_cost_tiers(
        ids, y, s, cost_ratios={"warn": 100, "cordon": 20, "migrate": 5, "drain": 2}
    )
    present = [tiers[t] for t in ACTION_TIERS if tiers[t] is not None]
    assert present == sorted(present)
    assert tiers["warn"] is not None and tiers["warn"] < (tiers["cordon"] or 1.0)


def test_no_profitable_threshold_gives_none():
    ids, y, s = _one_row_per_drive([0, 0, 0, 1], [0.9, 0.8, 0.7, 0.1])
    tiers = tune_cost_tiers(ids, y, s, cost_ratios={t: 0.5 for t in ACTION_TIERS})
    assert all(v is None for v in tiers.values())


def test_a_huge_ratio_flags_down_to_the_lowest_failing_drive():
    ids, y, s = _one_row_per_drive([1, 0, 1, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1])
    assert tune_cost_tiers(ids, y, s, cost_ratios={t: 1e9 for t in ACTION_TIERS})["warn"] == 0.3


def test_the_failing_drive_score_is_its_positive_window_maximum():
    # drive "a" fails: only its positive row counts; drive "b" is healthy: all rows count
    ids = np.array(["a", "a", "b", "b"])
    y = np.array([0, 1, 0, 0])
    s = np.array([0.95, 0.4, 0.5, 0.2])
    tiers = tune_cost_tiers(ids, y, s, cost_ratios={t: 10 for t in ACTION_TIERS})
    assert tiers["warn"] == 0.4  # flags a (0.4) and b (0.5): 10*1 - 1 = 9 > 0


@pytest.mark.parametrize(
    "ratios",
    [
        {"warn": 2, "cordon": 5, "migrate": 5, "drain": 5},
        {"warn": 0, "cordon": 0, "migrate": 0, "drain": 0},
    ],
)
def test_bad_ratios_are_refused(ratios):
    ids, y, s = _one_row_per_drive([1, 0], [0.9, 0.1])
    with pytest.raises(ValueError):
        tune_cost_tiers(ids, y, s, cost_ratios=ratios)


def test_the_agent_reads_the_tier_set_it_is_told_to(tmp_path):
    pl.DataFrame({"drive_id": ["a"], "date": [dt.date(2026, 2, 1)], "score": [0.5]}).write_parquet(
        tmp_path / "scores.parquet"
    )
    path = tmp_path / "agent_thresholds_stack.json"
    path.write_text(
        json.dumps(
            {
                "score_column": "score",
                "action_tiers": {"warn": 0.6, "cordon": 0.8, "migrate": None, "drain": None},
                "action_tiers_cost": {"warn": 0.2, "cordon": 0.5, "migrate": 0.9, "drain": None},
            }
        )
    )
    fallback = {"warn": 0.3, "cordon": 0.5, "migrate": 0.7, "drain": 0.9}
    _, _, lift = load_precomputed_scores(tmp_path / "scores.parquet", fallback, path)
    _, _, cost = load_precomputed_scores(
        tmp_path / "scores.parquet", fallback, path, tier_key="action_tiers_cost"
    )
    assert lift["warn"] == 0.6 and cost["warn"] == 0.2 and cost["migrate"] == 0.9
    with pytest.raises(KeyError):
        load_precomputed_scores(tmp_path / "scores.parquet", fallback, path, tier_key="nope")
