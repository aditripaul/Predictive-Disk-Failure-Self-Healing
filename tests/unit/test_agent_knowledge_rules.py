"""Knowledge rules (src/agent/knowledge/rules.py) and the screen's machinery
(scripts/screen_agent_knowledge.py): docs/agent_knowledge_base.md sections 7-8."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data_contracts.schemas import ActionTier
from src.agent.knowledge.rules import (
    RULES,
    apply_tier_rule,
    rule_none,
    rule_r2_chronic_demote,
    rule_r2b_first_flag_promote,
    rule_r3_outcome_memory,
    tier_of,
)

T = {"warn": 0.6, "cordon": 0.8, "migrate": 0.9, "drain": 0.95}
CHRONIC = {
    "open_episode_days": 20,
    "days_above_cordon_window": 15,
    "prior_episodes": 0,
    "days_above_warn_window": 20,
    "history_days": 90,
    "last_resolved_outcome": None,
    "days_since_last_resolved": None,
}
FRESH = {
    **CHRONIC,
    "open_episode_days": 0,
    "days_above_cordon_window": 0,
    "days_above_warn_window": 1,
}


def test_tier_of_follows_the_ladder():
    assert [tier_of(p, T) for p in (0.1, 0.6, 0.85, 0.92, 0.99)] == [
        ActionTier.MONITOR,
        ActionTier.WARN,
        ActionTier.CORDON,
        ActionTier.MIGRATE,
        ActionTier.DRAIN,
    ]


def test_r2_demotes_only_a_long_escalation():
    assert rule_r2_chronic_demote(ActionTier.CORDON, CHRONIC, {"n_days": 14}, T) == ActionTier.WARN
    assert rule_r2_chronic_demote(ActionTier.DRAIN, CHRONIC, {"n_days": 14}, T) == ActionTier.WARN
    assert (
        rule_r2_chronic_demote(ActionTier.CORDON, CHRONIC, {"n_days": 21}, T) == ActionTier.CORDON
    )
    assert rule_r2_chronic_demote(ActionTier.WARN, CHRONIC, {"n_days": 7}, T) == ActionTier.WARN
    assert rule_r2_chronic_demote(ActionTier.CORDON, None, {"n_days": 7}, T) == ActionTier.CORDON


def test_r2b_promotes_a_first_near_cordon_flag_with_history():
    params = {"delta": 0.05, "min_history_days": 60, "p_fail": 0.77}
    assert rule_r2b_first_flag_promote(ActionTier.WARN, FRESH, params, T) == ActionTier.CORDON
    assert (
        rule_r2b_first_flag_promote(ActionTier.WARN, FRESH, {**params, "p_fail": 0.7}, T)
        == ActionTier.WARN
    )
    assert (
        rule_r2b_first_flag_promote(ActionTier.WARN, {**FRESH, "history_days": 30}, params, T)
        == ActionTier.WARN
    )
    assert (
        rule_r2b_first_flag_promote(ActionTier.WARN, {**FRESH, "prior_episodes": 1}, params, T)
        == ActionTier.WARN
    )


def test_r3_lowers_one_band_after_a_recent_survival():
    ctx = {**CHRONIC, "last_resolved_outcome": "survived_horizon", "days_since_last_resolved": 10}
    assert rule_r3_outcome_memory(ActionTier.MIGRATE, ctx, {"m_days": 14}, T) == ActionTier.CORDON
    assert rule_r3_outcome_memory(ActionTier.MIGRATE, ctx, {"m_days": 7}, T) == ActionTier.MIGRATE
    assert rule_r3_outcome_memory(ActionTier.MONITOR, ctx, {"m_days": 14}, T) == ActionTier.MONITOR
    assert (
        rule_r3_outcome_memory(
            ActionTier.WARN, {**ctx, "last_resolved_outcome": "failed"}, {"m_days": 14}, T
        )
        == ActionTier.WARN
    )


@pytest.mark.parametrize("name", sorted(RULES))
def test_the_score_transform_lands_in_the_rules_tier(name):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 300)
    contexts = [CHRONIC if i % 3 == 0 else FRESH if i % 3 == 1 else None for i in range(300)]
    params = {
        "none": {},
        "r2_chronic_demote": {"n_days": 7},
        "r2b_first_flag_promote": {"delta": 0.1, "min_history_days": 60},
        "r3_outcome_memory": {"m_days": 30},
    }[name]
    q = apply_tier_rule(RULES[name], p, contexts, params, T)
    for pi, qi, ctx in zip(p, q, contexts, strict=True):
        expected = (
            tier_of(pi, T)
            if ctx is None
            else RULES[name](tier_of(pi, T), ctx, {**params, "p_fail": pi}, T)
        )
        assert tier_of(qi, T) == expected
        if ctx is None:
            assert qi == pi
    assert rule_none(ActionTier.DRAIN, CHRONIC, {}, T) == ActionTier.DRAIN


# --- the screen's machinery ---------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "screen_agent_knowledge.py"


@pytest.fixture(scope="module")
def screen():
    spec = importlib.util.spec_from_file_location("screen_agent_knowledge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(n_drives=200, days=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drives):
        fails = d % 10 == 0
        for day in range(days):
            p = rng.uniform(0.7, 0.99) if fails else rng.uniform(0.0, 0.85)
            rows.append({"drive_id": f"d{d}", "date": day, "p_fail": p, "label": int(fails)})
    return pl.DataFrame(rows)


def test_identical_flags_have_a_zero_paired_difference(screen):
    obs = _obs()
    codes = screen.dense_drive_codes(obs)
    failed = obs["label"].to_numpy() == 1
    flags = obs["p_fail"].to_numpy() >= 0.8
    diff = screen.paired_difference(codes, failed, flags, flags, n_boot=50, seed=0)
    assert diff["precision_difference"] == 0 and diff["precision_ci"] == [0.0, 0.0]


def test_dropping_false_alarms_raises_precision_with_an_interval_above_zero(screen):
    obs = _obs()
    codes = screen.dense_drive_codes(obs)
    failed = obs["label"].to_numpy() == 1
    flags = obs["p_fail"].to_numpy() >= 0.75
    better = flags & failed  # an oracle that removes every false alarm
    diff = screen.paired_difference(codes, failed, flags, better, n_boot=100, seed=0)
    assert diff["precision_ci"][0] > 0 and diff["recall_difference"] == 0


def test_a_rule_that_keeps_the_warn_set_is_judged_at_cordon_alone(screen):
    obs = _obs()
    p = obs["p_fail"].to_numpy()
    demoted = np.where(p >= T["cordon"], T["cordon"] - 1e-9, p)  # cordon -> warn only
    assert screen.judged_tiers(obs, demoted, T, T) == ("cordon",)
    removed = np.where(p >= T["warn"], 0.0, p)  # drops warn flags too
    assert screen.judged_tiers(obs, removed, T, T) == ("warn", "cordon")


def test_no_rule_wins_when_nothing_helps(screen):
    obs = _obs()
    contexts = [None] * obs.height  # no history: every rule is a no-op
    selection = screen.select_param("r2_chronic_demote", [{"n_days": 7}], obs, contexts, T, {})
    assert selection["chosen"] is None


def test_the_q3_step_refuses_without_a_passing_rule(screen, tmp_path):
    reasons = screen.q3_guard(tmp_path)
    assert any("no validation.json" in r for r in reasons)
    (tmp_path / "validation.json").write_text(json.dumps({"rules": {"r2": {"rule_passed": False}}}))
    assert any("no rule passed" in r for r in screen.q3_guard(tmp_path))
    (tmp_path / "validation.json").write_text(json.dumps({"rules": {"r2": {"rule_passed": True}}}))
    (tmp_path / "q3_evaluation.json").write_text("{}")
    assert any("one Q3 look has been taken" in r for r in screen.q3_guard(tmp_path))
