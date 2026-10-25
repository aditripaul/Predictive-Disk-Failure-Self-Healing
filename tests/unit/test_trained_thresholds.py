"""The live agent and batch scoring must act on the same thresholds.

`make train` tunes a threshold per action tier and records it as
`action_tiers.json`. `pipelines/score_fleet.py` loaded it; the agent's plan
node took the hand-picked cutoffs in `configs/agent.yaml` unconditionally.
The same drive could therefore be cordoned by one path and drained by the
other, and every precision figure in the runbook described only the batch
path.

Both now call `src/models/trained_thresholds.py`. These tests pin the
resolution, the fallbacks that must not raise, and - the point of the change
- that the two paths produce identical thresholds for one run.
"""

from unittest.mock import patch

import pytest
import yaml

from src.models.threshold import UNREACHABLE_THRESHOLD
from src.models.trained_thresholds import (
    action_thresholds_for_run,
    resolve_trained_action_thresholds,
)

FALLBACK = {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}
TRAINED = {"warn": 0.11, "cordon": 0.24, "migrate": 0.37, "drain": 0.52}


def test_a_runs_recorded_thresholds_win_over_the_configured_cutoffs():
    with patch("mlflow.artifacts.load_dict", return_value=TRAINED):
        assert action_thresholds_for_run("run-1", FALLBACK) == TRAINED


def test_a_run_without_the_artifact_falls_back_rather_than_failing():
    """Runs trained before action_tiers.json existed are still scoreable."""
    with patch("mlflow.artifacts.load_dict", side_effect=OSError("no such artifact")):
        assert action_thresholds_for_run("old-run", FALLBACK) == FALLBACK


def test_an_unreachable_tier_becomes_a_threshold_nothing_crosses():
    """A tier whose precision target the model cannot reach is recorded as
    null. It must not become an always-on threshold."""
    with patch("mlflow.artifacts.load_dict", return_value={**TRAINED, "drain": None}):
        resolved = action_thresholds_for_run("run-1", FALLBACK)
    assert resolved["drain"] == UNREACHABLE_THRESHOLD
    assert resolved["drain"] > 1.0
    assert resolved["warn"] == TRAINED["warn"]


def test_the_agent_and_batch_scoring_resolve_the_same_thresholds():
    """The regression this module exists to prevent. Both paths are given the
    same run; if they ever diverge again, this fails."""
    model_config = {
        "primary_horizon_days": 30,
        "mlflow": {"tracking_uri": "sqlite:///nonexistent.db", "experiment_name": "x"},
    }
    agent_config = {"action_thresholds": FALLBACK}

    with (
        patch("mlflow.set_tracking_uri"),
        patch("src.models.trained_thresholds.latest_run_id_for_horizon", return_value="run-7"),
        patch("mlflow.artifacts.load_dict", return_value=TRAINED),
    ):
        agent_thresholds, run_id = resolve_trained_action_thresholds(
            model_config=model_config, agent_config=agent_config
        )
        batch_thresholds = action_thresholds_for_run("run-7", FALLBACK)

    assert run_id == "run-7"
    assert agent_thresholds == batch_thresholds == TRAINED


def test_resolution_never_raises_when_no_model_is_available():
    """A fresh checkout has no MLflow store. The loop must still start, say
    which thresholds it got, and report no run id so the caller can tell."""
    model_config = {
        "primary_horizon_days": 30,
        "mlflow": {"tracking_uri": "sqlite:///nonexistent.db", "experiment_name": "x"},
    }
    with patch(
        "src.models.trained_thresholds.latest_run_id_for_horizon",
        side_effect=RuntimeError("No MLflow runs"),
    ):
        thresholds, run_id = resolve_trained_action_thresholds(
            model_config=model_config, agent_config={"action_thresholds": FALLBACK}
        )
    assert thresholds == FALLBACK
    assert run_id is None


def test_an_explicit_run_id_skips_the_search():
    model_config = {
        "primary_horizon_days": 30,
        "mlflow": {"tracking_uri": "sqlite:///nonexistent.db", "experiment_name": "x"},
    }
    with (
        patch("mlflow.set_tracking_uri"),
        patch("src.models.trained_thresholds.latest_run_id_for_horizon") as search,
        patch("mlflow.artifacts.load_dict", return_value=TRAINED),
    ):
        thresholds, run_id = resolve_trained_action_thresholds(
            model_config=model_config,
            agent_config={"action_thresholds": FALLBACK},
            run_id="pinned",
        )
    search.assert_not_called()
    assert (thresholds, run_id) == (TRAINED, "pinned")


def test_the_shipped_configs_resolve_without_a_trained_model():
    """Against the real configs/*.yaml, not a fixture: a checkout with no
    runs must still produce a usable threshold for every tier.

    Restores the tracking URI afterwards - resolution sets it globally, and
    leaking the project store into a later test that expects a tmp one would
    be a confusing failure far from its cause."""
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    try:
        thresholds, run_id = resolve_trained_action_thresholds()
    finally:
        mlflow.set_tracking_uri(previous_uri)
    agent_config = yaml.safe_load(open("configs/agent.yaml").read())
    assert set(thresholds) >= set(agent_config["action_thresholds"])
    assert all(isinstance(v, float) for v in thresholds.values())
    if run_id is None:
        assert thresholds == agent_config["action_thresholds"]


def test_live_plan_kwargs_passes_thresholds_the_plan_node_accepts():
    """The returned dict is spread into make_plan_node, so a renamed key
    would be silently ignored rather than failing."""
    import inspect

    from src.agent.demo import live_plan_kwargs
    from src.agent.nodes import make_plan_node

    with patch("src.agent.demo.resolve_trained_action_thresholds", return_value=(TRAINED, "run-9")):
        kwargs = live_plan_kwargs()
    assert kwargs == {"action_thresholds": TRAINED}
    accepted = set(inspect.signature(make_plan_node).parameters)
    assert set(kwargs) <= accepted


@pytest.mark.parametrize("tier", ["warn", "cordon", "migrate", "drain"])
def test_every_tier_survives_resolution(tier: str):
    with patch("mlflow.artifacts.load_dict", return_value=TRAINED):
        assert action_thresholds_for_run("run-1", FALLBACK)[tier] == TRAINED[tier]


def test_no_other_module_resolves_action_tiers_on_its_own():
    """The structural guard. The agent and batch scoring drifted apart
    because each loaded `action_tiers.json` itself; keeping that load in one
    module is what stops it recurring, so nothing else may do it."""
    from pathlib import Path

    owner = Path("src/models/trained_thresholds.py")
    offenders = []
    for path in [*Path("src").rglob("*.py"), *Path("pipelines").rglob("*.py")]:
        if path == owner:
            continue
        # Reading the artifact, not merely naming it: `pipelines/train_model.py`
        # legitimately mentions action_tiers.json because it writes it.
        for line in path.read_text().splitlines():
            if "action_tiers" in line and "load_dict" in line:
                offenders.append(f"{path}: {line.strip()}")
    assert offenders == [], f"resolve via {owner} instead of: {offenders}"


def test_guardrail_violations_carry_no_enum_into_checkpointed_state():
    """A GuardrailSeverity instance in AgentState logs "Deserializing
    unregistered type ... will be blocked in a future version" and would
    eventually fail to load. model_dump(mode="json") renders it as a plain
    string, which still compares equal because it is a str Enum."""
    from enum import Enum

    from data_contracts.schemas import GuardrailSeverity
    from src.guardrails.adapter import build_guardrail_evaluator

    evaluate = build_guardrail_evaluator()
    result = evaluate(
        {
            "action_id": "a-1",
            "drive_id": "d-1",
            "proposed_action": "drain",
            "p_fail": 0.99,
            "feature_confidence": 0.1,  # below the floor, so a violation fires
            "feature_maturity": "WARMUP",
            "stale_telemetry": True,
            "is_last_healthy_node_in_domain": True,
            "quorum_ok_after_action": False,
            "high_io_period": False,
            "maintenance_window_active": False,
        }
    )
    assert result["violations"], "fixture should trip at least one guardrail"
    for violation in result["violations"]:
        for key, value in violation.items():
            assert not isinstance(value, Enum), f"{key} is still an enum: {value!r}"
    severities = [v["severity"] for v in result["violations"]]
    assert any(s == GuardrailSeverity.HARD for s in severities), (
        "str Enum comparison must still hold against the plain string"
    )
