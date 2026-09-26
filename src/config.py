"""Loads the project's YAML configs into typed, ready-to-use settings
objects, so runtime code (the agent demo, the orchestrator, the guardrail
adapter) reads the same configs/*.yaml the docs describe instead of
duplicating their values as hardcoded Python literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    return yaml.safe_load(path.read_text())


@dataclass
class AgentSettings:
    checkpoint_path: str
    action_ledger_path: str
    target_cycle_time_seconds: float
    cooldown_seconds: int
    hysteresis_cycles_required: int
    critical_sla_hours: float
    non_critical_sla_hours: float
    timeout_fallback_action: str
    action_thresholds: dict[str, float]
    min_confidence_for_destructive_action: float

    @classmethod
    def load(cls) -> AgentSettings:
        raw = load_yaml("agent.yaml")
        return cls(
            checkpoint_path=raw["checkpoint"]["path"],
            action_ledger_path=raw["checkpoint"]["action_ledger_path"],
            target_cycle_time_seconds=raw["loop"]["target_cycle_time_seconds"],
            cooldown_seconds=raw["loop"]["cooldown_seconds"],
            hysteresis_cycles_required=raw["loop"]["hysteresis_cycles_required"],
            critical_sla_hours=raw["human_review"]["critical_sla_hours"],
            non_critical_sla_hours=raw["human_review"]["non_critical_sla_hours"],
            timeout_fallback_action=raw["human_review"]["timeout_fallback_action"],
            action_thresholds=raw["action_thresholds"],
            min_confidence_for_destructive_action=raw["feature_confidence"][
                "min_confidence_for_destructive_action"
            ],
        )


@dataclass
class ModelSettings:
    primary_horizon_days: int

    @classmethod
    def load(cls) -> ModelSettings:
        raw = load_yaml("model.yaml")
        return cls(primary_horizon_days=raw["primary_horizon_days"])


@dataclass
class GuardrailSettings:
    prediction_threshold: float
    min_feature_confidence: float
    max_concurrent_drains: int
    max_actions_per_hour: int

    @classmethod
    def load(cls) -> GuardrailSettings:
        raw = load_yaml("guardrails.yaml")
        rules_by_id = {r["id"]: r for r in raw["rules"]}
        agent = AgentSettings.load()
        return cls(
            # "migrate" is the lowest p_fail cutoff at which any destructive
            # tier (migrate/drain) can be proposed; PRED_THRESHOLD must not
            # be set any tighter than that or it would block every migrate
            # action action_tiers.py legitimately proposes.
            prediction_threshold=agent.action_thresholds["migrate"],
            min_feature_confidence=agent.min_confidence_for_destructive_action,
            max_concurrent_drains=rules_by_id["HARD_MAX_DRAINS"]["params"][
                "max_concurrent_drains"
            ],
            max_actions_per_hour=rules_by_id["OPS_RATE_LIMIT"]["params"]["max_actions_per_hour"],
        )
