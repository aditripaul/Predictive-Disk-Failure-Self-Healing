"""Input context for a single guardrail evaluation
(docs/design_goal.md section 14).
"""

from __future__ import annotations

from dataclasses import dataclass

from data_contracts.schemas import ActionTier, FeatureMaturity

DESTRUCTIVE_TIERS = {ActionTier.MIGRATE, ActionTier.DRAIN}


@dataclass
class RuleContext:
    action_id: str
    drive_id: str
    proposed_action: ActionTier
    p_fail: float
    feature_confidence: float
    feature_maturity: FeatureMaturity
    stale_telemetry: bool

    # Fleet/operational state (Phase 8's simulator supplies real values;
    # sensible conservative defaults are used until then).
    is_last_healthy_node_in_domain: bool = False
    quorum_ok_after_action: bool = True
    concurrent_drains: int = 0
    high_io_period: bool = False
    maintenance_window_active: bool = False
    actions_in_last_hour: int = 0

    # Policy thresholds (configs/guardrails.yaml, configs/model.yaml).
    prediction_threshold: float = 0.80
    min_feature_confidence: float = 0.80
    max_concurrent_drains: int = 1
    max_actions_per_hour: int = 5

    def is_destructive(self) -> bool:
        return self.proposed_action in DESTRUCTIVE_TIERS
