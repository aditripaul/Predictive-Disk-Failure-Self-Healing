"""Typed data contracts shared across the ingestion, feature, agent, guardrail,
and reliability layers.

These models are the single source of truth for the shape of data crossing
module boundaries. Bulk telemetry/feature tables live in Parquet/DuckDB and are
referenced here by id, not embedded.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceDataset(str, Enum):
    BACKBLAZE = "backblaze"
    SMARTZ = "smartz"
    SYNTHETIC = "synthetic"


class DriveType(str, Enum):
    HDD = "HDD"
    SSD = "SSD"
    UNKNOWN = "UNKNOWN"


class FeatureMaturity(str, Enum):
    WARMUP = "WARMUP"
    MATURE = "MATURE"
    STALE = "STALE"
    DECOMMISSIONED = "DECOMMISSIONED"


class CensoringFlag(str, Enum):
    FAILURE = "failure"
    REPLACEMENT = "replacement"
    REMOVED = "removed"
    CENSORED = "censored"


class EventType(str, Enum):
    CONFIRMED_FAILURE = "confirmed_failure"
    FAILURE_FOLLOWED_BY_REPLACEMENT = "failure_followed_by_replacement"
    PREVENTIVE_REPLACEMENT = "preventive_replacement"
    REMOVED_WITHOUT_FAILURE = "removed_without_failure"
    CENSORED = "censored"


class ActionTier(str, Enum):
    MONITOR = "monitor"
    WARN = "warn"
    CORDON = "cordon"
    MIGRATE = "migrate"
    DRAIN = "drain"
    HUMAN_REVIEW = "human_review"


class GuardrailSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    NONE = "none"


class HumanDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    NOT_REQUIRED = "not_required"


class CanonicalTelemetryRecord(BaseModel):
    """One row of the silver-layer canonical SMART telemetry table."""

    drive_id: str
    date: date
    source_dataset: SourceDataset
    drive_model: str
    model_family: str
    manufacturer: str | None = None
    capacity_gb: float
    drive_type: DriveType = DriveType.UNKNOWN
    smart_attribute_name: str
    smart_raw_value: float
    smart_normalized_value: float | None = None
    smart_badness_value: float
    failure_date: date | None = None
    removal_date: date | None = None
    days_since_last_telemetry: int = 0
    telemetry_gap_flag: bool = False
    stale_telemetry_flag: bool = False
    censoring_flag: CensoringFlag | None = None


class DriveMetadata(BaseModel):
    """One row of the silver-layer drive metadata table."""

    drive_id: str
    first_seen_date: date
    last_seen_date: date
    failure_date: date | None = None
    replacement_date: date | None = None
    model_family: str
    capacity_gb: float
    drive_type: DriveType = DriveType.UNKNOWN
    total_observed_days: int
    active_history_days: int
    survivorship_valid: bool
    feature_maturity: FeatureMaturity


class FeatureConfidence(BaseModel):
    """Confidence that a drive's current feature vector is fresh/complete
    enough to justify a destructive autonomous action."""

    drive_id: str
    as_of: datetime
    telemetry_coverage_30d: float = Field(ge=0.0, le=1.0)
    hours_since_last_telemetry: float
    attribute_coverage_factor: float = Field(ge=0.0, le=1.0)
    feature_confidence: float = Field(ge=0.0, le=1.0)
    feature_maturity: FeatureMaturity


class FeatureRecord(BaseModel):
    """Reference to a materialized gold-layer feature row; the bulk feature
    vector lives in Parquet and is looked up by (drive_id, date, feature_version)."""

    drive_id: str
    date: date
    feature_version: int
    feature_table_path: str


class PredictionOutput(BaseModel):
    drive_id: str
    as_of: datetime
    model_name: str
    model_version: str
    horizon_days: int
    p_fail: float = Field(ge=0.0, le=1.0)
    feature_confidence: FeatureConfidence
    top_contributing_features: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    action_id: str
    drive_id: str
    node_id: str | None = None
    proposed_action: ActionTier
    prediction: PredictionOutput
    rationale: str
    created_at: datetime


class GuardrailViolation(BaseModel):
    rule_id: str
    severity: GuardrailSeverity
    message: str


class GuardrailResult(BaseModel):
    action_id: str
    evaluated_at: datetime
    passed: bool
    violations: list[GuardrailViolation] = Field(default_factory=list)
    final_action: ActionTier
    evaluation_latency_ms: float


class ExecutionResult(BaseModel):
    action_id: str
    executed_at: datetime
    success: bool
    compensating_action_triggered: bool = False
    error: str | None = None


class ValidationResult(BaseModel):
    action_id: str
    validated_at: datetime
    data_integrity_ok: bool
    service_continuity_ok: bool
    quorum_ok: bool


class TrustScoreRecord(BaseModel):
    decision_id: str
    action_id: str
    correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    timeliness: float | None = Field(default=None, ge=0.0, le=1.0)
    necessity: float | None = Field(default=None, ge=0.0, le=1.0)
    safety_violation: bool = False
    guardrail_severity: GuardrailSeverity = GuardrailSeverity.NONE
    trust_score_provisional: float = Field(ge=0.0, le=1.0)
    trust_score_final: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionAuditRecord(BaseModel):
    decision_id: str
    run_id: str
    timestamp: datetime
    drive_id: str
    fleet_snapshot_id: str
    prediction: PredictionOutput
    proposed_action: ActionTier
    final_action: ActionTier
    guardrail_result: GuardrailResult
    human_review_required: bool
    human_decision: HumanDecision = HumanDecision.NOT_REQUIRED
    override_reason_code: str | None = None
    execution_result: ExecutionResult | None = None
    validation_result: ValidationResult | None = None
    trust_score: TrustScoreRecord | None = None
    checkpoint_id: str | None = None
    feature_explanations: list[str] = Field(default_factory=list)
