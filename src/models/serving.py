"""Batch fleet scoring - the first real consumer of the typed
`PredictionOutput`/`ActionProposal` data contracts
(`data_contracts/schemas.py`), previously declared but never imported
anywhere in real code.

`pipelines/score_fleet.py` runs the trained model over each drive's most
recent gold-feature row and produces one validated `PredictionOutput`
(plus an `ActionProposal` for any drive the action-tier policy flags
above MONITOR) - a periodic, offline fleet risk report, independent of
the live MAPE-K agent loop. The loop's `AgentState` is deliberately lean
(docs/design_goal.md section 5.6, `src/agent/state.py`) and cannot carry
`PredictionOutput`'s full nested `FeatureConfidence` without abandoning
that design decision; this module is where the richer contract genuinely
applies, because a batch scoring run has the full gold-feature row on
hand rather than a lean per-cycle state dict.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import numpy as np
import polars as pl

from data_contracts.schemas import (
    ActionProposal,
    ActionTier,
    FeatureConfidence,
    FeatureMaturity,
    PredictionOutput,
)
from src.models.action_tiers import determine_action_tier
from src.models.training import predict_proba_positive

#: Columns every row of `latest_features` must carry beyond the model's
#: own feature columns - all produced by the existing gold-feature
#: pipeline (src/preprocess/telemetry_gaps.py, src/features/confidence.py,
#: src/preprocess/feature_maturity.py joined in).
REQUIRED_CONFIDENCE_COLUMNS = (
    "telemetry_coverage_30d",
    "hours_since_last_telemetry",
    "attribute_coverage_factor",
    "feature_confidence",
    "feature_maturity",
)


def score_latest_drive_day(
    latest_features: pl.DataFrame,
    model: Any,
    feature_columns: list[str],
    *,
    model_name: str,
    model_version: str,
    horizon_days: int,
    as_of: dt.datetime,
) -> list[PredictionOutput]:
    """`latest_features` is one row per drive (its most recent gold-feature
    row - see `pipelines/score_fleet.py::_latest_row_per_drive`) and must
    carry `REQUIRED_CONFIDENCE_COLUMNS` plus every column in
    `feature_columns`. Returns one validated `PredictionOutput` per row."""
    missing = [c for c in REQUIRED_CONFIDENCE_COLUMNS if c not in latest_features.columns]
    if missing:
        raise ValueError(f"latest_features is missing required columns: {missing}")

    x = latest_features.select(feature_columns).fill_null(0.0).to_numpy()
    p_fail_scores = np.asarray(predict_proba_positive(model, x))

    predictions = []
    for row, p_fail in zip(latest_features.to_dicts(), p_fail_scores, strict=True):
        confidence = FeatureConfidence(
            drive_id=row["drive_id"],
            as_of=as_of,
            telemetry_coverage_30d=row["telemetry_coverage_30d"],
            hours_since_last_telemetry=row["hours_since_last_telemetry"],
            attribute_coverage_factor=row["attribute_coverage_factor"],
            feature_confidence=row["feature_confidence"],
            feature_maturity=FeatureMaturity(row["feature_maturity"]),
        )
        predictions.append(
            PredictionOutput(
                drive_id=row["drive_id"],
                as_of=as_of,
                model_name=model_name,
                model_version=model_version,
                horizon_days=horizon_days,
                p_fail=float(p_fail),
                feature_confidence=confidence,
            )
        )
    return predictions


def propose_actions(
    predictions: list[PredictionOutput],
    *,
    action_thresholds: dict[str, float],
    min_confidence_for_destructive_action: float,
    stale_telemetry_by_drive: dict[str, bool] | None = None,
) -> list[ActionProposal]:
    """Runs the same action-tier policy the live agent uses
    (`src/models/action_tiers.py::determine_action_tier`) over each
    prediction, producing an `ActionProposal` for anything above MONITOR
    (a MONITOR "proposal" isn't actionable, so it's omitted)."""
    stale_telemetry_by_drive = stale_telemetry_by_drive or {}
    proposals = []
    for prediction in predictions:
        confidence = prediction.feature_confidence
        tier = determine_action_tier(
            p_fail=prediction.p_fail,
            feature_confidence=confidence.feature_confidence,
            feature_maturity=confidence.feature_maturity,
            stale_telemetry=stale_telemetry_by_drive.get(prediction.drive_id, False),
            action_thresholds=action_thresholds,
            min_confidence_for_destructive_action=min_confidence_for_destructive_action,
        )
        if tier == ActionTier.MONITOR:
            continue
        proposals.append(
            ActionProposal(
                action_id=str(uuid.uuid4()),
                drive_id=prediction.drive_id,
                proposed_action=tier,
                prediction=prediction,
                rationale=(
                    f"p_fail={prediction.p_fail:.3f} triggers {tier.value} "
                    f"(feature_confidence={confidence.feature_confidence:.2f}, "
                    f"feature_maturity={confidence.feature_maturity.value})"
                ),
                created_at=prediction.as_of,
            )
        )
    return proposals
