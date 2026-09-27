"""Model card generation (docs/project_plan.md Phase 6 Key Task 7 —
"Register model... Store: threshold policy; feature schema; dataset
version; model card; explainability artifacts" — and Deliverables/Exit
Criteria "Model card").

A model card is a structured, human-readable summary of what a trained
model is, how it performs, and where it should not be trusted. MLflow's
own params/metrics logging answers "what run had what number" for a
developer who already knows what to query; the model card is written for
an auditor or on-call operator who does not, and pairs directly with the
explainability artifacts (`src/models/explainability.py`) it links to.
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def build_model_card(
    *,
    horizon_days: int,
    model_params: dict[str, Any],
    model_version: int,
    model_type: str = "lightgbm",
    feature_registry_version: int,
    dataset_version: str | None,
    feature_columns: list[str],
    threshold_result: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    shap_top_features: list[dict[str, Any]],
    train_row_count: int,
    test_warning_lead_time: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "model_details": {
            "type": model_type,
            "version": model_version,
            "params": model_params,
            "primary_horizon_days": horizon_days,
        },
        "intended_use": {
            "purpose": (
                "Predict the probability of disk failure within the configured "
                "horizon, driving a MAPE-K self-healing agent's action-tier "
                "decisions (monitor/warn/cordon/migrate/drain)."
            ),
            "out_of_scope": (
                "Not validated for drive models/vendors outside the training "
                "distribution until the external SMART-Z validation split "
                "(split=external_smartz) shows acceptable cross-vendor "
                "generalization; not a substitute for guardrail or human "
                "review on destructive actions."
            ),
        },
        "training_data": {
            "dataset_version": dataset_version or "unversioned",
            "feature_registry_version": feature_registry_version,
            "feature_count": len(feature_columns),
            "training_row_count": train_row_count,
        },
        "evaluation": {
            "threshold_policy": threshold_result,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_warning_lead_time": test_warning_lead_time,
        },
        "explainability": {
            "method": "shap.TreeExplainer",
            "top_features": shap_top_features,
        },
        "limitations_and_risks": [
            "Trained primarily on Backblaze data; SMART-Z is an external "
            "validation split, never part of training, so cross-vendor "
            "performance is only as good as that split's evaluation shows.",
            "Destructive actions (cordon/migrate/drain) additionally require "
            "feature_confidence, feature_maturity == MATURE, non-stale "
            "telemetry, and guardrail checks to pass - this model's p_fail "
            "alone never gates execution.",
            "The precision/recall targets (>=95% precision, 35-50% recall) "
            "are tuned on the validation split; the test-split metrics above "
            "are the more honest read of generalization.",
        ],
    }


def render_model_card_markdown(card: dict[str, Any]) -> str:
    details = card["model_details"]
    training = card["training_data"]
    evaluation = card["evaluation"]
    lines = [
        "# Model Card",
        "",
        f"Generated: {card['generated_at']}",
        "",
        "## Model Details",
        f"- Type: {details['type']}",
        f"- Version: {details['version']}",
        f"- Primary horizon: {details['primary_horizon_days']} days",
        f"- Params: {details['params']}",
        "",
        "## Intended Use",
        f"- Purpose: {card['intended_use']['purpose']}",
        f"- Out of scope: {card['intended_use']['out_of_scope']}",
        "",
        "## Training Data",
        f"- Dataset version: {training['dataset_version']}",
        f"- Feature registry version: {training['feature_registry_version']}",
        f"- Feature count: {training['feature_count']}",
        f"- Training rows: {training['training_row_count']}",
        "",
        "## Evaluation",
        f"- Threshold policy: {evaluation['threshold_policy']}",
        f"- Validation metrics: {evaluation['validation_metrics']}",
        f"- Test metrics: {evaluation['test_metrics']}",
        f"- Test warning lead time: {evaluation['test_warning_lead_time']}",
        "",
        "## Explainability",
        f"- Method: {card['explainability']['method']}",
        "- Top features:",
    ]
    for entry in card["explainability"]["top_features"]:
        lines.append(f"  - {entry}")
    lines += ["", "## Limitations and Risks"]
    lines += [f"- {risk}" for risk in card["limitations_and_risks"]]
    lines.append("")
    return "\n".join(lines)
