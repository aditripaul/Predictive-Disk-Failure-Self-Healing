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
    shap_top_features: list[dict[str, Any]] | None,
    train_row_count: int,
    test_warning_lead_time: dict[str, Any] | None = None,
    logistic_regression_baseline: dict[str, Any] | None = None,
    smote_comparison: dict[str, Any] | None = None,
    drive_level_metrics: dict[str, Any] | None = None,
    action_tiers: dict[str, Any] | None = None,
    precision_at_recall: list[dict[str, Any]] | None = None,
    two_stage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "model_details": {
            "type": model_type,
            "version": model_version,
            "params": model_params,
            "primary_horizon_days": horizon_days,
            # Second-stage report (src/models/two_stage.py); None = single stage.
            "two_stage": two_stage,
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
            "logistic_regression_baseline": logistic_regression_baseline,
            "smote_comparison": smote_comparison,
            # Per-drive precision/recall/AUPRC (src/models/evaluation.py::
            # drive_level_metrics) - the level the model goal is stated at.
            "drive_level_metrics": drive_level_metrics,
            # Per agent action: score threshold chosen on validation for a
            # drive-level precision target, with validation/test precision
            # and recall (src/models/threshold.py::tune_action_tiers).
            "action_tiers": action_tiers,
            # Drive-level precision at fixed recall levels; thresholds chosen
            # on validation, applied to test (evaluation.precision_at_recall_table).
            "precision_at_recall": precision_at_recall,
        },
        # `shap_top_features=None` means SHAP was skipped for this run
        # (configs/model.yaml diagnostics.shap_enabled: false).
        "explainability": {
            "method": "shap.TreeExplainer" if shap_top_features is not None else None,
            "top_features": shap_top_features or [],
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


def _reference_rate_text(metrics: dict[str, Any]) -> str:
    """'; 98.9% at 15% failing, 99.8% at 50% failing' - the same alerts on
    populations with more failures (evaluation.precision_at_failure_rate)."""
    by_rate = metrics.get("precision_at_failure_rate")
    if not by_rate:
        return ""
    return "; " + ", ".join(
        f"{value:.1%} at {float(rate):.0%} failing" for rate, value in by_rate.items()
    )


def _two_stage_lines(two_stage: dict[str, Any] | None) -> list[str]:
    if not two_stage:
        return ["- Stages: one (no second-stage model)"]
    if two_stage.get("status") != "ok":
        return [f"- Stages: one (second stage not trained: {two_stage.get('status')})"]
    candidates = two_stage["candidates"]
    lines = [
        f"- Stages: two. Rows scoring >= {two_stage['candidate_threshold']:.4f} on the first "
        f"model (the validation threshold catching {two_stage['candidate_recall']:.0%} of "
        f"failing drives) are re-ranked by a second model ({two_stage['stage2_trees']} trees) "
        f"trained on {candidates['train_rows']} candidate train rows "
        f"({candidates['train_positive_rows']} positive). Every figure below is for the "
        "combined score."
    ]
    for row in two_stage.get("stage1_precision_at_recall") or []:
        if row.get("threshold") is None:
            continue
        test = row["test"]
        lines.append(
            f"  - first model alone, recall >= {row['target_recall']:.0%}: test precision "
            f"{test['precision']:.1%} at recall {test['recall']:.1%} "
            f"({test['caught_drive_count']} caught, {test['false_alarm_drive_count']} false alarms)"
        )
    return lines


def _precision_at_recall_lines(rows: list[dict[str, Any]] | None) -> list[str]:
    if not rows:
        return []
    lines = [
        "- Drive-level precision at recall (threshold from validation, measured on test). "
        "The first precision is at the fleet's own failure rate; the figures after the "
        "semicolon restate the same alerts for test sets with more failures, as most "
        "published results use:"
    ]
    for row in rows:
        if row.get("threshold") is None:
            lines.append(f"  - recall >= {row['target_recall']:.0%}: not reachable on validation")
            continue
        test = row["test"]
        lines.append(
            f"  - recall >= {row['target_recall']:.0%}: test precision {test['precision']:.1%} "
            f"at recall {test['recall']:.1%} ({test['caught_drive_count']}/"
            f"{test['failing_drive_count']} failing drives, "
            f"{test['false_alarm_drive_count']} false alarms{_reference_rate_text(test)})"
        )
    return lines


def _action_tier_lines(action_tiers: dict[str, Any] | None) -> list[str]:
    if not action_tiers:
        return []
    lines = [
        "- Action tiers (drive level; threshold chosen on validation for the precision target):"
    ]
    for tier, info in action_tiers.items():
        if info is None:
            lines.append(f"  - {tier}: precision target unreachable on validation - never fires")
            continue
        val, test = info["validation"], info["test"]
        lines.append(
            f"  - {tier}: score >= {info['threshold']:.4f} (target precision "
            f"{info['target_precision']:.0%}) -> validation precision {val['precision']:.1%} "
            f"recall {val['recall']:.1%}; test precision {test['precision']:.1%} "
            f"recall {test['recall']:.1%} ({test['caught_drive_count']}/"
            f"{test['failing_drive_count']} failing drives, "
            f"{test['false_alarm_drive_count']} false alarms)"
        )
    return lines


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
        *_two_stage_lines(details.get("two_stage")),
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
        f"- Drive-level metrics: {evaluation.get('drive_level_metrics')}",
        *_precision_at_recall_lines(evaluation.get("precision_at_recall")),
        *_action_tier_lines(evaluation.get("action_tiers")),
        f"- Test warning lead time: {evaluation['test_warning_lead_time']}",
        f"- Logistic Regression sanity baseline: {evaluation['logistic_regression_baseline']}",
        f"- SMOTE comparison: {evaluation['smote_comparison']}",
        "",
        "## Explainability",
    ]
    if card["explainability"]["method"] is None:
        lines.append(
            "- SHAP skipped for this run (configs/model.yaml diagnostics.shap_enabled: false)"
        )
    else:
        lines += [f"- Method: {card['explainability']['method']}", "- Top features:"]
        for entry in card["explainability"]["top_features"]:
            lines.append(f"  - {entry}")
    lines += ["", "## Limitations and Risks"]
    lines += [f"- {risk}" for risk in card["limitations_and_risks"]]
    lines.append("")
    return "\n".join(lines)
