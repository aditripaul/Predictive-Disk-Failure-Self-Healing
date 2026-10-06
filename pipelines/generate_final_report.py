"""Entry point for the Phase 11 final evaluation report.

Aggregates whatever real evaluation artifacts exist under data/audit/ (model
metrics, guardrail latency, chaos test results) into a single report. Chaos
and guardrail-latency results come from the automated test suite (real,
deterministic); model/cross-dataset metrics require real Backblaze/SMART-Z
data to be ingested first (Phase 1) and are reported as "not yet available"
until then, rather than fabricated.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")

logger = get_logger(__name__)


def _run_pytest(node_ids: list[str]) -> dict:
    result = subprocess.run(
        ["uv", "run", "pytest", *node_ids, "-v", "--tb=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    passed = result.returncode == 0
    return {"passed": passed, "node_ids": node_ids, "stdout_tail": result.stdout[-2000:]}


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _goal_status(model_report: dict | None, model_config: dict) -> dict:
    """The model goal from configs/model.yaml, set against what the trained
    model actually reached (drive level). Says plainly when the goal is not
    met, so the report cannot read as success by omission."""
    threshold_cfg = model_config["threshold"]
    goal = {
        "target_precision": threshold_cfg["target_precision"],
        "target_recall_range": threshold_cfg["target_recall_range"],
    }
    if model_report is None or "threshold" not in model_report:
        return {**goal, "status": "not_yet_available", "reason": "No trained model evaluation."}
    chosen = model_report["threshold"]
    test = model_report.get("test_drive_level", {})
    return {
        **goal,
        "precision_at_recall": model_report.get("precision_at_recall"),
        "data_build": model_report.get("data_build"),
        "target_met": bool(chosen.get("target_met", False)),
        "validation_precision_at_chosen_threshold": chosen.get("precision"),
        "validation_recall_at_chosen_threshold": chosen.get("recall"),
        "test_drive_precision": test.get("precision"),
        "test_drive_recall": test.get("recall"),
        "test_fleet_failure_rate": test.get("fleet_failure_rate"),
        "test_lift": test.get("lift"),
        # The same alerts restated for test sets with more failing drives, as
        # most published results use. Not a separate measurement; the target
        # above is judged on the fleet figure.
        "test_drive_precision_at_failure_rate": test.get("precision_at_failure_rate"),
        "precision_gap_to_target": chosen.get("precision_gap_to_target"),
        "summary": (
            "Goal met on validation."
            if chosen.get("target_met")
            else "Goal not met: the target precision is not reachable at the required recall "
            "with this data. Action tiers (model_report.action_tiers) give the reachable "
            "operating points."
        ),
    }


def _data_period(data_config: dict, model_config: dict) -> dict:
    return {
        "ingested_from": data_config.get("sources", {}).get("backblaze", {}).get("start_date"),
        "ingested_to": data_config.get("sources", {}).get("backblaze", {}).get("end_date"),
        "splits": model_config.get("splits"),
        "primary_horizon_days": model_config.get("primary_horizon_days"),
    }


#: The figures `make plots` renders (pipelines/generate_performance_plots.py),
#: with why each may be absent. The SHAP plot needs diagnostics.shap_enabled.
PLOT_FILES = {
    "calibration_validation.png": "calibration of validation predictions",
    "calibration_test.png": "calibration of test predictions",
    "metric_comparison.png": "validation vs test metrics",
    "precision_at_k_test.png": "precision at top-K on test",
    "class_imbalance.png": "label imbalance by horizon and split",
    "shap_feature_importance.png": "SHAP feature importance (needs diagnostics.shap_enabled)",
    "roc_curves.png": "ROC curves: LightGBM vs logistic baseline, row and drive level (test)",
    "variable_importance.png": "variable importance by share of gain (LightGBM)",
    "failures_by_family.png": "failed drives per drive family",
}


def _plots_section(audit_dir: Path) -> dict:
    plots_dir = audit_dir / "plots"
    return {
        name: {
            "description": description,
            "path": str(plots_dir / name),
            "present": (plots_dir / name).exists(),
        }
        for name, description in PLOT_FILES.items()
    }


def build_report() -> dict:
    audit_dir = Path("data/audit/data_quality_reports")

    model_report = _load_json_if_exists(audit_dir / "model_evaluation_report.json")
    label_report = _load_json_if_exists(audit_dir / "label_imbalance_report.json")
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())

    chaos_results = _run_pytest(["tests/chaos/test_chaos_scenarios.py"])
    latency_results = _run_pytest(["tests/chaos/test_guardrail_latency.py"])
    replay_results = _run_pytest(
        [
            "tests/integration/test_agent_graph.py::test_crash_recovery_replay_does_not_duplicate_execution"
        ]
    )

    report = {
        "plots": _plots_section(Path("data/audit")),
        "data_period": _data_period(data_config, model_config),
        "goal_status": _goal_status(model_report, model_config),
        "prediction_precision_recall": model_report
        or {
            "status": "not_yet_available",
            "reason": (
                "No trained model evaluation found. Requires real Backblaze data "
                "ingested (Phase 1) and `make train` run against it."
            ),
        },
        "label_imbalance": label_report
        or {"status": "not_yet_available", "reason": "Requires real ingested data."},
        "cross_dataset_generalization": {
            "status": "not_yet_available",
            "reason": (
                "Requires both Backblaze and SMART-Z raw data ingested and a model "
                "trained on Backblaze, evaluated against SMART-Z (docs/dataset_"
                "strategy.md section 3.2)."
            ),
        },
        "chaos_test_results": chaos_results,
        "guardrail_latency_results": latency_results,
        "crash_recovery_replay_results": replay_results,
        "guardrail_compliance": {
            "hard_guardrail_violations_executed": 0,
            "note": (
                "By construction: the agent's Plan node only executes when "
                "guardrail_result.passed is True (src/agent/nodes.py); every hard "
                "violation routes to human_review instead. See "
                "tests/unit/test_guardrails.py and tests/chaos/test_chaos_scenarios.py."
            ),
        },
    }
    return report


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    report = build_report()
    out_dir = Path("data/audit/data_quality_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("final_evaluation_report_written", path=str(out_path))

    for key in ("chaos_test_results", "guardrail_latency_results", "crash_recovery_replay_results"):
        logger.info("report_section", section=key, passed=report[key]["passed"])
    for key in ("prediction_precision_recall", "label_imbalance", "cross_dataset_generalization"):
        status = report[key].get("status", "available")
        logger.info("report_section", section=key, status=status)


if __name__ == "__main__":
    main()
