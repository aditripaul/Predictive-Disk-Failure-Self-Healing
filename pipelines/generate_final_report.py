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

from src.logging_config import configure_logging, get_logger

DATA_CONFIG_PATH = Path("configs/data.yaml")

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


def build_report() -> dict:
    audit_dir = Path("data/audit/data_quality_reports")

    model_report = _load_json_if_exists(audit_dir / "model_evaluation_report.json")
    label_report = _load_json_if_exists(audit_dir / "label_imbalance_report.json")

    chaos_results = _run_pytest(["tests/chaos/test_chaos_scenarios.py"])
    latency_results = _run_pytest(["tests/chaos/test_guardrail_latency.py"])
    replay_results = _run_pytest(
        ["tests/integration/test_agent_graph.py::test_crash_recovery_replay_does_not_duplicate_execution"]
    )

    report = {
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
