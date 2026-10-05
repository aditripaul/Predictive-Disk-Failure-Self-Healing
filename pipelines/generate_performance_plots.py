"""Entry point for `make plots`.

Renders performance-metric plots (calibration reliability diagrams, SHAP
global feature importance, class imbalance, validation-vs-test metric
comparison, precision-at-top-K) from the JSON reports `make train` /
`make build-labels` already write, to `data/audit/plots/`. Each plot is
skipped with a clear warning (not an error) if its source report doesn't
exist yet, so this can be run at any point in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from src.logging_config import configure_logging, get_logger
from src.reporting.figure_data import failure_counts_by_family
from src.reporting.plots import (
    plot_calibration_curve,
    plot_class_imbalance,
    plot_failures_by_family,
    plot_feature_importance,
    plot_metric_comparison,
    plot_precision_at_k,
    plot_roc_curves,
    plot_shap_feature_importance,
    save_figure,
)
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    reports_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    plots_dir = Path(data_config["audit_dir"]) / "plots"

    written: list[Path] = []

    model_report_path = reports_dir / "model_evaluation_report.json"
    if model_report_path.exists():
        report = json.loads(model_report_path.read_text())
        validation_metrics = report["validation_metrics"]
        test_metrics = report["test_metrics"]

        written.append(
            save_figure(
                plot_calibration_curve(
                    validation_metrics["calibration"], title="Validation calibration"
                ),
                plots_dir / "calibration_validation.png",
            )
        )
        written.append(
            save_figure(
                plot_calibration_curve(test_metrics["calibration"], title="Test calibration"),
                plots_dir / "calibration_test.png",
            )
        )
        written.append(
            save_figure(
                plot_metric_comparison(validation_metrics, test_metrics),
                plots_dir / "metric_comparison.png",
            )
        )
        written.append(
            save_figure(
                plot_precision_at_k(test_metrics, title="test"),
                plots_dir / "precision_at_k_test.png",
            )
        )

        shap_report_path = reports_dir / "shap_feature_importance.json"
        if shap_report_path.exists():
            shap_importance = json.loads(shap_report_path.read_text())
            written.append(
                save_figure(
                    plot_shap_feature_importance(shap_importance),
                    plots_dir / "shap_feature_importance.png",
                )
            )
        else:
            logger.warning("shap_feature_importance_report_missing", path=str(shap_report_path))
    else:
        logger.warning(
            "model_evaluation_report_missing",
            path=str(model_report_path),
            hint="run `make train` first",
        )

    # Figures from the paper that apply to this pipeline (see
    # src/reporting/figure_data.py). Each is skipped with a warning if its
    # source is missing, as above.
    if model_report_path.exists():
        model_report = json.loads(model_report_path.read_text())
        if "roc_curves" in model_report:
            curve_labels = {
                "lightgbm_test_row": "LightGBM, row level",
                "logistic_test_row": "Logistic baseline, row level",
                "lightgbm_test_drive": "LightGBM, drive level",
                "logistic_test_drive": "Logistic baseline, drive level",
            }
            curves = {
                label: model_report["roc_curves"][key]
                for key, label in curve_labels.items()
                if key in model_report["roc_curves"]
            }
            written.append(
                save_figure(
                    plot_roc_curves(curves, title="ROC curves on the test split"),
                    plots_dir / "roc_curves.png",
                )
            )
        if "feature_importance" in model_report:
            written.append(
                save_figure(
                    plot_feature_importance(
                        model_report["feature_importance"],
                        title="Variable importance (share of total gain)",
                    ),
                    plots_dir / "variable_importance.png",
                )
            )

    metadata_path = Path(data_config["silver_dir"]) / "drive_metadata" / "part.parquet"
    if metadata_path.exists():
        metadata = pl.read_parquet(metadata_path, columns=["model_family", "failure_date"])
        written.append(
            save_figure(
                plot_failures_by_family(failure_counts_by_family(metadata)),
                plots_dir / "failures_by_family.png",
            )
        )
    else:
        logger.warning(
            "drive_metadata_missing", path=str(metadata_path), hint="run `make build-silver` first"
        )

    imbalance_path = reports_dir / "label_imbalance_report.json"
    if imbalance_path.exists():
        imbalance_report = json.loads(imbalance_path.read_text())
        written.append(
            save_figure(
                plot_class_imbalance(imbalance_report["class_distribution"]),
                plots_dir / "class_imbalance.png",
            )
        )
    else:
        logger.warning(
            "label_imbalance_report_missing",
            path=str(imbalance_path),
            hint="run `make build-labels` first",
        )

    logger.info("performance_plots_written", count=len(written), plots_dir=str(plots_dir))


if __name__ == "__main__":
    main()
