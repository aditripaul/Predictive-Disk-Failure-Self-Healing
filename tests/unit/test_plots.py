import matplotlib.pyplot as plt

from src.reporting.plots import (
    plot_calibration_curve,
    plot_class_imbalance,
    plot_metric_comparison,
    plot_precision_at_k,
    plot_shap_feature_importance,
    save_figure,
)

CALIBRATION = {
    "expected_calibration_error": 0.05,
    "brier_score": 0.02,
    "bins": [
        {
            "bin_lower": 0.0,
            "bin_upper": 0.5,
            "count": 10,
            "mean_predicted_probability": 0.1,
            "observed_fraction_positive": 0.05,
        },
        {
            "bin_lower": 0.5,
            "bin_upper": 1.0,
            "count": 0,
            "mean_predicted_probability": None,
            "observed_fraction_positive": None,
        },
        {
            "bin_lower": 0.9,
            "bin_upper": 1.0,
            "count": 5,
            "mean_predicted_probability": 0.95,
            "observed_fraction_positive": 1.0,
        },
    ],
}

METRICS = {
    "auprc": 0.9,
    "precision": 0.95,
    "recall": 0.4,
    "false_positive_rate": 0.01,
    "false_negative_rate": 0.6,
    "precision_at_top_1pct": 1.0,
    "precision_at_top_5pct": 0.6,
    "precision_at_top_10pct": 0.3,
}


def test_plot_calibration_curve_skips_empty_bins():
    fig = plot_calibration_curve(CALIBRATION, title="Test")
    ax = fig.axes[0]
    # one line for "perfectly calibrated", one for the model
    assert len(ax.lines) == 2
    model_line = ax.lines[1]
    assert list(model_line.get_xdata()) == [0.1, 0.95]
    plt.close(fig)


def test_plot_calibration_curve_title_includes_ece_and_brier():
    fig = plot_calibration_curve(CALIBRATION, title="My Title")
    assert "My Title" in fig.axes[0].get_title()
    assert "0.0500" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_shap_feature_importance_limits_to_top_n():
    importance = [{"feature": f"f{i}", "mean_abs_shap": 1.0 / (i + 1)} for i in range(30)]
    fig = plot_shap_feature_importance(importance, top_n=5)
    ax = fig.axes[0]
    assert len(ax.patches) == 5
    plt.close(fig)


def test_plot_class_imbalance_groups_by_horizon_and_split():
    distribution = [
        {"horizon_days": 14, "split": "train", "failure_rate": 0.02},
        {"horizon_days": 14, "split": "test", "failure_rate": 0.01},
        {"horizon_days": 30, "split": "train", "failure_rate": 0.05},
    ]
    fig = plot_class_imbalance(distribution)
    ax = fig.axes[0]
    # 2 horizons x 2 splits present = up to 4 bars (missing combos plot as 0)
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["train", "test"]
    plt.close(fig)


def test_plot_metric_comparison_has_val_and_test_bars():
    fig = plot_metric_comparison(METRICS, METRICS)
    ax = fig.axes[0]
    assert len(ax.patches) == 10  # 5 metrics x 2 (validation, test)
    plt.close(fig)


def test_plot_precision_at_k_sorts_by_k_ascending():
    fig = plot_precision_at_k(METRICS, title="test")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["1pct", "5pct", "10pct"]
    plt.close(fig)


def test_save_figure_writes_a_png_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out_path = tmp_path / "nested" / "figure.png"

    result = save_figure(fig, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.fignum_exists(fig.number)
