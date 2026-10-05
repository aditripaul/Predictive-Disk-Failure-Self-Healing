import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.reporting.figure_data import failure_counts_by_family, gain_importance, roc_points
from src.reporting.plots import plot_failures_by_family, plot_feature_importance, plot_roc_curves


def test_roc_points_auc_matches_a_perfect_ranker():
    y = np.array([0, 0, 1, 1])
    curve = roc_points(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert curve["auc"] == pytest.approx(1.0)
    assert curve["fpr"][0] == 0.0 and curve["tpr"][-1] == 1.0


def test_roc_points_with_one_class_is_empty_not_an_error():
    curve = roc_points(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]))
    assert curve == {"fpr": [], "tpr": [], "auc": None}


def test_roc_points_thins_long_curves():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    curve = roc_points(y, rng.random(5000) + 0.3 * y, max_points=50)
    assert len(curve["fpr"]) <= 50


def test_gain_importance_is_normalized_and_sorted():
    result = gain_importance(["a", "b", "c"], np.array([1.0, 3.0, 0.0]), top_n=2)
    assert [r["feature"] for r in result] == ["b", "a"]
    assert result[0]["importance"] == pytest.approx(0.75)


def test_failure_counts_by_family_counts_drives_and_failures():
    metadata = pl.DataFrame(
        {
            "model_family": ["X", "X", "X", "Y"],
            "failure_date": [dt.date(2026, 1, 5), None, None, None],
        }
    )
    rows = {r["model_family"]: r for r in failure_counts_by_family(metadata)}
    assert rows["X"] == {
        "model_family": "X",
        "drives": 3,
        "failed_drives": 1,
        "failure_share": pytest.approx(1 / 3),
    }
    assert rows["Y"]["failed_drives"] == 0


def test_plot_functions_draw_figures_from_the_data():
    importance = gain_importance(["a", "b"], np.array([2.0, 1.0]))
    assert plot_feature_importance(importance, title="t") is not None
    curves = {"model": roc_points(np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.9]))}
    assert plot_roc_curves(curves, title="t") is not None
    rows = failure_counts_by_family(
        pl.DataFrame({"model_family": ["X"], "failure_date": [dt.date(2026, 1, 1)]})
    )
    assert plot_failures_by_family(rows) is not None
