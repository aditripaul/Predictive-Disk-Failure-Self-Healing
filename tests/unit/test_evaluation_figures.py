"""The figures' numbers, not their pixels: each confusion matrix must equal
`drive_level_metrics` at the same threshold, and each marker must sit on the
curve it annotates."""

import numpy as np
import pytest

from src.models.evaluation import drive_level_metrics, recall_at_top_k
from src.reporting.evaluation_figures import (
    evaluation_figure_data,
    thin_curve,
    write_evaluation_figures,
)


def _fleet(seed: int = 0, drives: int = 400, days: int = 5):
    rng = np.random.default_rng(seed)
    drive_ids = np.repeat([f"d{i:04d}" for i in range(drives)], days)
    failing = rng.random(drives) < 0.1
    y = np.zeros(drives * days, dtype=int)
    for i in np.flatnonzero(failing):
        y[i * days + days - 2 : (i + 1) * days] = 1  # last two days in the window
    scores = rng.random(drives * days) * 0.6 + y * rng.random(drives * days) * 0.5
    return drive_ids, y, scores


def test_confusion_counts_equal_drive_level_metrics():
    drive_ids, y, scores = _fleet()
    thresholds = {"primary": 0.7, "warn": 0.5}
    data = evaluation_figure_data(drive_ids, y, scores, thresholds)
    for point in data["operating_points"]:
        expected = drive_level_metrics(drive_ids, y, scores, point.threshold)
        assert point.caught == expected["caught_drive_count"]
        assert point.false_alarms == expected["false_alarm_drive_count"]
        assert point.caught + point.missed == expected["failing_drive_count"]
        assert point.precision == pytest.approx(expected["precision"])
        assert point.recall == pytest.approx(expected["recall"])


def test_operating_points_lie_on_the_drive_curve():
    drive_ids, y, scores = _fleet(seed=1)
    data = evaluation_figure_data(drive_ids, y, scores, {"primary": 0.65}, row_level=False)
    (curve,) = data["curves"]
    (point,) = data["operating_points"]
    on_curve = np.isclose(curve.precision, point.precision) & np.isclose(curve.recall, point.recall)
    assert on_curve.any()


def test_unreachable_tier_is_skipped():
    drive_ids, y, scores = _fleet()
    data = evaluation_figure_data(drive_ids, y, scores, {"primary": 0.7, "drain": None})
    assert [p.name for p in data["operating_points"]] == ["primary"]


def test_thinning_keeps_both_ends():
    precision = np.linspace(0, 1, 10_001)
    recall = precision[::-1]
    p, r = thin_curve(precision, recall, max_points=100)
    assert len(p) <= 100
    assert (p[0], r[0], p[-1], r[-1]) == (precision[0], recall[0], precision[-1], recall[-1])


def test_top_k_counts_the_best_scored_drives():
    drive_ids = np.array(["a", "a", "b", "c", "d"])
    y = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.3, 0.2])
    rows = recall_at_top_k(drive_ids, y, scores, [1, 2, 3, 10])
    assert [(r["caught"], r["false_alarms"]) for r in rows] == [(1, 0), (1, 1), (2, 1), (2, 2)]
    assert rows[0]["precision"] == 1.0 and rows[0]["recall"] == 0.5
    assert rows[3]["flagged"] == 4  # capped at the number of drives


def test_write_evaluation_figures_writes_both_files(tmp_path):
    drive_ids, y, scores = _fleet()
    data = write_evaluation_figures(
        tmp_path,
        "train",
        drive_ids,
        y,
        scores,
        {"primary": 0.7},
        in_sample=True,
        top_k=[10, 40],
        log_to_mlflow=False,
    )
    assert sorted(p.name for p in data["paths"]) == [
        "train_confusion.png",
        "train_precision_recall.png",
    ]
    assert all(p.stat().st_size > 0 for p in data["paths"])
    assert "in-sample, optimistic" in data["title"]
    assert [row["k"] for row in data["top_k"]] == [10, 40]
