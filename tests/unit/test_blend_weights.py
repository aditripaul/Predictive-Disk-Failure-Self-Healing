"""The blend-weight search: the grid, the fast drive-level evaluation, and its guards."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.blend_weights import (
    DriveRows,
    bootstrap_difference,
    objective,
    pipeline_criterion,
    recall_at_far,
    simplex_grid,
)
from src.models.evaluation import drive_level_table
from src.models.threshold import tune_drive_level_threshold


def _rows(n_drives: int = 400, rows_per_drive: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    drive_ids = np.repeat([f"D{i:04d}" for i in range(n_drives)], rows_per_drive)
    failing = rng.random(n_drives) < 0.08
    y = np.zeros(n_drives * rows_per_drive, dtype=np.int8)
    for d in np.flatnonzero(failing):  # a failing drive's last 5 rows are inside its window
        y[d * rows_per_drive + rows_per_drive - 5 : (d + 1) * rows_per_drive] = 1
    signal = np.repeat(failing, rows_per_drive) * 1.0 + rng.normal(0, 0.8, len(y))
    return drive_ids, y, signal


def test_simplex_grid_covers_every_vector_once() -> None:
    grid = simplex_grid(4, 0.1)
    assert grid.shape == (286, 4)  # C(13, 3)
    assert np.allclose(grid.sum(axis=1), 1.0) and (grid >= 0).all()
    assert len({tuple(r) for r in np.round(grid, 6)}) == 286
    assert any(np.allclose(r, [0, 0, 0, 1]) for r in grid)
    with pytest.raises(ValueError):
        simplex_grid(4, 0.3)


def test_fast_drive_scores_match_the_repos_drive_level_table() -> None:
    drive_ids, y, scores = _rows()
    labels_ref, scores_ref = drive_level_table(drive_ids, y, scores)
    rows = DriveRows.build(drive_ids, y)
    fast = rows.drive_scores(rows.arrange(scores))
    # drive order differs, so compare the (label, score) pairs as multisets
    ref_pairs = sorted(zip(labels_ref.tolist(), scores_ref.tolist(), strict=True))
    fast_pairs = sorted(zip(rows.labels.tolist(), fast.tolist(), strict=True))
    assert fast_pairs == ref_pairs


def test_a_weight_vector_is_one_matrix_product() -> None:
    drive_ids, y, a = _rows(seed=1)
    b = np.random.default_rng(2).random(len(a))
    rows = DriveRows.build(drive_ids, y)
    arranged = rows.arrange(np.column_stack([a, b]))
    w = np.array([0.3, 0.7])
    assert np.allclose(
        rows.drive_scores(arranged @ w), rows.drive_scores(rows.arrange(0.3 * a + 0.7 * b))
    )


@pytest.mark.parametrize("target_precision", [0.5, 2.0])  # goal reachable / not reachable
def test_pipeline_criterion_picks_the_point_tune_drive_level_threshold_picks(
    target_precision,
) -> None:
    drive_ids, y, scores = _rows(seed=3)
    ref = tune_drive_level_threshold(
        drive_ids, y, scores, target_precision=target_precision, target_recall_range=(0.1, 0.5)
    )
    rows = DriveRows.build(drive_ids, y)
    mine = pipeline_criterion(
        rows.labels,
        rows.drive_scores(rows.arrange(scores)),
        target_precision=target_precision,
        recall_floor=0.1,
    )
    assert mine["threshold"] == pytest.approx(ref["threshold"])
    assert mine["precision"] == pytest.approx(ref["precision"])
    assert mine["recall"] == pytest.approx(ref["recall"])


def test_recall_at_far_is_one_for_a_perfect_ranking_and_low_for_noise() -> None:
    labels = np.r_[np.ones(50), np.zeros(5000)].astype(int)
    perfect = np.r_[np.ones(50), np.zeros(5000)]
    noise = np.random.default_rng(0).random(5050)
    assert recall_at_far(labels, perfect) == [1.0, 1.0, 1.0]
    assert objective(labels, noise) < 0.2


def test_bootstrap_of_identical_scores_straddles_zero() -> None:
    labels = np.r_[np.ones(60), np.zeros(6000)].astype(int)
    scores = np.random.default_rng(1).random(6060) + labels * 0.5
    out = bootstrap_difference(labels, scores, scores, n_boot=30, seed=0)
    assert out["difference"] == 0.0 and out["ci_low"] <= 0 <= out["ci_high"]
    assert out["se_a"] > 0


def test_objective_follows_the_false_alarm_budgets_it_is_given() -> None:
    rng = np.random.default_rng(5)
    labels = np.r_[np.ones(200), np.zeros(20000)].astype(int)
    scores = rng.random(20200) + labels * 0.8
    low = objective(labels, scores, (1e-4, 2e-4))
    high = objective(labels, scores, (1e-2, 5e-2))
    assert 0 <= low < high <= 1  # more alerts allowed, more failures caught
    assert objective(labels, scores) == objective(labels, scores, (1e-4, 2e-4, 5e-4))
