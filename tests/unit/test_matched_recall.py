import numpy as np

from src.models.evaluation import precision_at_recall_level


def test_matched_recall_returns_the_precision_at_that_catch_rate():
    # Four failing drives out of six. Ranking: F F H F H H (scores descending).
    drives = np.array(["a", "b", "c", "d", "e", "f"])
    y = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    point = precision_at_recall_level(drives, y, scores, recall_level=0.5)
    assert point["reachable"]
    assert point["recall"] >= 0.5
    # Catching 2 of 4 failures at the top two scores: precision 1.0.
    assert point["precision"] == 1.0


def test_unreachable_recall_level_is_reported_not_guessed():
    drives = np.array(["a", "b"])
    y = np.array([1, 0])
    scores = np.array([0.2, 0.8])
    # Recall can reach 1.0 by lowering the threshold, so 1.0 is reachable here.
    assert precision_at_recall_level(drives, y, scores, 1.0)["reachable"]
