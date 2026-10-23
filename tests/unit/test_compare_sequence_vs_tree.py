import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "compare_sequence_vs_tree", Path("scripts/compare_sequence_vs_tree.py")
)
assert _spec and _spec.loader
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def test_recall_at_far_respects_the_false_alarm_budget():
    labels = np.array([0] * 1000 + [1] * 4)
    healthy = np.linspace(0, 0.9, 1000)
    scores = np.concatenate([healthy, [0.95, 0.5, 0.2, 0.99]])
    recall, threshold = compare._recall_at_far(labels, scores, 0.001)  # 1 healthy alarm allowed
    flagged_healthy = int((healthy > threshold).sum())
    assert flagged_healthy <= 1
    assert recall == 0.5  # only the two failing drives above the threshold


def test_rank_is_in_unit_interval_and_order_preserving():
    ranks = compare._rank(np.array([0.3, 0.1, 0.2]))
    assert ranks.argsort().tolist() == [1, 2, 0] and 0 < ranks.min() and ranks.max() == 1.0
