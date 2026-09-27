"""Property-based tests for secondary evaluation metrics
(`src/models/evaluation.py`): precision-at-K is always a valid fraction,
and calibration binning always accounts for every row exactly once
regardless of how the random scores happen to fall into bins.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.evaluation import compute_calibration, precision_at_k

LABELS = st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=200)


@given(
    y_true=LABELS, seed=st.integers(min_value=0, max_value=1_000_000), k=st.integers(min_value=1)
)
@settings(max_examples=50)
def test_precision_at_k_is_always_a_valid_fraction(y_true, seed, k):
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_scores = rng.uniform(size=len(y_true))

    result = precision_at_k(y_true_arr, y_scores, k=k)
    assert 0.0 <= result <= 1.0


@given(
    y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=300),
    seed=st.integers(min_value=0, max_value=1_000_000),
    n_bins=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=50)
def test_calibration_bin_counts_always_sum_to_the_total_row_count(y_true, seed, n_bins):
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_scores = rng.uniform(size=len(y_true))

    result = compute_calibration(y_true_arr, y_scores, n_bins=n_bins)
    assert sum(b["count"] for b in result["bins"]) == len(y_true)
    assert len(result["bins"]) == n_bins
    assert 0.0 <= result["expected_calibration_error"] <= 1.0 + 1e-9
    assert result["brier_score"] >= 0.0
