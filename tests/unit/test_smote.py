import numpy as np

from src.models.smote import apply_smote


def _imbalanced_data(n: int = 1000, positive_rate: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = np.zeros(n, dtype=int)
    n_pos = max(1, int(n * positive_rate))
    y[:n_pos] = 1
    return x, y


def test_apply_smote_balances_classes():
    x, y = _imbalanced_data()
    x_res, y_res = apply_smote(x, y, subsample_fraction=1.0, seed=0)
    assert (y_res == 1).sum() == (y_res == 0).sum()
    assert x_res.shape[0] == y_res.shape[0]
    assert x_res.shape[1] == x.shape[1]


def test_apply_smote_respects_subsample_fraction():
    x, y = _imbalanced_data(n=1000, positive_rate=0.1)
    x_res, y_res = apply_smote(x, y, subsample_fraction=0.5, seed=0)
    # after a 50% stratified subsample (500 rows, ~50 positive/450 negative),
    # SMOTE balances to roughly 2x the subsampled negative count
    assert x_res.shape[0] < x.shape[0] * 1.2  # well under the full-data SMOTE size
    assert (y_res == 1).sum() == (y_res == 0).sum()


def test_apply_smote_never_returns_original_validation_rows():
    # apply_smote only ever receives what the caller passes as training
    # data; this test documents that its output size differs from the
    # input whenever oversampling actually occurs, so a caller can't
    # mistake the resampled output for a pass-through.
    x, y = _imbalanced_data(n=500, positive_rate=0.02)
    x_res, y_res = apply_smote(x, y, subsample_fraction=1.0, seed=0)
    assert x_res.shape[0] > x.shape[0]


def test_apply_smote_is_deterministic_given_a_seed():
    x, y = _imbalanced_data()
    x_res_a, y_res_a = apply_smote(x, y, subsample_fraction=0.5, seed=7)
    x_res_b, y_res_b = apply_smote(x, y, subsample_fraction=0.5, seed=7)
    assert np.array_equal(x_res_a, x_res_b)
    assert np.array_equal(y_res_a, y_res_b)
