import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="LSTM branch requires `uv sync --extra torch`")

from src.models.evaluation import compute_auprc  # noqa: E402
from src.models.lstm import SequenceLSTM, predict_proba_positive_lstm, train_lstm  # noqa: E402


def _toy_sequence_data(n: int = 200, time_steps: int = 10, features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, time_steps, features)).astype(np.float32)
    # label depends on the last timestep's first feature - a signal only
    # visible if the model actually attends to the end of the sequence.
    y = (x[:, -1, 0] > 0).astype(np.float32)
    return x, y


def test_sequence_lstm_forward_produces_one_logit_per_sample():
    model = SequenceLSTM(input_size=3, hidden_size=8)
    x = torch.randn(5, 10, 3)
    logits = model(x)
    assert logits.shape == (5,)


def test_train_lstm_and_predict_proba_recovers_a_learnable_signal():
    x, y = _toy_sequence_data()
    model = train_lstm(x, y, hidden_size=8, n_epochs=15, batch_size=16, seed=0)
    scores = predict_proba_positive_lstm(model, x)

    assert scores.shape == (200,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert compute_auprc(y, scores) > 0.7


def test_train_lstm_is_deterministic_given_a_seed():
    x, y = _toy_sequence_data()
    model_a = train_lstm(x, y, hidden_size=8, n_epochs=5, batch_size=16, seed=42)
    model_b = train_lstm(x, y, hidden_size=8, n_epochs=5, batch_size=16, seed=42)

    scores_a = predict_proba_positive_lstm(model_a, x)
    scores_b = predict_proba_positive_lstm(model_b, x)
    assert np.allclose(scores_a, scores_b)


def test_train_lstm_handles_severe_class_imbalance_without_error():
    x, y = _toy_sequence_data(n=100)
    y = np.zeros_like(y)
    y[:2] = 1.0  # 2 positives out of 100 - exercises the pos_weight branch
    model = train_lstm(x, y, hidden_size=4, n_epochs=3, batch_size=16, seed=0)
    scores = predict_proba_positive_lstm(model, x)
    assert scores.shape == (100,)


def test_train_lstm_handles_zero_positive_labels_without_dividing_by_zero():
    x, y = _toy_sequence_data(n=50)
    y = np.zeros_like(y)
    model = train_lstm(x, y, hidden_size=4, n_epochs=2, batch_size=16, seed=0)
    scores = predict_proba_positive_lstm(model, x)
    assert scores.shape == (50,)
