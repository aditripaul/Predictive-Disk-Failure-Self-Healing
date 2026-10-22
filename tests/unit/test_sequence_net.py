import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.models.sequence_net import SequenceNet, predict_scores, train_sequence_net  # noqa: E402


def _data(n=600, steps=10, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, steps, channels)).astype(np.float32)
    # failing windows have a rising channel 0 over time; healthy ones are noise
    y = (rng.random(n) < 0.2).astype(np.float32)
    ramp = np.linspace(0, 2, steps, dtype=np.float32)[None, :]
    x[:, :, 0] += y[:, None] * ramp
    static = rng.normal(size=(n, 4)).astype(np.float32)
    return x, static, y


def test_net_learns_a_rising_trend_and_keeps_row_order():
    x, static, y = _data()
    split = 400

    def batch(arrays):
        xs, ss = arrays

        def fn(rows):
            return xs[rows], ss[rows]

        return fn

    train_fn = batch((x[:split], static[:split]))
    val_fn = batch((x[split:], static[split:]))
    model = SequenceNet(n_channels=3, n_static=4, hidden_size=16, num_layers=1)

    def evaluate(m):
        return float(average_precision_score(y[split:], predict_scores(m, val_fn, len(y) - split)))

    model, history = train_sequence_net(
        model, train_fn, y[:split], evaluate=evaluate, max_epochs=15, patience=5, batch_size=64
    )
    scores = predict_scores(model, val_fn, len(y) - split, batch_size=37)
    assert scores.shape == (len(y) - split,)
    assert roc_auc_score(y[split:], scores) > 0.85
    assert history and history[0]["epoch"] == 1


def test_early_stopping_restores_the_best_weights():
    x, static, y = _data(n=200)

    def fn(rows):
        return x[rows], static[rows]

    calls = iter([0.5, 0.9, 0.1, 0.1, 0.1, 0.1])
    model = SequenceNet(n_channels=3, n_static=4, hidden_size=8, num_layers=1)
    _, history = train_sequence_net(
        model, fn, y, evaluate=lambda m: next(calls), max_epochs=10, patience=2, batch_size=64
    )
    assert [h["validation_average_precision"] for h in history] == [0.5, 0.9, 0.1, 0.1]
