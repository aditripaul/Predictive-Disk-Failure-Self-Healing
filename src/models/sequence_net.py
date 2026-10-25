"""LSTM classifier for the sequence-vs-tree comparison (needs the `torch`
extra: `uv sync --extra torch`). Never imported by `pipelines/train_model.py`.

The network is deliberately plain: an LSTM over the 30-day window, the final
hidden state joined with the per-drive static vector (vendor, capacity), and a
small MLP head. What the comparison tests is whether a recurrent reading of the
raw daily history ranks failing drives differently from the tree model's
hand-built window features, not whether a bigger network can be tuned.

Training mirrors the tree model's choices so the two differ in architecture,
not in recipe: a positive-class weight of `(n_neg / n_pos) ** power`, early
stopping on average precision over a validation subsample, thresholds chosen
elsewhere on validation only.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

#: `(row_indices) -> (x [B, T, C], static [B, S])`, supplied by the caller so the
#: dense telemetry never has to be copied into one array per split.
BatchFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class SequenceNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_static: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static = nn.Sequential(nn.Linear(n_static, 16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        joined = torch.cat([hidden[-1], self.static(static)], dim=1)
        logits: torch.Tensor = self.head(joined).squeeze(-1)
        return logits


def predict_scores(
    model: SequenceNet, batch_fn: BatchFn, n_rows: int, *, batch_size: int = 16384
) -> np.ndarray:
    """Probabilities for rows `0..n_rows-1`, in order."""
    model.eval()
    out = np.empty(n_rows, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            rows = np.arange(start, min(start + batch_size, n_rows))
            x, static = batch_fn(rows)
            logits = model(torch.from_numpy(x), torch.from_numpy(static))
            out[rows] = torch.sigmoid(logits).numpy()
    return out


def train_sequence_net(
    model: SequenceNet,
    train_batch_fn: BatchFn,
    y_train: np.ndarray,
    *,
    evaluate: Callable[[SequenceNet], float],
    positive_weight_power: float = 0.5,
    learning_rate: float = 2e-3,
    batch_size: int = 2048,
    max_epochs: int = 12,
    patience: int = 3,
    seed: int = 0,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> tuple[SequenceNet, list[dict[str, float]]]:
    """Trains `model` and returns it with the best-validation weights restored,
    plus per-epoch `{"epoch", "loss", "validation_average_precision"}`.

    `evaluate(model)` returns the validation average precision; training stops
    after `patience` epochs without improvement (the tree model's rule)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    weight = (n_neg / n_pos) ** positive_weight_power if n_pos > 0 else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    stale = 0
    history: list[dict[str, float]] = []
    n = len(y_train)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            rows = np.sort(order[start : start + batch_size])
            x, static = train_batch_fn(rows)
            logits = model(torch.from_numpy(x), torch.from_numpy(static))
            loss = loss_fn(logits, torch.from_numpy(y_train[rows].astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        score = float(evaluate(model))
        mean_loss = total_loss / n
        history.append({"epoch": epoch, "loss": mean_loss, "validation_average_precision": score})
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, score)
        if score > best_score:
            best_score, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history
