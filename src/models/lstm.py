"""Optional LSTM sequence-based comparison branch (docs/dataset_strategy.md
section 16.2 "LSTM / Sequence Branch"; docs/design_goal.md: "Optional
sequence-based model comparison using LSTM"). Requires the `torch` extra
(`uv sync --extra torch`) - this module is never imported by the primary
training pipeline (`pipelines/train_model.py`), only by its own entry
point (`pipelines/train_lstm.py`), so the default install stays free of
PyTorch's footprint.

"The LSTM branch should be compared against the tree-based baseline, not
assumed to be superior" (docs section 16.2) - this module only ever
produces a comparison AUPRC, never a production model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SequenceLSTM(nn.Module):
    """A small LSTM classifier over `[batch, time_steps, features]`
    sequence tensors (docs section 16.2's recommended input format), with
    dropout for regularization (per the doc's "dropout and early
    stopping" recommendation - early stopping is the training loop's job,
    not the model's)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_layer_hidden = hidden[-1]  # [batch, hidden_size]
        logits: torch.Tensor = self.classifier(self.dropout(last_layer_hidden)).squeeze(-1)
        return logits  # [batch] - raw logits, not probabilities


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 32,
    seed: int = 0,
) -> SequenceLSTM:
    """`x_train` is `[samples, time_steps, features]`, `y_train` is
    `[samples]` (0/1). Uses `BCEWithLogitsLoss(pos_weight=...)` for the
    same rare-failure class imbalance the tree models handle via
    `is_unbalance`/`scale_pos_weight` (docs/dataset_strategy.md section
    15)."""
    torch.manual_seed(seed)

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])

    model = SequenceLSTM(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    n = x_tensor.shape[0]

    model.train()
    for _ in range(n_epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_tensor[idx])
            loss = loss_fn(logits, y_tensor[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_proba_positive_lstm(model: SequenceLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        probabilities: torch.Tensor = torch.sigmoid(logits)
    return probabilities.numpy()
