"""Sequence feature tensors for the optional LSTM branch
(docs/dataset_strategy.md section 10.7 "Feature Family G — Sequence
Features for LSTM Branch" / section 16.2 "LSTM / Sequence Branch").

Builds `[samples, time_steps, features]` tensors from the gold wide
feature table, with time-decay sample weights (recent days matter more
than older ones), and writes them as memory-mapped `.npy` files rather
than holding them as large in-memory arrays (docs section 10.7 storage
recommendation).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

#: docs/dataset_strategy.md section 10.7's suggested initial 10 attributes.
DEFAULT_SEQUENCE_ATTRIBUTES: tuple[str, ...] = (
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
    "raw_read_error_rate",
    "seek_error_rate",
    "spin_retry_count",
    "command_timeout",
    "power_on_hours",
    "smart_overall_health_normalized",
)

DEFAULT_TIME_STEPS = 30


def compute_time_decay_weights(time_steps: int, *, lambda_: float = 0.1) -> np.ndarray:
    """`weight_t = exp(-lambda * age_in_days)` (docs section 10.7): index 0
    is the OLDEST day in the window (highest age, lowest weight), index
    `time_steps - 1` is the most recent day (age 0, weight 1.0)."""
    ages = np.arange(time_steps - 1, -1, -1, dtype=np.float64)
    return np.exp(-lambda_ * ages)


def apply_time_decay_weights(tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scales each time step of `tensor` (`[samples, time_steps,
    features]`) by its `weights` entry (`[time_steps]`, as produced by
    `compute_time_decay_weights`) - "encourages the model to focus on
    recent degradation" (docs section 10.7) by damping older days'
    magnitude before the sequence ever reaches the model, rather than
    relying on the model to learn that weighting on its own."""
    if weights.shape[0] != tensor.shape[1]:
        raise ValueError(
            f"weights length {weights.shape[0]} must match tensor time_steps {tensor.shape[1]}"
        )
    return tensor * weights[np.newaxis, :, np.newaxis]


def build_sequence_tensors(
    wide_features: pl.DataFrame,
    *,
    attributes: tuple[str, ...] = DEFAULT_SEQUENCE_ATTRIBUTES,
    time_steps: int = DEFAULT_TIME_STEPS,
) -> tuple[np.ndarray, list[str], list[str], list[dt.date]]:
    """For every `drive_id`, takes the last `time_steps` days of
    `attributes` values (sorted by date, nulls filled with 0.0) ending at
    that drive's last observed day - one sample per drive, the "samples"
    axis of `[samples, time_steps, features]`. Drives with fewer than
    `time_steps` days of history are left-padded with zeros (not dropped):
    a young drive is exactly the case the model most needs early signal
    on, and feature-maturity gating elsewhere already governs whether a
    *prediction* on it is trusted.

    Returns `(tensor, attributes_used, drive_ids, as_of_dates)`, where
    `tensor.shape == (len(drive_ids), time_steps, len(attributes_used))`
    and `attributes_used` is the subset of `attributes` actually present
    in `wide_features` (in the same order)."""
    attributes_used = [a for a in attributes if a in wide_features.columns]
    if not attributes_used:
        raise ValueError(f"None of {attributes} found in wide_features columns")

    drive_ids: list[str] = []
    as_of_dates: list[dt.date] = []
    samples: list[np.ndarray] = []

    sorted_features = wide_features.sort(["drive_id", "date"])
    for (drive_id,), group in sorted_features.group_by("drive_id", maintain_order=True):
        tail = group.tail(time_steps)
        values = tail.select(attributes_used).fill_null(0.0).to_numpy().astype(np.float64)

        n_missing = time_steps - values.shape[0]
        if n_missing > 0:
            pad = np.zeros((n_missing, values.shape[1]), dtype=np.float64)
            values = np.vstack([pad, values])

        drive_ids.append(drive_id)
        as_of_dates.append(tail["date"][-1])
        samples.append(values)

    tensor = np.stack(samples).astype(np.float32)
    return tensor, attributes_used, drive_ids, as_of_dates


def write_sequence_tensors(
    tensor: np.ndarray,
    *,
    attributes_used: list[str],
    drive_ids: list[str],
    as_of_dates: list[dt.date],
    out_dir: Path,
) -> Path:
    """Writes the tensor as a memory-mapped `.npy` file (never fully
    materialized twice in memory) plus a small Parquet sidecar mapping
    each tensor row to its `drive_id`/`as_of_date`, and a JSON sidecar
    recording which attributes (and in what order) fill the feature axis
    - all three are needed to make the raw tensor interpretable later."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "sequences.npy"

    memmap = np.lib.format.open_memmap(
        tensor_path, mode="w+", dtype=tensor.dtype, shape=tensor.shape
    )
    memmap[:] = tensor
    memmap.flush()

    pl.DataFrame({"drive_id": drive_ids, "as_of_date": as_of_dates}).write_parquet(
        out_dir / "sequences_metadata.parquet"
    )
    (out_dir / "sequences_attributes.json").write_text(json.dumps(attributes_used))
    return tensor_path


def load_sequence_tensors(out_dir: Path) -> tuple[np.ndarray, pl.DataFrame, list[str]]:
    """Reads back what `write_sequence_tensors` wrote. The tensor is
    memory-mapped read-only (`mode="r"`), not loaded fully into RAM."""
    tensor: np.ndarray = np.lib.format.open_memmap(out_dir / "sequences.npy", mode="r")
    metadata = pl.read_parquet(out_dir / "sequences_metadata.parquet")
    attributes_used: list[str] = json.loads((out_dir / "sequences_attributes.json").read_text())
    return tensor, metadata, attributes_used
