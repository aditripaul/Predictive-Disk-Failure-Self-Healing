"""Dense per-drive telemetry and 30-day windows for the sequence model.

`src/features/sequences.py` (the optional LSTM branch) builds ONE window per
drive, ending at the drive's last observed day. That cannot be compared with
the tree model, which is scored on every labelled drive-day and then reduced
to one score per drive. This module instead holds each drive's whole history
as a calendar-aligned tensor `[drives, days, channels]` and cuts a window
ending at ANY (drive, day) on demand, so the sequence model can be trained on,
and score, exactly the rows `pipelines/train_model.py` uses.

Channels, per attribute `a` of `A`: the log1p value (0 where not reported),
then a "reported" mask per attribute, then the log1p drive age and a
"telemetry present that day" flag. A vendor that does not report an attribute
(Seagate-only `smart_187` / `smart_188`) is therefore distinguishable from a
drive that reports zero, which `fill_null(0.0)` hides in the tree pipeline.

Values are stored as float16 (a 273-day, 363k-drive fleet is ~4.4GB that way)
and standardized on the fly with statistics taken from training dates only.
numpy and polars only: nothing here imports torch.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

#: Attributes left on their own scale instead of log1p (it can be negative or
#: is a level, not a counter).
LEVEL_ATTRIBUTES = frozenset({"temperature_celsius"})

#: Manufacturers given their own one-hot slot; everything else shares "other".
KNOWN_MANUFACTURERS = ("seagate", "hgst", "toshiba", "wdc", "western digital", "samsung")


@dataclass
class DenseTelemetry:
    values: np.ndarray  # float16 [drives, days, channels]
    drive_ids: list[str]  # row i of `values` is drive_ids[i]
    start_date: dt.date  # day index 0
    attributes: list[str]
    static: np.ndarray  # float32 [drives, static_features]
    mean: np.ndarray | None = None  # float32 [channels]; set by `fit_normalization`
    std: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[2])

    @property
    def n_days(self) -> int:
        return int(self.values.shape[1])

    def day_index(self, dates: pl.Series) -> np.ndarray:
        return ((dates - self.start_date).dt.total_days()).to_numpy().astype(np.int64)


def channel_count(n_attributes: int) -> int:
    return 2 * n_attributes + 2


def _static_features(manufacturer: list[str | None], capacity_gb: np.ndarray) -> np.ndarray:
    onehot = np.zeros((len(manufacturer), len(KNOWN_MANUFACTURERS) + 1), dtype=np.float32)
    for i, name in enumerate(manufacturer):
        key = (name or "").strip().lower()
        slot = KNOWN_MANUFACTURERS.index(key) if key in KNOWN_MANUFACTURERS else -1
        onehot[i, slot] = 1.0
    capacity = np.nan_to_num(capacity_gb.astype(np.float64), nan=0.0)
    log_capacity = np.log1p(np.maximum(capacity, 0.0))
    if len(log_capacity) == 0:
        return onehot[:, :0].reshape(0, static_feature_count())
    spread = log_capacity.std() or 1.0
    standardized = ((log_capacity - log_capacity.mean()) / spread).astype(np.float32)
    return np.concatenate([onehot, standardized[:, None]], axis=1)


def static_feature_count() -> int:
    return len(KNOWN_MANUFACTURERS) + 2


def build_dense_telemetry(
    features_path: Path,
    *,
    attributes: list[str],
    start_date: dt.date,
    end_date: dt.date,
    drive_fraction: float = 1.0,
) -> DenseTelemetry:
    """Reads the gold feature table one Parquet row group at a time (the file
    is ~4GB; see `join_by_native_row_groups` for why not a plain scan) and
    scatters each row into `values[drive, day]`.

    `drive_fraction < 1` keeps a deterministic hash-subset of drives, for smoke
    runs; every window and label for a kept drive is still complete."""
    lazy = pl.scan_parquet(features_path)
    available = set(lazy.collect_schema().names())
    missing = [a for a in attributes if a not in available]
    if missing:
        raise ValueError(f"attributes not in the gold feature table: {missing}")

    drives = (
        lazy.group_by("drive_id")
        .agg(
            pl.col("manufacturer").first().alias("manufacturer"),
            pl.col("capacity_gb").first().alias("capacity_gb"),
        )
        .sort("drive_id")
        .collect()
    )
    if drive_fraction < 1.0:
        keep = (pl.col("drive_id").hash(seed=0) % 10_000) < int(drive_fraction * 10_000)
        drives = drives.filter(keep)
    drives = drives.with_row_index("drive_idx")
    n_drives = drives.height
    n_days = (end_date - start_date).days + 1
    n_attr = len(attributes)
    values = np.zeros((n_drives, n_days, channel_count(n_attr)), dtype=np.float16)

    index = drives.select(["drive_id", pl.col("drive_idx").cast(pl.Int64)])
    columns = ["drive_id", "date", *attributes, "drive_age_days"]
    parquet = pq.ParquetFile(features_path)
    for group in range(parquet.num_row_groups):
        frame = pl.from_arrow(parquet.read_row_group(group, columns=columns))
        assert isinstance(frame, pl.DataFrame)
        frame = frame.join(index, on="drive_id", how="inner")
        if frame.height == 0:
            continue
        day = ((frame["date"] - start_date).dt.total_days()).to_numpy().astype(np.int64)
        inside = (day >= 0) & (day < n_days)
        if not inside.all():
            frame, day = frame.filter(pl.Series(inside)), day[inside]
        drive = frame["drive_idx"].to_numpy()
        for c, name in enumerate(attributes):
            raw = frame[name].cast(pl.Float64).to_numpy()
            reported = ~np.isnan(raw)
            raw = np.nan_to_num(raw, nan=0.0)
            transformed = raw if name in LEVEL_ATTRIBUTES else np.log1p(np.maximum(raw, 0.0))
            values[drive, day, c] = transformed
            values[drive, day, n_attr + c] = reported
        age = np.nan_to_num(frame["drive_age_days"].cast(pl.Float64).to_numpy(), nan=0.0)
        values[drive, day, 2 * n_attr] = np.log1p(np.maximum(age, 0.0))
        values[drive, day, 2 * n_attr + 1] = 1.0

    capacity = drives["capacity_gb"].cast(pl.Float64).to_numpy()
    return DenseTelemetry(
        values=values,
        drive_ids=drives["drive_id"].to_list(),
        start_date=start_date,
        attributes=list(attributes),
        static=_static_features(drives["manufacturer"].to_list(), capacity),
    )


def fit_normalization(dense: DenseTelemetry, *, last_day: int, chunk_drives: int = 20_000) -> None:
    """Per-channel mean/std over REPORTED cells on days `<= last_day`, so no
    statistic comes from validation or test dates. Mask channels keep mean 0,
    std 1 (they are not rescaled)."""
    n_attr = len(dense.attributes)
    n_value = n_attr + 1  # attributes + age
    total = np.zeros(n_value)
    total_sq = np.zeros(n_value)
    count = np.zeros(n_value)
    last = min(last_day, dense.n_days - 1) + 1
    for start in range(0, dense.values.shape[0], chunk_drives):
        block = dense.values[start : start + chunk_drives, :last].astype(np.float64)
        present = np.concatenate(
            [block[:, :, n_attr : 2 * n_attr], block[:, :, 2 * n_attr + 1 :]], axis=2
        )
        value = np.concatenate([block[:, :, :n_attr], block[:, :, 2 * n_attr : 2 * n_attr + 1]], 2)
        value = value * present
        total += value.sum(axis=(0, 1))
        total_sq += (value**2).sum(axis=(0, 1))
        count += present.sum(axis=(0, 1))
    mean_v = total / np.maximum(count, 1)
    var_v = total_sq / np.maximum(count, 1) - mean_v**2
    std_v = np.sqrt(np.maximum(var_v, 0.0))
    std_v = np.where(std_v < 1e-3, 1.0, std_v)

    mean = np.zeros(dense.n_channels, dtype=np.float32)
    std = np.ones(dense.n_channels, dtype=np.float32)
    mean[:n_attr], std[:n_attr] = mean_v[:n_attr], std_v[:n_attr]
    mean[2 * n_attr], std[2 * n_attr] = mean_v[n_attr], std_v[n_attr]
    dense.mean, dense.std = mean, std


def gather_windows(
    dense: DenseTelemetry, drive_idx: np.ndarray, day_idx: np.ndarray, window_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(x, static)` for each (drive, day): `x` is `[batch, window_days,
    channels]` float32 ending on `day_idx` (inclusive), standardized, with days
    before the start of the data and days a drive did not report left as zero;
    `static` is `[batch, static_features]`."""
    if dense.mean is None or dense.std is None:
        raise ValueError("call fit_normalization first")
    n_attr = len(dense.attributes)
    offsets = np.arange(-(window_days - 1), 1)
    days = day_idx[:, None] + offsets[None, :]
    in_range = days >= 0
    x = dense.values[drive_idx[:, None], np.clip(days, 0, None)].astype(np.float32)
    x[~in_range] = 0.0
    reported = x[:, :, n_attr : 2 * n_attr]
    present = x[:, :, 2 * n_attr + 1 : 2 * n_attr + 2]
    x[:, :, :n_attr] = (x[:, :, :n_attr] - dense.mean[:n_attr]) / dense.std[:n_attr] * reported
    age = x[:, :, 2 * n_attr : 2 * n_attr + 1]
    x[:, :, 2 * n_attr : 2 * n_attr + 1] = (
        (age - dense.mean[2 * n_attr]) / dense.std[2 * n_attr] * present
    )
    return x, dense.static[drive_idx]
