"""Rank-blend ensemble and split/leak checks for the merged approach
(`pipelines/train_merged_models.py`).

Models on different scales (class-weighted LightGBM scores, AFT/Cox risks,
sequence-net probabilities) are blended by their rank against the
VALIDATION distribution of the same model (aryan_v2 `evaluate.val_rank` /
`rank_ens`): each score becomes the share of validation scores at or below
it, and the blend is the mean of those shares. The reference is validation
only, so test scores are mapped without looking at test.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import polars as pl


def validation_rank(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Share of finite `reference` (validation) scores <= each score, in [0, 1].
    Non-finite scores stay NaN."""
    ref = np.asarray(reference, dtype=np.float64)
    ref = np.sort(ref[np.isfinite(ref)])
    if len(ref) == 0:
        raise ValueError("validation reference has no finite scores")
    x = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.searchsorted(ref, np.where(finite, x, 0.0), side="right") / len(ref)
    return np.where(finite, out, np.nan)


def rank_blend(
    validation: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    members: Sequence[str],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted mean of the members' validation-referenced ranks of `target`.
    `validation[m]` is model m's validation scores (the reference);
    `target[m]` the scores to blend (validation itself, or test)."""
    if not members:
        raise ValueError("an ensemble needs at least one member")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(members) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"bad ensemble weights {list(w)} for members {list(members)}")
    w = w / w.sum()
    blended = np.zeros(len(target[members[0]]), dtype=np.float64)
    for weight, name in zip(w, members, strict=True):
        if len(target[name]) != len(blended):
            raise ValueError(
                f"member {name!r} has {len(target[name])} rows, expected {len(blended)}"
            )
        blended += weight * validation_rank(validation[name], target[name])
    return blended


def assert_split_dates(
    dates: dict[str, pl.Series],
    *,
    train_last_row: dt.date,
    validation_start: dt.date,
    validation_end: dt.date,
    test_start: dt.date,
    test_end: dt.date,
) -> dict[str, dict[str, str]]:
    """Raises unless every split's row dates lie inside its window and the
    windows are ordered train < validation < test. Returns min/max per split."""
    bounds = {
        "train": (None, train_last_row),
        "validation": (validation_start, validation_end),
        "test": (test_start, test_end),
    }
    if not (train_last_row < validation_start <= validation_end < test_start <= test_end):
        raise AssertionError(f"split windows out of order: {bounds}")
    seen: dict[str, dict[str, str]] = {}
    for name, (lo, hi) in bounds.items():
        if name not in dates:
            continue
        col = dates[name]
        if col.len() == 0:
            raise AssertionError(f"split {name!r} is empty")
        first, last = col.min(), col.max()
        if not isinstance(first, dt.date) or not isinstance(last, dt.date):
            raise AssertionError(f"split {name!r} date column is not a date: {col.dtype}")
        if (lo is not None and first < lo) or last > hi:
            raise AssertionError(
                f"split {name!r} rows run {first!s}..{last!s}, outside its window {lo}..{hi}"
            )
        seen[name] = {"min_date": str(first), "max_date": str(last)}
    return seen


def assert_no_row_overlap(ids: dict[str, pl.DataFrame]) -> None:
    """Raises if any (drive_id, date) row appears in two splits."""
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = (
                ids[a]
                .select("drive_id", "date")
                .join(ids[b].select("drive_id", "date"), on=["drive_id", "date"], how="inner")
            )
            if shared.height:
                raise AssertionError(f"{shared.height} (drive_id, date) rows in both {a} and {b}")


def assert_survival_censored(
    dates: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, outcome_cutoff: dt.date
) -> None:
    """Raises if a survival target uses an outcome after `outcome_cutoff`:
    every observed failure time and every censoring time, added to its row's
    date, must land on or before it."""
    days = np.asarray(dates).astype("datetime64[D]")
    horizon = np.where(np.isfinite(upper), upper, lower)
    # MIN_SURVIVAL_DAYS lifts zero-day times to 0.5; floor keeps that on the row's day.
    reached = days + np.floor(horizon).astype("timedelta64[D]")
    latest = reached.max() if len(reached) else None
    if latest is not None and latest > np.datetime64(outcome_cutoff, "D"):
        raise AssertionError(
            f"survival target reaches {latest}, after the split's outcome cutoff {outcome_cutoff}"
        )
