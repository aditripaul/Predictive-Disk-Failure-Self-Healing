"""Checks that train, validation and test stay separate (docs/dataset_strategy.md section 9.5).

A row's label looks `horizon_days` ahead, so two periods are separate only if the later one starts
more than one horizon after the last labelled row of the earlier one, and if nothing fitted on the
training period saw a later row. `splits.strict_separation: true` turns these checks on for
`make train`; `scripts/check_split_separation.py` (`make check-splits`) runs them on demand and
reads the label table. With the setting off (the default) nothing here runs, because the
drive-holdout design of runbook 1.1.8 shares dates between train and validation by construction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from src.labels.splits import (
    embargo_days,
    feature_stats_cutoff,
    resolved_test_end,
    split_boundary,
    split_period_predicate,
)


@dataclass(frozen=True)
class SeparationViolation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


class SplitSeparationError(ValueError):
    """Raised when `splits.strict_separation` is on and a check fails."""


def strict_separation_enabled(model_config: dict) -> bool:
    return bool(model_config.get("splits", {}).get("strict_separation", False))


def check_split_config(model_config: dict, horizon_days: int) -> list[SeparationViolation]:
    """Violations of the separation rules in `configs/model.yaml`'s `splits`, for one horizon."""
    splits = model_config.get("splits", {})
    found: list[SeparationViolation] = []

    def fail(rule: str, detail: str) -> None:
        found.append(SeparationViolation(rule, detail))

    if not splits.get("purge_label_window", True):
        fail(
            "purge_label_window",
            "off: train labels run past train_end into the validation period",
        )
    if not splits.get("validation_after_train_only", True):
        fail(
            "validation_after_train_only",
            "off: validation shares dates with train, so it is not a later period",
        )
    if float(splits.get("holdout_fraction", 0) or 0) > 0:
        fail(
            "holdout_fraction",
            "above 0: held-out drives' training-period rows would become validation rows",
        )

    train_start = split_boundary(model_config, "train_start")
    train_end = split_boundary(model_config, "train_end")
    validation_end = split_boundary(model_config, "validation_end")
    if train_end is None or validation_end is None:
        fail("boundaries", "train_end and validation_end must both be set")
        return found
    if train_start is not None and not train_start < train_end:
        fail("boundaries", f"train_start {train_start} is not before train_end {train_end}")
    if not train_end < validation_end:
        fail("boundaries", f"train_end {train_end} is not before validation_end {validation_end}")

    horizon = dt.timedelta(days=horizon_days)
    embargo = embargo_days(model_config)
    if embargo < horizon_days:
        fail(
            "test_embargo_days",
            f"{embargo} is shorter than the {horizon_days}-day horizon: validation labels "
            "reach into the test period",
        )
    test_end = resolved_test_end(model_config, horizon_days)
    test_first = validation_end + dt.timedelta(days=embargo + 1)
    if test_end is None:
        fail("test_end", "neither test_end nor observation_end is set")
    elif test_end < test_first:
        fail("test_end", f"the test period is empty: it would start {test_first}, end {test_end}")

    cutoff = feature_stats_cutoff(model_config)
    last_train_row = train_end - horizon
    if split_boundary(model_config, "feature_stats_end") is None:
        fail(
            "feature_stats_end",
            "unset: the feature statistics fall back to sealed_start and use validation rows",
        )
    elif cutoff is not None and cutoff > last_train_row + dt.timedelta(days=1):
        fail(
            "feature_stats_end",
            f"{cutoff} is after the day following the last train row ({last_train_row}): "
            "later rows shape the training features",
        )
    return found


def check_label_table_separation(
    labels: pl.LazyFrame, model_config: dict, horizon_days: int, *, test_split: str
) -> list[SeparationViolation]:
    """Violations in the rows the three roles would actually read from the label table.

    `labels` needs `date`, `horizon_days`, `label` and `split`. Each role is read exactly as
    `make train` reads it (`split_period_predicate`, plus the role's label-table split and an
    observed label); the check is on the resulting date ranges."""
    horizon = dt.timedelta(days=horizon_days)
    base = labels.filter((pl.col("horizon_days") == horizon_days) & pl.col("label").is_not_null())
    ranges: dict[str, tuple[dt.date, dt.date] | None] = {}
    for role, source_split in (
        ("train", "train"),
        ("validation", "validation"),
        ("test", test_split),
    ):
        rows = base.filter(pl.col("split") == source_split)
        predicate = split_period_predicate(role, model_config, horizon_days)
        if predicate is not None:
            rows = rows.filter(predicate)
        bounds = rows.select(
            pl.col("date").min().alias("a"), pl.col("date").max().alias("b")
        ).collect()
        first, last = bounds["a"][0], bounds["b"][0]
        ranges[role] = None if first is None else (first, last)

    found: list[SeparationViolation] = []
    for role, span in ranges.items():
        if span is None:
            found.append(SeparationViolation("empty", f"no {role} rows with an observed label"))
    train, validation, test = ranges["train"], ranges["validation"], ranges["test"]
    if train and validation and not validation[0] > train[1] + horizon:
        found.append(
            SeparationViolation(
                "label_horizon",
                f"validation starts {validation[0]}, but train labels run to "
                f"{train[1] + horizon} ({train[1]} + {horizon_days} days)",
            )
        )
    if validation and test and not test[0] > validation[1] + horizon:
        found.append(
            SeparationViolation(
                "label_horizon",
                f"test starts {test[0]}, but validation labels run to "
                f"{validation[1] + horizon} ({validation[1]} + {horizon_days} days)",
            )
        )
    if train and test and not test[0] > train[1] + horizon:
        found.append(
            SeparationViolation(
                "label_horizon",
                f"test starts {test[0]}, but train labels run to {train[1] + horizon}",
            )
        )
    return found


def raise_on_violations(violations: list[SeparationViolation], *, where: str) -> None:
    if violations:
        lines = "\n".join(f"  - {v}" for v in violations)
        raise SplitSeparationError(
            f"splits.strict_separation is on and {where} breaks it:\n{lines}\n"
            "See docs/dataset_strategy.md section 9.5, or set splits.strict_separation: false."
        )
