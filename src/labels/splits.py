"""Temporal and drive-level splitting (docs/dataset_strategy.md section 9).

Random splitting is forbidden because it leaks future information; splits
must be chronological, with an optional drive-level holdout to test whether
the model is memorizing specific drives rather than generalizing.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

SPLIT_VERSION = "v1.0"


def add_chronological_split(
    df: pl.DataFrame,
    *,
    train_end: str | dt.date,
    validation_end: str | dt.date,
    date_column: str = "date",
    sealed_start: str | dt.date | None = None,
) -> pl.DataFrame:
    """Adds a `split` column: "train" up to `train_end`, "validation" up to
    `validation_end`, "test" afterwards. From `sealed_start` on, rows are
    "sealed": a final window that no model choice may look at. Only the
    frozen-model evaluation (pipelines/evaluate_frozen.py) reads it, once."""
    train_end = _to_date(train_end)
    validation_end = _to_date(validation_end)

    split_expr = (
        pl.when(pl.col(date_column) <= train_end)
        .then(pl.lit("train"))
        .when(pl.col(date_column) <= validation_end)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
    )
    if sealed_start is not None:
        split_expr = (
            pl.when(pl.col(date_column) >= _to_date(sealed_start))
            .then(pl.lit("sealed"))
            .otherwise(split_expr)
        )

    return df.with_columns(
        split_expr.alias("split"),
        pl.lit(SPLIT_VERSION).alias("split_version"),
        pl.lit("chronological").alias("split_strategy"),
    )


def sample_holdout_drive_ids(
    drive_ids: pl.Series | list[str], *, holdout_fraction: float = 0.1, seed: int = 42
) -> list[str]:
    """The random ~`holdout_fraction` subset of `drive_ids` that
    `apply_drive_level_holdout` reassigns to validation. Extracted so a
    caller processing a dataset in batches (see
    `pipelines/build_labels.py`) can sample this set ONCE, from the full
    fleet-wide drive list, and pass the SAME set to every batch's
    `apply_drive_level_holdout` call - resampling independently per batch
    would give each batch its own random ~10%, silently breaking the
    "these drives, consistently held out everywhere" guarantee the whole
    point of this holdout depends on."""
    unique_ids = pl.Series("drive_id", drive_ids).unique().sort()
    if holdout_fraction <= 0:
        # A chronological validation period (docs/dataset_strategy.md section 9.5) holds no
        # drives out; max(1, ...) below would still move one drive.
        return []
    n_holdout = max(1, int(unique_ids.len() * holdout_fraction))
    return unique_ids.sample(n=n_holdout, seed=seed).to_list()


def apply_drive_level_holdout(
    df: pl.DataFrame,
    *,
    holdout_fraction: float = 0.1,
    seed: int = 42,
    holdout_drive_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Reassigns a random subset of drives so they appear only in
    validation/test, never in train — testing generalization to unseen
    physical units. Assumes `add_chronological_split` has already run.

    Pass a precomputed `holdout_drive_ids` (from `sample_holdout_drive_ids`,
    run once over the full fleet) when calling this once per batch of a
    larger, batched dataset - otherwise each batch samples its own
    independent ~`holdout_fraction`, which is wrong (see
    `sample_holdout_drive_ids`'s docstring). Left as `None` (the default),
    this samples from `df`'s own drive_id column exactly as before, which
    is correct only when `df` already contains the whole fleet."""
    if holdout_drive_ids is None:
        drive_ids = df.select("drive_id").unique().sort("drive_id")
        n_holdout = max(1, int(drive_ids.height * holdout_fraction)) if holdout_fraction > 0 else 0
        holdout_drive_ids = drive_ids.sample(n=n_holdout, seed=seed)["drive_id"].to_list()

    is_holdout = pl.col("drive_id").is_in(holdout_drive_ids)
    df = df.with_columns(
        pl.when(is_holdout & (pl.col("split") == "train"))
        .then(pl.lit("validation"))
        .otherwise(pl.col("split"))
        .alias("split"),
        pl.when(is_holdout)
        .then(pl.lit("drive_holdout"))
        .otherwise(pl.col("split_strategy"))
        .alias("split_strategy"),
    )
    return df


def apply_vendor_holdout(
    df: pl.DataFrame,
    *,
    source_column: str = "source_dataset",
    external_source: str = "smartz",
) -> pl.DataFrame:
    """External split (docs/dataset_strategy.md section 9.3/9.4): every row
    from `external_source` is reassigned `split="external_smartz"` and
    `split_strategy="vendor_holdout"`, overriding whatever the chronological
    (+ drive-holdout) split assigned. SMART-Z exists to test cross-vendor
    generalization and identify Backblaze-specific overfitting, so it must
    never leak into train/validation - it is evaluation-only. A no-op if the
    frame carries no `source_column` (e.g. a Backblaze-only build)."""
    if source_column not in df.columns:
        return df

    is_external = pl.col(source_column) == external_source
    return df.with_columns(
        pl.when(is_external)
        .then(pl.lit("external_smartz"))
        .otherwise(pl.col("split"))
        .alias("split"),
        pl.when(is_external)
        .then(pl.lit("vendor_holdout"))
        .otherwise(pl.col("split_strategy"))
        .alias("split_strategy"),
    )


def _to_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


# --- split periods: shared by `make train`, the separation checks and the tests -------------


def split_boundary(model_config: dict, name: str) -> dt.date | None:
    """`splits.<name>` as a date, or None when unset."""
    value = model_config.get("splits", {}).get(name)
    return dt.date.fromisoformat(str(value)) if value else None


def resolved_test_end(model_config: dict, horizon_days: int) -> dt.date | None:
    """Last date of the test split: `splits.test_end`, capped at
    `splits.observation_end - horizon_days` when that is set. A later row's
    horizon runs past the data, so a healthy row there has no label (it is
    dropped) while a failing one keeps its positive label - those dates would
    add failing drives with no healthy rows beside them. The cap lets runs of
    different horizons share one config and each use all the test data its
    labels allow (30 days: 08-31, 60 days: 08-01 for data ending 09-30)."""
    test_end = split_boundary(model_config, "test_end")
    observation_end = split_boundary(model_config, "observation_end")
    if observation_end is None:
        return test_end
    cap = observation_end - dt.timedelta(days=horizon_days)
    return cap if test_end is None else min(test_end, cap)


def embargo_days(model_config: dict) -> int:
    """`splits.test_embargo_days`: days after `validation_end` that no test row may use. 0
    (the default) keeps the test split starting right after validation, as before. A
    validation row's label looks `horizon_days` ahead, so a test period that starts sooner
    than that after validation ends is scored on outcomes the validation labels already
    carry (docs/dataset_strategy.md section 9.5)."""
    return int(model_config.get("splits", {}).get("test_embargo_days") or 0)


def feature_stats_cutoff(model_config: dict) -> dt.date | None:
    """Rows before this date feed the fleet-wide feature statistics (model-family z-scores,
    expected SMART attributes): `splits.feature_stats_end` when set, else `splits.sealed_start`
    (the behaviour before the setting existed). None means all rows."""
    return split_boundary(model_config, "feature_stats_end") or split_boundary(
        model_config, "sealed_start"
    )


def split_period_predicate(
    split_name: str, model_config: dict, horizon_days: int
) -> pl.Expr | None:
    """The dates each split may use, beyond the label table's own `split`
    column. `split_name` is the role ("train", "validation", "test"), not
    the label-table split the test rows come from (`test_source_split`).

    - **train:** rows on or after `train_start` (when set). With
      `purge_label_window` (default on), only rows dated at least
      `horizon_days` before `train_end`: a later row's label is decided by
      what happens after `train_end`, inside a later validation period.
      Turn it off when validation shares the training dates.
    - **validation:** rows on or after `train_start` (when set). With
      `validation_after_train_only` (default on), only rows after
      `train_end`; the label table also puts the held-out drives' rows from
      the training period here, and thresholds chosen on those carried over
      to test worse than ones chosen on a later period.
    - **test:** only rows up to `resolved_test_end`, and, with
      `test_embargo_days`, only rows more than that many days after `validation_end`."""
    splits = model_config.get("splits", {})
    train_start = split_boundary(model_config, "train_start")
    train_end = split_boundary(model_config, "train_end")
    predicates: list[pl.Expr] = []
    if split_name in ("train", "validation") and train_start is not None:
        predicates.append(pl.col("date") >= train_start)
    if split_name == "train" and train_end and splits.get("purge_label_window", True):
        predicates.append(pl.col("date") <= train_end - dt.timedelta(days=horizon_days))
    if split_name == "validation" and train_end and splits.get("validation_after_train_only", True):
        predicates.append(pl.col("date") > train_end)
    if split_name == "test":
        test_end = resolved_test_end(model_config, horizon_days)
        if test_end is not None:
            predicates.append(pl.col("date") <= test_end)
        embargo = embargo_days(model_config)
        validation_end = split_boundary(model_config, "validation_end")
        if embargo > 0 and validation_end is not None:
            predicates.append(pl.col("date") > validation_end + dt.timedelta(days=embargo))
    if not predicates:
        return None
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = combined & predicate
    return combined
