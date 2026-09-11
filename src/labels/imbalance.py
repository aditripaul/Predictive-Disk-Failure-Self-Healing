"""Class imbalance reporting and weighting (docs/dataset_strategy.md
section 15). Disk failure is rare (<1% of drive-days), so accuracy is a
misleading metric; AUPRC and precision-first thresholding drive evaluation
downstream (Phase 5), but the weighting factor is computed here alongside
the label table.
"""

from __future__ import annotations

import polars as pl


def compute_scale_pos_weight(labels: pl.DataFrame, *, split: str = "train") -> float | None:
    """`scale_pos_weight = negative_count / positive_count`, computed on the
    training split only (per docs: full validation distribution is used for
    final evaluation, not for weighting)."""
    train_labels = labels.filter((pl.col("split") == split) & pl.col("label").is_not_null())
    n_pos = train_labels.filter(pl.col("label") == 1).height
    n_neg = train_labels.filter(pl.col("label") == 0).height
    if n_pos == 0:
        return None
    return n_neg / n_pos


def class_distribution_report(labels: pl.DataFrame) -> list[dict]:
    """Per (horizon_days, split) counts of positive/negative/censored rows
    and the resulting failure rate, for the imbalance report."""
    report = (
        labels.group_by(["horizon_days", "split"])
        .agg(
            pl.len().alias("total_rows"),
            (pl.col("label") == 1).sum().alias("positive_count"),
            (pl.col("label") == 0).sum().alias("negative_count"),
            pl.col("label").is_null().sum().alias("censored_count"),
        )
        .with_columns(
            (pl.col("positive_count") / (pl.col("positive_count") + pl.col("negative_count")))
            .fill_nan(0.0)
            .alias("failure_rate")
        )
        .sort(["horizon_days", "split"])
    )
    return report.to_dicts()
