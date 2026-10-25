"""Which failing drives the model misses, and how many any SMART model could catch.

The counterpart of `false_positives.py`. The docs say many failing drives give
no warning in their SMART counters and that this caps recall; this module
measures it, on the snapshots saved by `make agent-pit`, without re-scoring.

Each failing drive-snapshot is put in one of four classes by what its core
counters showed over the trailing window at the snapshot date - the telemetry
a policy at that date could have seen:

  * **no_data** - none of the core counters is reported (null).
  * **silent**  - every reported counter is zero throughout the window.
  * **static**  - some counter is non-zero, but none moved in the window. Many
    healthy drives look like this, so it is weak evidence.
  * **active**  - some counter rose inside the window.

A failing DRIVE takes the most informative class over its positive snapshots
(active > static > silent > no_data): the best chance the policy had. It is
caught if any positive snapshot scored at or above the threshold, which is how
`src/models/evaluation.drive_level_table` scores a failing drive.

Two readings follow. The share of failing drives that are silent or no_data
bounds the recall reachable from the core counters at any precision; the
recall inside `active` says how much of the reachable signal the model is
leaving behind at the chosen threshold.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Classes from least to most informative; a drive takes its highest.
SIGNAL_CLASSES = ("no_data", "silent", "static", "active")


def _class_order() -> pl.DataFrame:
    return pl.DataFrame(
        {"signal_class": list(SIGNAL_CLASSES), "_order": range(len(SIGNAL_CLASSES))}
    )


def window_columns(attributes: tuple[str, ...], window_days: int) -> list[str]:
    """The gold columns the classification reads: each attribute's trailing
    max (non-zero anywhere in the window?) and range (did it move?)."""
    return [f"{a}_{window_days}d_{op}" for a in attributes for op in ("max", "range")]


def signal_class(
    frame: pl.DataFrame, attributes: tuple[str, ...], *, window_days: int = 30
) -> pl.DataFrame:
    """`frame` with a `signal_class` column, from the gold window columns.

    An attribute whose window columns are absent from the frame is skipped,
    and a null value means the drive does not report that counter: it counts
    as neither zero nor non-zero. The window is the feature pipeline's, so it
    is counted in observations and a gap in telemetry lengthens it."""
    present = [
        a
        for a in attributes
        if f"{a}_{window_days}d_max" in frame.columns
        and f"{a}_{window_days}d_range" in frame.columns
    ]
    if not present:
        raise ValueError(f"no {window_days}-day max/range columns for {attributes} in the frame")
    reported = pl.any_horizontal([pl.col(f"{a}_{window_days}d_max").is_not_null() for a in present])
    nonzero = pl.any_horizontal(
        [pl.col(f"{a}_{window_days}d_max").fill_null(0) > 0 for a in present]
    )
    moved = pl.any_horizontal(
        [pl.col(f"{a}_{window_days}d_range").fill_null(0) > 0 for a in present]
    )
    return frame.with_columns(
        pl.when(moved)
        .then(pl.lit("active"))
        .when(nonzero)
        .then(pl.lit("static"))
        .when(reported)
        .then(pl.lit("silent"))
        .otherwise(pl.lit("no_data"))
        .alias("signal_class")
    )


def failing_drives(obs: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """One row per failing drive: its best signal class, its highest score
    over its positive snapshots, whether that caught it, and its drive model.

    `obs` needs `drive_id, p_fail, label, signal_class`. Only positive
    snapshots count: a flag before the drive's warning window is neither a
    catch nor a miss."""
    rank = {c: i for i, c in enumerate(SIGNAL_CLASSES)}
    positives = obs.filter(pl.col("label") == 1).with_columns(
        pl.col("signal_class").replace_strict(rank, return_dtype=pl.Int8).alias("_rank")
    )
    aggs = [
        pl.col("_rank").max().alias("_rank"),
        pl.col("p_fail").max().alias("max_p_fail"),
        pl.len().alias("positive_snapshots"),
    ]
    if "drive_model" in obs.columns:
        aggs.append(pl.col("drive_model").first())
    return (
        positives.group_by("drive_id")
        .agg(aggs)
        .with_columns(
            pl.col("_rank")
            .replace_strict({i: c for c, i in rank.items()}, return_dtype=pl.String)
            .alias("signal_class"),
            (pl.col("max_p_fail") >= threshold).alias("caught"),
        )
        .drop("_rank")
        .sort("drive_id")
    )


def recall_by_class(drives: pl.DataFrame) -> pl.DataFrame:
    """Failing drives, catches and recall per signal class, in class order,
    with each class's share of all failing drives and of all misses."""
    total = drives.height
    missed_total = int((~drives["caught"]).sum())
    stats = drives.group_by("signal_class").agg(
        pl.len().alias("failing_drives"), pl.col("caught").sum().alias("caught")
    )
    order = _class_order()
    return (
        order.join(stats, on="signal_class", how="left")
        .with_columns(pl.col("failing_drives").fill_null(0), pl.col("caught").fill_null(0))
        .with_columns(
            (pl.col("failing_drives") - pl.col("caught")).alias("missed"),
            (pl.col("failing_drives") / max(total, 1)).alias("share_of_failing"),
        )
        .with_columns(
            pl.when(pl.col("failing_drives") > 0)
            .then(pl.col("caught") / pl.col("failing_drives"))
            .otherwise(None)
            .alias("recall"),
            (pl.col("missed") / max(missed_total, 1)).alias("share_of_missed"),
        )
        .sort("_order")
        .drop("_order")
    )


def class_base_rates(obs: pl.DataFrame) -> pl.DataFrame:
    """Snapshot-level: each class's share of healthy and of failing
    snapshots, and the failure rate inside it. A class that holds failing
    snapshots far more often than healthy ones carries signal; `static`
    usually holds plenty of both, which is why its ceiling is optimistic."""
    healthy_total = max(int((obs["label"] == 0).sum()), 1)
    failing_total = max(int((obs["label"] == 1).sum()), 1)
    stats = obs.group_by("signal_class").agg(
        (pl.col("label") == 0).sum().alias("healthy"),
        (pl.col("label") == 1).sum().alias("failing"),
    )
    order = _class_order()
    return (
        order.join(stats, on="signal_class", how="inner")
        .with_columns(
            (pl.col("healthy") / healthy_total).alias("share_of_healthy"),
            (pl.col("failing") / failing_total).alias("share_of_failing"),
            (pl.col("failing") / (pl.col("healthy") + pl.col("failing"))).alias("failure_rate"),
        )
        .with_columns((pl.col("share_of_failing") / pl.col("share_of_healthy")).alias("lift"))
        .sort("_order")
        .drop("_order")
    )


def recall_ceiling(drives: pl.DataFrame) -> dict[str, float]:
    """Upper bounds on drive recall from the core counters, at any precision.

    `with_static` assumes a non-zero but unmoving counter can be told apart
    from the healthy drives that show the same - optimistic, since many do.
    `active_only` assumes it cannot. The model's own recall is beside them."""
    n = max(drives.height, 1)
    counts = {c: int((drives["signal_class"] == c).sum()) for c in SIGNAL_CLASSES}
    return {
        "failing_drives": drives.height,
        "model_recall": float(drives["caught"].sum()) / n,
        "ceiling_with_static": (counts["active"] + counts["static"]) / n,
        "ceiling_active_only": counts["active"] / n,
    }


def false_alarm_rate_to_catch(drives: pl.DataFrame, healthy_scores: np.ndarray) -> pl.DataFrame:
    """For each failing drive, the share of healthy snapshots scoring at or
    above its best score: the false-alarm rate a single threshold would have
    to accept to catch it. Joined back as `far_to_catch`.

    Snapshot-level, as the point-in-time tiers are; a near-zero value for a
    missed drive means it sat just under the threshold, a large one means the
    model ranks it among ordinary drives."""
    healthy = np.sort(np.asarray(healthy_scores, dtype=float))
    if healthy.size == 0:
        raise ValueError("no healthy snapshots to compare against")
    best = drives["max_p_fail"].to_numpy().astype(float)
    above = healthy.size - np.searchsorted(healthy, best, side="left")
    return drives.with_columns(pl.Series("far_to_catch", above / healthy.size))


def far_quantiles(
    drives: pl.DataFrame, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> pl.DataFrame:
    """Quantiles of `far_to_catch` among MISSED drives, per signal class."""
    missed = drives.filter(~pl.col("caught"))
    aggs = [pl.len().alias("missed")] + [
        pl.col("far_to_catch").quantile(q, interpolation="linear").alias(f"far_q{int(q * 100)}")
        for q in quantiles
    ]
    order = _class_order()
    return (
        order.join(missed.group_by("signal_class").agg(aggs), on="signal_class", how="inner")
        .sort("_order")
        .drop("_order")
    )


def class_by_family(drives: pl.DataFrame, *, min_failing: int = 10, top: int = 15) -> pl.DataFrame:
    """Per drive model: failing drives, recall, and the share whose counters
    gave nothing (silent or no_data). A weak family with a high silent share
    is short of telemetry; one with a low share is where the model loses
    signal it had. Families under `min_failing` fold into "(other)"."""
    if "drive_model" not in drives.columns:
        return pl.DataFrame()
    return (
        drives.with_columns(
            pl.col("signal_class").is_in(["silent", "no_data"]).alias("_quiet"),
            pl.col("drive_model").count().over("drive_model").alias("_n"),
        )
        .with_columns(
            pl.when(pl.col("_n") >= min_failing)
            .then(pl.col("drive_model").cast(pl.String))
            .otherwise(pl.lit("(other)"))
            .alias("drive_model")
        )
        .group_by("drive_model")
        .agg(
            pl.len().alias("failing_drives"),
            pl.col("caught").mean().alias("recall"),
            pl.col("_quiet").mean().alias("quiet_share"),
            (pl.col("signal_class") == "active").mean().alias("active_share"),
            pl.col("caught")
            .filter(pl.col("signal_class") == "active")
            .mean()
            .alias("recall_in_active"),
        )
        .sort("failing_drives", descending=True)
        .head(top)
    )
