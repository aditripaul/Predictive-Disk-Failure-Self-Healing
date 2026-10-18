"""Where the model's false positives come from, and which ones a policy could cut.

A loop cannot change the ranking, so the only false positives it can remove
are ones that share something the score does not account for. This module
looks for that, on observations saved by `make agent-pit`, without re-scoring:

  * **Late hits.** A drive flagged at T, healthy for 30 days, failing in days
    31-60. Counted as a false alarm by a 30-day label, but not a wasted
    action. Measurement, not a fix - and reported beside the 30-day figure,
    never instead of it (the goal is pinned to 30 days, runbook 7).
  * **Composition.** Precision by drive model and by age. A family whose
    flags are mostly false, with enough drives to trust it, is a target.
  * **Flag streaks.** Precision of drives flagged for the first time against
    drives flagged in several consecutive snapshots without failing. A drive
    that holds a high score for weeks and survives has shown something.
  * **Attribute profile.** Which priority attributes are non-zero among
    false positives against true ones.

Every figure here is precision among FLAGGED drive-snapshots, at one tier.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

#: power_on_hours bucket edges, in years of continuous operation.
AGE_EDGES_YEARS = (1, 2, 3, 4, 5, 6)
HOURS_PER_YEAR = 8760


def flagged_at(obs: pl.DataFrame, threshold: float) -> pl.DataFrame:
    return obs.filter(pl.col("p_fail") >= threshold)


def late_hit_dates(dates: list[dt.date], *, data_end: dt.date, horizon_days: int) -> list[dt.date]:
    """Snapshot dates whose days 31-60 are fully observed. A later date would
    undercount late hits, because failures after the data ends are unseen."""
    return [d for d in dates if d + dt.timedelta(days=2 * horizon_days) <= data_end]


def late_hits(
    flagged: pl.DataFrame, future_positive: pl.DataFrame, *, horizon_days: int
) -> dict[str, float]:
    """Precision under the 30-day label, and with days 31-60 credited.

    `future_positive` is `drive_id, date` rows where the label at `date` is 1:
    the drive fails in (date, date + horizon]. A flag at T whose drive is
    positive at T + horizon therefore failed in (T + horizon, T + 2 horizon].
    """
    shifted = future_positive.select(
        "drive_id", (pl.col("date") - pl.duration(days=horizon_days)).alias("date")
    ).with_columns(pl.lit(True).alias("late"))
    joined = flagged.join(shifted, on=["drive_id", "date"], how="left").with_columns(
        pl.col("late").fill_null(False)
    )
    caught = int((joined["label"] == 1).sum())
    false_alarms = joined.filter(pl.col("label") == 0)
    late = int(false_alarms["late"].sum())
    n = joined.height
    return {
        "flagged": n,
        "caught": caught,
        "false_alarms": false_alarms.height,
        "late_hits": late,
        "late_share_of_false_alarms": late / false_alarms.height if false_alarms.height else 0.0,
        "precision_30d": caught / n if n else 0.0,
        "precision_with_late_credit": (caught + late) / n if n else 0.0,
    }


def precision_by_group(
    flagged: pl.DataFrame,
    column: str,
    *,
    population: pl.DataFrame | None = None,
    min_flagged: int = 30,
    top: int = 15,
) -> pl.DataFrame:
    """Precision per value of `column`, largest false-positive count first.

    Groups under `min_flagged` are folded into "(other)": a family with eight
    flags and zero hits is noise, not a target. `share_of_fp` says how much of
    the false-positive problem a group is, which is what decides whether
    fixing it matters.

    Pass `population` - every scored observation, flagged or not - to get each
    group's own base rate and **lift** (precision / base rate). Without it a
    low-precision group is ambiguous: the model may be weak there, or the
    group may just fail less often, and precision moves with the base rate
    (runbook 1.7). Read `plr` (recall / false-alarm rate) to compare groups:
    it does not move with the base rate. Lift does, mildly, once a group's
    base rate is no longer tiny."""
    if column not in flagged.columns:
        return pl.DataFrame()
    total_fp = int((flagged["label"] == 0).sum())
    stats = flagged.group_by(column).agg(
        pl.len().alias("flagged"), (pl.col("label") == 1).sum().alias("caught")
    )
    with_population = population is not None and column in population.columns
    if with_population:
        assert population is not None
        pop = population.group_by(column).agg(
            pl.len().alias("observations"), (pl.col("label") == 1).sum().alias("failing")
        )
        stats = pop.join(stats, on=column, how="left").with_columns(
            pl.col("flagged").fill_null(0), pl.col("caught").fill_null(0)
        )
    sums = ["flagged", "caught", *(["observations", "failing"] if with_population else [])]
    grouped = (
        stats.with_columns(
            pl.when(pl.col("flagged") >= min_flagged)
            .then(pl.col(column).cast(pl.String))
            .otherwise(pl.lit("(other)"))
            .alias(column)
        )
        .group_by(column)
        .agg([pl.col(c).sum() for c in sums])
        .filter(pl.col("flagged") > 0)
        .with_columns(
            (pl.col("flagged") - pl.col("caught")).alias("false_alarms"),
            (pl.col("caught") / pl.col("flagged")).alias("precision"),
        )
        .with_columns((pl.col("false_alarms") / max(total_fp, 1)).alias("share_of_fp"))
    )
    if with_population:
        grouped = (
            grouped.with_columns(
                (pl.col("failing") / pl.col("observations")).alias("base_rate"),
                (pl.col("caught") / pl.col("failing")).alias("recall"),
                (pl.col("false_alarms") / (pl.col("observations") - pl.col("failing"))).alias(
                    "false_alarm_rate"
                ),
            )
            .with_columns(
                (pl.col("precision") / pl.col("base_rate")).alias("lift"),
                # Lift still falls as a group's base rate rises at the same
                # ranking quality; PLR does not. A family failing at 0.8% looked
                # weak on lift and on precision for different reasons - PLR is
                # the column to read across families.
                (pl.col("recall") / pl.col("false_alarm_rate")).alias("plr"),
            )
            .drop("observations")
        )
    return grouped.sort("false_alarms", descending=True).head(top)


def age_bucket(flagged: pl.DataFrame) -> pl.DataFrame:
    """`flagged` with an `age_years` bucket label from power_on_hours."""
    if "power_on_hours" not in flagged.columns:
        return flagged
    years = pl.col("power_on_hours") / HOURS_PER_YEAR
    edges = (0, *AGE_EDGES_YEARS)
    labels = ["unknown", *[f"{lo}-{hi}y" for lo, hi in zip(edges, edges[1:], strict=False)]]
    labels.append(f"{edges[-1]}y+")
    # Bucket index: -1 unknown, then 0..len(edges)-1 by how many edges age passes.
    index = (
        pl.when(years.is_null())
        .then(pl.lit(-1))
        .otherwise(pl.sum_horizontal([(years >= e).cast(pl.Int32) for e in AGE_EDGES_YEARS]))
    )
    mapping = {i - 1: label for i, label in enumerate(labels)}
    return flagged.with_columns(
        index.replace_strict(mapping, return_dtype=pl.String).alias("age_years")
    )


def flag_streaks(
    by_date: dict[dt.date, pl.DataFrame], threshold: float, *, cap: int = 3
) -> pl.DataFrame:
    """For each flagged observation, how many immediately preceding snapshots
    also flagged the drive, capped at `cap`.

    Only dates with at least `cap` earlier snapshots are scored, so every
    streak value up to `cap` is actually possible there; otherwise early dates
    would read as 'first-time' merely for lack of history."""
    dates = sorted(by_date)
    flagged_ids = {d: set(flagged_at(by_date[d], threshold)["drive_id"].to_list()) for d in dates}
    rows = []
    for i, day in enumerate(dates):
        if i < cap:
            continue
        for row in flagged_at(by_date[day], threshold).select("drive_id", "label").iter_rows():
            streak = 0
            while streak < cap and row[0] in flagged_ids[dates[i - 1 - streak]]:
                streak += 1
            rows.append((row[0], day, row[1], streak))
    return pl.DataFrame(
        rows,
        schema={"drive_id": pl.String, "date": pl.Date, "label": pl.Int8, "streak": pl.Int64},
        orient="row",
    )


def attribute_profile(flagged: pl.DataFrame, attributes: tuple[str, ...]) -> pl.DataFrame:
    """Share of true and false positives with each attribute non-zero.

    An attribute non-zero far more often among false positives is one the
    model over-trusts; one non-zero mainly among true positives is doing its
    job."""
    rows = []
    hits, misses = flagged.filter(pl.col("label") == 1), flagged.filter(pl.col("label") == 0)
    for attribute in attributes:
        if attribute not in flagged.columns:
            continue

        def share(frame: pl.DataFrame, a: str = attribute) -> float:
            values = frame[a].to_numpy()
            values = values[~np.isnan(values.astype(float))] if len(values) else values
            return float((values > 0).mean()) if len(values) else 0.0

        rows.append(
            {
                "attribute": attribute,
                "nonzero_in_true_positives": share(hits),
                "nonzero_in_false_positives": share(misses),
            }
        )
    return pl.DataFrame(rows)
