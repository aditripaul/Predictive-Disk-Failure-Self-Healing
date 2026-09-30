"""Feature Family A — time-window rolling aggregates
(docs/dataset_strategy.md section 10.1).

Operates on a wide per-drive-day frame (see src/features/pivot.py), one
attribute per column. Windows are time-based (`Nd`) and grouped by drive_id
so one drive's history never leaks into another drive's window (leakage
rule 2 in docs/dataset_strategy.md section 17).
"""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path

import polars as pl

from src.logging_config import get_logger

DEFAULT_WINDOWS_DAYS = (7, 14, 30)

logger = get_logger(__name__)


def add_rolling_aggregates(
    df: pl.DataFrame,
    attributes: list[str],
    *,
    windows_days: tuple[int, ...] = DEFAULT_WINDOWS_DAYS,
) -> pl.DataFrame:
    """Adds `{attr}_{window}d_{mean,median,min,max,std,range}` for every
    attribute and window. `df` must be sorted by `date` (globally) so that
    every drive's own rows are individually in non-decreasing date order -
    `.rolling(group_by=...)` below only needs that, not a full
    `[drive_id, date]` sort (verified: it tolerates groups appearing in
    any relative order, as long as each group's own index-column values
    are ascending)."""
    # Roll from a fixed, narrow base (just the id columns + the original
    # attributes) and never join `df`, or any growing accumulator, inside
    # this loop. A join's cost scales with the width of both sides
    # (verified: joining a 50-extra-column frame measurably costs ~2x a
    # narrow one, at equal row count), so accumulating windows via
    # `combined = combined.join(rolled, ...)` still hits the same wall
    # once enough windows have piled up - confirmed on real data: 7d and
    # 14d succeeded (`combined` then 32/62 columns), but 30d crashed once
    # `combined` had grown to 62 columns going in.
    #
    # Every window's `rolled` result is rolled from the exact same `base`,
    # so - verified empirically - each one comes back with the identical
    # (drive_id, date) row order as every other one (even though that
    # order need not match `base`'s own row order). That means they can
    # be combined with a single horizontal concat instead of a join: no
    # hashing, no repeated key-matching. But keeping every window's
    # `rolled` frame alive in a Python list until that final concat
    # reintroduced a different growing-memory problem - confirmed on real
    # data: doing that made the crash happen *earlier* (during the 14d
    # window itself) than the join-based version had (which crashed at
    # 30d), because the old join-based loop freed each `rolled` as soon as
    # it was absorbed into `combined`, while the list keeps ALL of them
    # resident at once. So each window's `rolled` is written to its own
    # temp Parquet file and freed immediately instead; the final combine
    # reads them back lazily (still via one cheap horizontal concat, no
    # joins) so at most one window's worth of rolled data is ever resident
    # in RAM at a time, matching the join-based version's memory shape
    # while keeping the join-based version's per-window cost problem gone.
    base = df.select(["drive_id", "date", *attributes])
    with tempfile.TemporaryDirectory() as tmp_dir:
        window_paths = []
        for window in windows_days:
            t0 = time.perf_counter()
            agg_exprs = []
            for attr in attributes:
                period = f"{window}d"
                agg_exprs.extend(
                    [
                        pl.col(attr).mean().alias(f"{attr}_{window}d_mean"),
                        pl.col(attr).median().alias(f"{attr}_{window}d_median"),
                        pl.col(attr).min().alias(f"{attr}_{window}d_min"),
                        pl.col(attr).max().alias(f"{attr}_{window}d_max"),
                        pl.col(attr).std().fill_null(0.0).alias(f"{attr}_{window}d_std"),
                    ]
                )
            rolled = base.rolling(index_column="date", period=period, group_by="drive_id").agg(
                agg_exprs
            )
            for attr in attributes:
                rolled = rolled.with_columns(
                    (pl.col(f"{attr}_{window}d_max") - pl.col(f"{attr}_{window}d_min")).alias(
                        f"{attr}_{window}d_range"
                    )
                )
            window_path = Path(tmp_dir) / f"window_{window}.parquet"
            rolled.write_parquet(window_path, compression="zstd")
            logger.info(
                "add_rolling_aggregates_window_done",
                window_days=window,
                elapsed_seconds=round(time.perf_counter() - t0, 2),
                row_count=rolled.height,
                column_count=len(rolled.columns),
            )
            window_paths.append(window_path)
            del rolled
            gc.collect()

        combined = pl.concat(
            [
                pl.scan_parquet(window_paths[0]).select(["drive_id", "date"]),
                *[pl.scan_parquet(p).drop(["drive_id", "date"]) for p in window_paths],
            ],
            how="horizontal_extend",
        ).collect()

    return df.join(combined, on=["drive_id", "date"], how="left")
