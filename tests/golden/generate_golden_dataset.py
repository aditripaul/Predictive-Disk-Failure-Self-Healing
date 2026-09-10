"""Generates the small golden dataset used for regression smoke tests.

Produces synthetic healthy and failing drive-day trajectories (not derived
from real Backblaze data) so CI can run without external data access. The
real golden dataset described in the project plan should be built from
sampled Backblaze trajectories once ingestion (Phase 1) lands.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

OUTPUT_PATH = Path(__file__).parent / "golden_dataset.parquet"

N_HEALTHY_DRIVES = 100
N_FAILING_DRIVES = 100
HISTORY_DAYS = 60
START_DATE = dt.date(2023, 1, 1)


def _make_drive_rows(drive_id: str, failing: bool, rng: np.random.Generator) -> list[dict]:
    rows = []
    reallocated = 0.0
    pending = 0.0
    for day in range(HISTORY_DAYS):
        date = START_DATE + dt.timedelta(days=day)
        if failing:
            progress = day / HISTORY_DAYS
            reallocated += rng.poisson(progress * 2)
            pending += rng.poisson(progress * 3)
        else:
            reallocated += rng.poisson(0.02)
            pending = max(0.0, pending + rng.normal(0, 0.1))

        rows.append(
            {
                "drive_id": drive_id,
                "date": date,
                "reallocated_sector_count": float(reallocated),
                "current_pending_sector_count": float(max(pending, 0.0)),
                "power_on_hours": float(day * 24),
                "label": int(failing and day >= HISTORY_DAYS - 14),
            }
        )
    return rows


def build_golden_dataset() -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    for i in range(N_HEALTHY_DRIVES):
        rows.extend(_make_drive_rows(f"healthy-{i:03d}", failing=False, rng=rng))
    for i in range(N_FAILING_DRIVES):
        rows.extend(_make_drive_rows(f"failing-{i:03d}", failing=True, rng=rng))
    return pl.DataFrame(rows)


if __name__ == "__main__":
    df = build_golden_dataset()
    df.write_parquet(OUTPUT_PATH, compression="zstd")
    print(f"Wrote {df.height} rows to {OUTPUT_PATH}")
