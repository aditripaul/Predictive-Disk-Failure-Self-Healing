"""Synthetic placeholder dataset for exercising the full pipeline offline
- ingestion through training - without any real data download.

Unlike a trivial fixture, this generates real degrading-vs-healthy
trajectories (mirroring `tests/golden/generate_golden_dataset.py`, but in
raw Backblaze-shaped columns so it goes through the actual Bronze/Silver/
Gold pipeline, not a pre-computed gold-feature shape) spanning
`configs/model.yaml`'s default chronological split boundaries
(`train_end`/`validation_end`/`test_end`), with failing drives' failure
dates spread across the whole window so `make train` sees real positive
and negative labels in every split, not just the trivial fixture this
used to be. The real trajectory-morphed chaos generator (noise injection,
timeline compression, correlated multi-drive failures) is built in
Phase 8.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata

# Spans configs/model.yaml's default chronological split boundaries
# (train_end=2021-12-31, validation_end=2022-12-31, test_end=2023-12-31)
# so every split has real drive-days out of the box, without editing config.
START_DATE = dt.date(2021, 1, 1)
END_DATE = dt.date(2023, 6, 1)

N_HEALTHY_DRIVES = 6
N_FAILING_DRIVES = 9
FAILING_HISTORY_DAYS_CAP = 180  # pre-failure history length per failing drive

CAPACITY_BYTES = 4_000_787_030_016
DRIVE_MODEL = "ST4000DM000"  # matches the "ST" prefix -> Seagate HDD family


def _row(drive_id: str, date: dt.date, *, reallocated: int, pending: int, failure: int) -> dict:
    return {
        "date": date,
        "drive_id": drive_id,
        "drive_model": DRIVE_MODEL,
        "capacity_bytes": CAPACITY_BYTES,
        "failure": failure,
        "smart_5_raw": reallocated,
        "smart_187_raw": 0,
        "smart_188_raw": 0,
        "smart_197_raw": pending,
        "smart_198_raw": 0,
    }


def _healthy_drive_rows(drive_id: str) -> list[dict]:
    """Runs the full window with small, mostly-flat SMART noise and never
    fails - the negative-label population."""
    rows = []
    reallocated = 0
    date = START_DATE
    day = 0
    while date <= END_DATE:
        if day % 200 == 0 and day > 0:
            reallocated += 1  # occasional, harmless reallocation - never looks like failure
        rows.append(_row(drive_id, date, reallocated=reallocated, pending=0, failure=0))
        date += dt.timedelta(days=1)
        day += 1
    return rows


def _failing_drive_rows(drive_id: str, failure_date: dt.date, history_days: int) -> list[dict]:
    """Reallocated/pending-sector counts accelerate as `failure_date`
    approaches (mirroring the golden dataset's degrading trajectory), with
    `failure=1` on the last (failure) day."""
    rows = []
    start = failure_date - dt.timedelta(days=history_days - 1)
    for day in range(history_days):
        date = start + dt.timedelta(days=day)
        progress = day / history_days
        reallocated = int(progress * progress * 200)
        pending = int(progress * progress * 100)
        is_failure_day = date == failure_date
        rows.append(
            _row(
                drive_id,
                date,
                reallocated=reallocated,
                pending=pending,
                failure=1 if is_failure_day else 0,
            )
        )
    return rows


def build_synthetic_stub() -> pl.DataFrame:
    rows: list[dict] = []
    for i in range(N_HEALTHY_DRIVES):
        rows.extend(_healthy_drive_rows(f"synthetic-healthy-{i:03d}"))

    total_span_days = (END_DATE - START_DATE).days
    for i in range(1, N_FAILING_DRIVES + 1):
        # Evenly spread failure dates across the whole window, so every
        # configured chronological split (train/validation/test) ends up
        # with at least one real failure in it.
        offset_days = int(i / (N_FAILING_DRIVES + 1) * total_span_days)
        failure_date = START_DATE + dt.timedelta(days=offset_days)
        history_days = min(FAILING_HISTORY_DAYS_CAP, offset_days + 1)
        rows.extend(
            _failing_drive_rows(
                f"synthetic-failing-{i:03d}", failure_date, history_days=history_days
            )
        )

    return pl.DataFrame(rows)


def ingest_synthetic_stub(bronze_root: Path) -> list[Path]:
    lf = build_synthetic_stub().lazy()
    lf = with_ingestion_metadata(
        lf, source_dataset=SourceDataset.SYNTHETIC, source_file="synthetic_stub"
    )
    return sink_partitioned_by_month(lf, bronze_root=bronze_root)
