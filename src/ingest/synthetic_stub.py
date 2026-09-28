"""Synthetic placeholder dataset for exercising the full pipeline offline
- ingestion through training - without any real data download.

Unlike a trivial fixture, this generates real degrading-vs-healthy
trajectories (mirroring `tests/golden/generate_golden_dataset.py`, but in
raw Backblaze-shaped columns so it goes through the actual Bronze/Silver/
Gold pipeline, not a pre-computed gold-feature shape). The date range and
failure spread are derived directly from `configs/model.yaml`'s own
chronological split boundaries (`train_end`/`validation_end`/`test_end`)
rather than a second, independently hardcoded range - the two silently
drifting apart (leaving `train`/`validation` with zero observed-label
rows) is exactly the bug this module exists to avoid, and it already
happened once when the split boundaries were moved without updating this
file to match. The real trajectory-morphed chaos generator (noise
injection, timeline compression, correlated multi-drive failures) is
built in Phase 8.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.config import load_yaml
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata

N_HEALTHY_DRIVES = 6
N_FAILING_DRIVES = 9
FAILING_HISTORY_DAYS_CAP = 180  # pre-failure history length per failing drive

#: How far before `train_end` telemetry starts, and how far past `test_end`
#: it continues - enough lead time for feature maturity (min_history_days,
#: default 30) well before the first split boundary, and enough trailing
#: days past `test_end` that healthy drives near the end of the `test`
#: split aren't all trivially censored (a row can only get a real 0/1
#: label if the drive is observed at least `primary_horizon_days` past it).
LEAD_DAYS_BEFORE_TRAIN_END = 400
TRAIL_DAYS_AFTER_TEST_END_BUFFER = 20

CAPACITY_BYTES = 4_000_787_030_016
DRIVE_MODEL = "ST4000DM000"  # matches the "ST" prefix -> Seagate HDD family


def _split_boundaries_from_model_config() -> tuple[dt.date, dt.date, dt.date, dt.date, dt.date]:
    """Returns `(start_date, train_end, validation_end, test_end,
    end_date)`, derived from `configs/model.yaml`'s `splits`/
    `primary_horizon_days` so the stub's date range can never silently
    drift out of sync with the boundaries `pipelines/build_labels.py`
    actually splits on."""
    model_config = load_yaml("model.yaml")
    splits = model_config["splits"]
    train_end = dt.date.fromisoformat(splits["train_end"])
    validation_end = dt.date.fromisoformat(splits["validation_end"])
    test_end = dt.date.fromisoformat(splits["test_end"])
    horizon_days = model_config["primary_horizon_days"]

    start_date = train_end - dt.timedelta(days=LEAD_DAYS_BEFORE_TRAIN_END)
    trail_days = max(horizon_days, TRAIL_DAYS_AFTER_TEST_END_BUFFER)
    end_date = test_end + dt.timedelta(days=trail_days)
    return start_date, train_end, validation_end, test_end, end_date


def _failure_dates_covering_every_split(
    start_date: dt.date, train_end: dt.date, validation_end: dt.date, test_end: dt.date
) -> list[dt.date]:
    """Places `N_FAILING_DRIVES` failure dates so every split has at least
    one: most spread across `train` (by far the largest bucket, since
    `LEAD_DAYS_BEFORE_TRAIN_END` sets it), and (at least) one placed at
    the very end of `validation`/`test` each - a failure's pre-failure
    window (`FAILING_HISTORY_DAYS_CAP`/labeling's `horizon_days` days
    before it) mostly falls *before* it, so anchoring it at a bucket's
    last day is what keeps that window inside the bucket rather than
    spilling entirely into the previous one. A day-count-proportional
    spread across the whole range starves `validation`/`test` of any
    failure at all when they're much shorter than `train`, which is
    exactly what happened before this function existed."""
    validation_failures = 2
    test_failures = 2
    train_failures = max(1, N_FAILING_DRIVES - validation_failures - test_failures)

    dates: list[dt.date] = []
    train_span_days = (train_end - start_date).days
    for i in range(1, train_failures + 1):
        offset_days = int(i / (train_failures + 1) * train_span_days)
        dates.append(start_date + dt.timedelta(days=offset_days))

    for i in range(validation_failures):
        dates.append(validation_end - dt.timedelta(days=i))
    for i in range(test_failures):
        dates.append(test_end - dt.timedelta(days=i))

    return dates


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


def _healthy_drive_rows(drive_id: str, *, start_date: dt.date, end_date: dt.date) -> list[dict]:
    """Runs the full window with small, mostly-flat SMART noise and never
    fails - the negative-label population."""
    rows = []
    reallocated = 0
    date = start_date
    day = 0
    while date <= end_date:
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
    start_date, train_end, validation_end, test_end, end_date = (
        _split_boundaries_from_model_config()
    )

    rows: list[dict] = []
    for i in range(N_HEALTHY_DRIVES):
        rows.extend(
            _healthy_drive_rows(
                f"synthetic-healthy-{i:03d}", start_date=start_date, end_date=end_date
            )
        )

    failure_dates = _failure_dates_covering_every_split(
        start_date, train_end, validation_end, test_end
    )
    for i, failure_date in enumerate(failure_dates, start=1):
        offset_days = (failure_date - start_date).days
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
