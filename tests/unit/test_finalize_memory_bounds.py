"""The unbatched steps at the end of `build_silver`'s finalize stage.

Bronze profiling already tripped the 20 GB cap at 365 days by aggregating a
whole table at once (`src/ingest/profiling.py`). Two steps in
`pipelines/build_silver.py::_stage_finalize` have the same shape - they run
over the *combined* 123M-row drive_day table rather than per batch:

  * `check_no_duplicate_drive_day_attribute` takes `unique()` over
    (drive_id, date), a hash set of every row's key.
  * `build_drive_metadata` groups by drive_id, which hashes every drive-day
    even though it returns one row per drive.

Both now collect with the streaming engine so they spill to disk. The
streaming engine is a different execution path, and this repo has one
recorded case of it misbehaving (src/models/features.py), so these tests pin
that it returns the SAME answers as the in-memory engine rather than assuming
it does.
"""

import datetime as dt

import polars as pl

from src.preprocess.feature_maturity import build_drive_metadata
from src.preprocess.quality_checks import (
    check_no_duplicate_drive_day_attribute,
    run_all_checks,
)


def _drive_days(n_drives: int = 40, n_days: int = 15, *, duplicates: int = 0) -> pl.DataFrame:
    """A drive-day table spanning several drives and dates, optionally with
    exact duplicate rows appended so the duplicate count is a known number."""
    start = dt.date(2025, 7, 1)
    rows = {
        "drive_id": [f"d{i:03d}" for i in range(n_drives) for _ in range(n_days)],
        "date": [start + dt.timedelta(days=d) for _ in range(n_drives) for d in range(n_days)],
        "model_family": ["ST" for _ in range(n_drives * n_days)],
        "capacity_gb": [16000.0 for _ in range(n_drives * n_days)],
        "stale_telemetry_flag": [False for _ in range(n_drives * n_days)],
        "failure_date": [None for _ in range(n_drives * n_days)],
    }
    frame = pl.DataFrame(rows, schema_overrides={"failure_date": pl.Date})
    if duplicates:
        frame = pl.concat([frame, frame.head(duplicates)])
    return frame


def test_duplicate_check_counts_the_same_streamed_as_in_memory():
    frame = _drive_days(duplicates=7)
    streamed = check_no_duplicate_drive_day_attribute(frame.lazy())

    # The in-memory reference, computed here rather than taken on trust.
    keys = ["drive_id", "date"]
    expected = frame.height - frame.select(keys).unique().height
    assert expected == 7
    assert streamed["violation_count"] == expected
    assert streamed["passed"] is False


def test_duplicate_check_passes_a_clean_table():
    result = check_no_duplicate_drive_day_attribute(_drive_days().lazy())
    assert result["violation_count"] == 0
    assert result["passed"] is True


def test_duplicate_check_uses_the_attribute_key_when_present():
    """On the canonical long table the key includes smart_attribute_name, so
    one drive-day legitimately has several rows and must not count as
    duplicated."""
    long = pl.DataFrame(
        {
            "drive_id": ["a", "a", "a"],
            "date": [dt.date(2025, 7, 1)] * 3,
            "smart_attribute_name": ["smart_5_raw", "smart_197_raw", "smart_198_raw"],
        }
    )
    assert check_no_duplicate_drive_day_attribute(long.lazy())["violation_count"] == 0


def test_drive_metadata_is_identical_streamed_and_in_memory():
    """One row per drive either way, with the same values - a streamed
    group_by that reordered or dropped a drive would silently change every
    downstream feature-maturity decision."""
    frame = _drive_days()
    streamed = build_drive_metadata(frame.lazy(), min_history_days=10)

    # Rebuild the same aggregation with the in-memory engine as the reference.
    reference = (
        frame.lazy()
        .group_by("drive_id")
        .agg(
            pl.col("date").min().alias("first_seen_date"),
            pl.col("date").max().alias("last_seen_date"),
            pl.col("date").n_unique().alias("total_observed_days"),
        )
        .collect()
        .sort("drive_id")
    )

    assert streamed.height == reference.height == 40
    got = streamed.select(["drive_id", "first_seen_date", "last_seen_date", "total_observed_days"])
    assert got.sort("drive_id").equals(reference.sort("drive_id"))


def test_drive_metadata_still_classifies_maturity():
    """The streamed collect must not disturb the derived columns."""
    metadata = build_drive_metadata(_drive_days(n_days=15).lazy(), min_history_days=10)
    assert set(metadata["feature_maturity"].unique()) == {"MATURE"}
    assert metadata["survivorship_valid"].all()

    warmup = build_drive_metadata(_drive_days(n_days=5).lazy(), min_history_days=10)
    assert set(warmup["feature_maturity"].unique()) == {"WARMUP"}
    assert not warmup["survivorship_valid"].any()


def test_run_all_checks_still_returns_every_check():
    results = run_all_checks(_drive_days().lazy())
    assert [r["check"] for r in results] == [
        "no_null_dates",
        "no_duplicate_drive_day_attribute",
        "capacity_positive",
        "failure_date_after_date",
    ]
    assert all(r["passed"] for r in results)
