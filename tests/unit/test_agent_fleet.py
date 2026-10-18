"""Binding the MAPE-K loop to the real feature table instead of two
hardcoded drives (`src/agent/fleet.py`).

The parts that need pinning are the ones that are easy to get wrong without
noticing: that the checkpointed state stays small, that the window filter
actually reaches the scan, that a snapshot overlapping the test split is
recorded, and that scoring the test window is distinguishable from scoring
the inference window.
"""

import datetime as dt

import polars as pl
import pytest

from src.agent.fleet import (
    load_fleet_snapshot,
    overlaps_test_split,
    score_snapshot,
    to_candidate_drives,
)

SPLITS = {"train_end": "2026-02-14", "validation_end": "2026-03-31", "test_end": "2026-05-31"}
THRESHOLDS = {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}


def _gold(tmp_path, n_drives=5, days=(dt.date(2026, 5, 20), dt.date(2026, 6, 15))):
    """A miniature gold feature table: each drive seen on each date, with the
    later date carrying the higher score so "latest row" is checkable."""
    rows = []
    for i in range(n_drives):
        for day_index, day in enumerate(days):
            rows.append(
                {
                    "drive_id": f"d{i:03d}",
                    "date": day,
                    "telemetry_coverage_30d": 1.0,
                    "hours_since_last_telemetry": 24.0,
                    "attribute_coverage_factor": 1.0,
                    "feature_confidence": 0.9,
                    "stale_telemetry_flag": False,
                    "smart_5_raw": float(i + day_index * 100),
                }
            )
    features = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(features)
    metadata = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {
            "drive_id": [f"d{i:03d}" for i in range(n_drives)],
            "feature_maturity": ["MATURE"] * n_drives,
        }
    ).write_parquet(metadata)
    return features, metadata


def test_the_snapshot_is_one_row_per_drive(tmp_path):
    features, metadata = _gold(tmp_path)
    snapshot = load_fleet_snapshot(features, metadata)
    assert snapshot.height == 5
    assert snapshot["drive_id"].n_unique() == 5
    assert "feature_maturity" in snapshot.columns


def test_the_window_filter_selects_the_right_latest_row(tmp_path):
    """Restricting to the test window must give that window's last row, not
    the drive's most recent row overall - otherwise a backtest silently
    scores June data."""
    features, metadata = _gold(tmp_path)
    test_window = load_fleet_snapshot(
        features, metadata, start_date=dt.date(2026, 4, 1), end_date=dt.date(2026, 5, 31)
    )
    assert test_window["date"].unique().to_list() == [dt.date(2026, 5, 20)]

    inference_window = load_fleet_snapshot(features, metadata, start_date=dt.date(2026, 6, 1))
    assert inference_window["date"].unique().to_list() == [dt.date(2026, 6, 15)]


def test_an_empty_window_yields_an_empty_snapshot(tmp_path):
    features, metadata = _gold(tmp_path)
    snapshot = load_fleet_snapshot(
        features, metadata, start_date=dt.date(2027, 1, 1), end_date=dt.date(2027, 1, 2)
    )
    assert snapshot.height == 0


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (dt.date(2026, 6, 1), dt.date(2026, 6, 30), False),  # inference window
        (dt.date(2026, 4, 1), dt.date(2026, 5, 31), True),  # the test split
        (dt.date(2026, 5, 31), dt.date(2026, 6, 30), True),  # straddles test_end
        (dt.date(2026, 1, 1), dt.date(2026, 3, 31), False),  # train + validation
        (None, None, True),  # unbounded cannot be shown not to overlap
    ],
)
def test_test_split_overlap_is_detected(start, end, expected):
    assert overlaps_test_split(start, end, SPLITS) is expected


class _Model:
    """Scores from the row's largest value, which the fixture makes
    `smart_5_raw` (100 + drive index). Taking the max rather than a fixed
    column keeps the test independent of whatever order
    `select_feature_columns` returns. Spreads scores across 0.0-0.98 so they
    straddle the 0.30 warn threshold."""

    def predict_proba(self, x):
        import numpy as np

        p = np.clip((x.max(axis=1) - 100.0) / 50.0, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def test_candidates_are_capped_so_the_checkpoint_stays_small(tmp_path):
    """AgentState is written to SQLite every cycle, so the fleet must not
    enter it. Only the top-scoring drive is ever acted on."""
    features, metadata = _gold(tmp_path, n_drives=50)
    snapshot = load_fleet_snapshot(features, metadata)
    scored = score_snapshot(snapshot, _Model())
    drives = to_candidate_drives(scored, action_thresholds=THRESHOLDS, max_candidates=10)
    assert len(drives) == 10
    assert drives == sorted(drives, key=lambda d: d["p_fail"], reverse=True)


def test_candidates_below_the_lowest_threshold_are_dropped(tmp_path):
    """Below the warn threshold the policy returns MONITOR anyway, so
    carrying the drive costs checkpoint space for no decision."""
    features, metadata = _gold(tmp_path, n_drives=50)
    scored = score_snapshot(load_fleet_snapshot(features, metadata), _Model())
    drives = to_candidate_drives(scored, action_thresholds=THRESHOLDS, max_candidates=1000)
    assert drives, "fixture should produce some candidates"
    assert all(d["p_fail"] >= min(THRESHOLDS.values()) for d in drives)
    assert len(drives) < scored.height


def test_a_candidate_carries_exactly_the_fields_the_policy_needs(tmp_path):
    features, metadata = _gold(tmp_path, n_drives=20)
    scored = score_snapshot(load_fleet_snapshot(features, metadata), _Model())
    drive = to_candidate_drives(scored, action_thresholds=THRESHOLDS)[0]
    assert set(drive) == {
        "drive_id",
        "p_fail",
        "feature_confidence",
        "feature_maturity",
        "stale_telemetry",
    }
    assert isinstance(drive["stale_telemetry"], bool)
    assert isinstance(drive["p_fail"], float)
