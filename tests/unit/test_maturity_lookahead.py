"""`feature_maturity` is a function of the dataset's LAST date.

`build_drive_metadata` marks a drive DECOMMISSIONED when its final row is
before the end of the data, and checks that before every other state. For
scoring a fleet *now* that is right: a drive that has stopped reporting is
gone. For a backtest over a past window it is look-ahead, because "not
DECOMMISSIONED" then means "was still reporting after the snapshot" - and a
drive that reports on the last day cannot have failed inside that window.

`determine_action_tier` gates migrate/drain on MATURE. In an Apr-May
backtest of data ending June 30, that gate therefore reads the outcome: every
drive it lets through survived, every drive it downgrades disappeared. The
agent run produced exactly that - migrate and drain 0/71 correct, the drives
downgraded to cordon 90% correct - the reverse of what a trustworthy
evidence gate would show.

These tests pin the mechanism, and `as_of_date`, the parameter that makes a
point-in-time answer possible.
"""

import datetime as dt

import polars as pl

from src.preprocess.feature_maturity import build_drive_metadata

DATA_END = dt.date(2026, 6, 30)


def _days(drive_id: str, first: dt.date, last: dt.date) -> list[dict]:
    rows, day = [], first
    while day <= last:
        rows.append(
            {
                "drive_id": drive_id,
                "date": day,
                "model_family": "ST",
                "capacity_gb": 16000.0,
                "stale_telemetry_flag": False,
            }
        )
        day += dt.timedelta(days=1)
    return rows


def _fleet() -> pl.DataFrame:
    return pl.DataFrame(
        _days("survivor", dt.date(2026, 1, 1), DATA_END)
        # Stops reporting 2026-05-10, i.e. fails or is retired mid-window.
        + _days("vanished", dt.date(2026, 1, 1), dt.date(2026, 5, 10))
    )


def _maturity(metadata: pl.DataFrame, drive_id: str) -> str:
    return metadata.filter(pl.col("drive_id") == drive_id)["feature_maturity"][0]


def test_a_drive_that_stops_reporting_before_the_data_ends_is_decommissioned():
    metadata = build_drive_metadata(_fleet(), min_history_days=30)
    assert _maturity(metadata, "survivor") == "MATURE"
    assert _maturity(metadata, "vanished") == "DECOMMISSIONED"


def test_the_decommissioned_check_outranks_every_other_state():
    """A drive with 130 observed days would otherwise be MATURE. Because the
    gate is `maturity == MATURE`, being gone by data end is enough to fail it
    - whether or not the drive was healthy when scored."""
    metadata = build_drive_metadata(_fleet(), min_history_days=30)
    vanished = metadata.filter(pl.col("drive_id") == "vanished")
    assert vanished["total_observed_days"][0] >= 30
    assert vanished["feature_maturity"][0] == "DECOMMISSIONED"


def test_maturity_at_a_past_snapshot_depends_on_what_happens_afterwards():
    """The look-ahead itself. Judged on 2026-05-01 - when 'vanished' was
    still reporting - the same drive is MATURE; judged against the end of the
    data it is DECOMMISSIONED. Nothing about the drive changed."""
    snapshot = dt.date(2026, 5, 1)
    fleet = _fleet()

    as_of_data_end = build_drive_metadata(fleet, min_history_days=30)
    point_in_time = build_drive_metadata(
        fleet.filter(pl.col("date") <= snapshot), min_history_days=30, as_of_date=snapshot
    )

    assert _maturity(as_of_data_end, "vanished") == "DECOMMISSIONED"
    assert _maturity(point_in_time, "vanished") == "MATURE"
    assert _maturity(point_in_time, "survivor") == "MATURE"


def test_a_drive_is_decommissioned_only_once_it_has_actually_gone():
    """Point-in-time, 'vanished' becomes DECOMMISSIONED the day after its
    last row - which is the information a live system would have."""
    after = dt.date(2026, 5, 11)
    metadata = build_drive_metadata(
        _fleet().filter(pl.col("date") <= after), min_history_days=30, as_of_date=after
    )
    assert _maturity(metadata, "vanished") == "DECOMMISSIONED"
    assert _maturity(metadata, "survivor") == "MATURE"


def test_mature_at_data_end_means_no_failure_inside_the_window_horizon():
    """The consequence for a backtest, stated as the invariant it breaks: a
    drive that reports on the final day cannot have failed during the 30
    days before it - so gating on MATURE selects negatives."""
    metadata = build_drive_metadata(_fleet(), min_history_days=30)
    mature = metadata.filter(pl.col("feature_maturity") == "MATURE")
    assert (mature["last_seen_date"] == DATA_END).all()
