"""The look-ahead diagnostic, checked against a fleet small enough to do by hand.

`scripts/check_agent_lookahead.py` is the instrument that said the Apr-May
agent backtest was not point-in-time, so it has to be right. Two of its
measurements went wrong once already: lead time was read off the label at the
scored row, but a drive's final row - the failure day - has a NULL label
(`event_date > date` in src/labels/labeling.py), so the filter dropped exactly
the drives caught on their last day and reported only the easy 15%.
"""

import datetime as dt
import importlib.util
from pathlib import Path

import polars as pl
import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_agent_lookahead", Path("scripts/check_agent_lookahead.py")
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)

END = dt.date(2026, 5, 31)
EARLIER = dt.date(2026, 5, 10)
THRESHOLDS = {"warn": 0.1, "cordon": 0.3, "migrate": 0.6, "drain": 0.9}


def _report() -> dict:
    def decision(drive_id: str, p_fail: float, as_of: dt.date) -> dict:
        return {
            "drive_id": drive_id,
            "proposed_action": "cordon",
            "p_fail": p_fail,
            "as_of": str(as_of),
        }

    return {
        "action_thresholds": THRESHOLDS,
        "candidate_count": 5,
        "above_threshold_count": 5,
        "decisions": [
            decision("A", 0.95, END),  # alive at T, fails in June
            decision("B", 0.70, END),  # alive at T, healthy
            decision("C", 0.40, END),  # alive at T, fails in June
            decision("D", 0.20, END),  # alive at T, healthy
            decision("F", 0.99, EARLIER),  # died mid-window: must NOT count
        ],
    }


def _labels() -> pl.DataFrame:
    """Label rows. E is a failing drive the model never flagged (below warn),
    so it belongs in the recall denominator but not in the pool. F has no row
    on the final day because it was already gone, but its scored row DOES
    carry a label - a drive whose telemetry gapped and returned later - so
    only the end-date restriction, not a null label, keeps it out of the
    clean subset. Without that, the restriction is untested."""
    rows = [
        ("A", END, 1),
        ("B", END, 0),
        ("C", END, 1),
        ("D", END, 0),
        ("E", END, 1),
        ("F", EARLIER, 1),
    ]
    return pl.DataFrame(
        {
            "drive_id": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "days_to_event": [10] * len(rows),
        },
        schema_overrides={"label": pl.Int8},
    )


def _metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": ["A", "B", "C", "D", "F"],
            "feature_maturity": [
                "DECOMMISSIONED",
                "MATURE",
                "DECOMMISSIONED",
                "MATURE",
                "DECOMMISSIONED",
            ],
            "last_seen_date": [
                dt.date(2026, 6, 20),
                dt.date(2026, 6, 30),
                dt.date(2026, 6, 5),
                dt.date(2026, 6, 30),
                EARLIER,
            ],
        }
    )


def test_the_clean_subset_excludes_drives_that_died_mid_window(capsys):
    report, labels = _report(), _labels()
    frame = script._frame(report, _metadata(), labels).with_columns(
        pl.col("drive_id").is_in(["A", "C", "F"]).alias("failed")
    )
    script._clean_subset(report, frame, labels, END)
    out = capsys.readouterr().out

    # 5 drives alive and labelled on the final day; F is not one of them.
    assert "5 drives alive and labelled, 3 failed" in out
    assert "base rate 60.000%" in out
    flagged = {line.split()[1]: line for line in out.splitlines() if line.startswith("   >= ")}
    # >= warn flags A, B, C, D (4) - not F, which would make it 5.
    assert flagged["warn"].split()[2:4] == ["4", "2"]
    # recall is against every drive alive at T, including the unflagged E.
    assert "66.7%" in flagged["warn"]
    assert flagged["drain"].split()[2:4] == ["1", "1"]
    assert "33.3%" in flagged["drain"]


def test_precision_in_the_clean_subset_ignores_the_maturity_gate(capsys):
    """A and C are DECOMMISSIONED only because they vanish in June - the
    outcome. Tiers come from p_fail alone, so that must not change them."""
    report, labels = _report(), _labels()
    frame = script._frame(report, _metadata(), labels).with_columns(
        pl.col("drive_id").is_in(["A", "C", "F"]).alias("failed")
    )
    script._clean_subset(report, frame, labels, END)
    out = capsys.readouterr().out
    cordon = next(line for line in out.splitlines() if line.startswith("   >= cordon"))
    # A, B, C flagged at >= 0.3; A and C failed -> 2 of 3.
    assert cordon.split()[2:4] == ["3", "2"]
    assert "66.7%" in cordon


def test_lead_time_counts_the_drive_caught_on_its_last_day():
    """The defect: a drive scored on its final row has a null label, so a
    label-based lead time dropped it. Measured to last_seen_date it is 0."""
    caught = pl.DataFrame(
        {
            "as_of": [dt.date(2026, 5, 10), dt.date(2026, 5, 20), dt.date(2026, 5, 31)],
            "last_seen_date": [dt.date(2026, 5, 10), dt.date(2026, 5, 27), dt.date(2026, 6, 15)],
        }
    )
    assert script.lead_days(caught).to_list() == [0, 7, 15]


def test_a_scored_row_on_the_failure_day_really_has_no_label():
    """Pins WHY the old lead-time filter was wrong, against the real label
    definition rather than an assumption about it."""
    from src.labels.event_types import CONFIRMED_FAILURE
    from src.labels.labeling import compute_labels_for_horizon

    fail_day = dt.date(2026, 5, 10)
    drive_days = pl.DataFrame(
        {"drive_id": ["d"] * 3, "date": [dt.date(2026, 5, 8), dt.date(2026, 5, 9), fail_day]}
    )
    metadata = pl.DataFrame(
        {
            "drive_id": ["d"],
            "event_date": [fail_day],
            "event_type": [CONFIRMED_FAILURE],
            "last_seen_date": [fail_day],
        }
    )
    labels = compute_labels_for_horizon(drive_days, metadata, horizon_days=30).sort("date")
    assert labels["label"].to_list() == [1, 1, None]
    assert labels["days_to_event"].to_list() == [2, 1, 0]


def test_mean_of_an_empty_group_reads_zero_not_an_error():
    assert script._mean(pl.Series([], dtype=pl.Boolean)) == pytest.approx(0.0)
    assert script._median(pl.Series([], dtype=pl.Int64)) == pytest.approx(0.0)
