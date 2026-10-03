import datetime as dt

import numpy as np
import pytest

from src.models.alerting import analyze_false_alarms, smooth_scores


def _dates(n, start=dt.date(2026, 3, 1)):
    return np.array([start + dt.timedelta(days=i) for i in range(n)])


def test_raw_is_the_identity():
    scores = np.array([0.1, 0.9, 0.2])
    out = smooth_scores(np.array(["a"] * 3), _dates(3), scores, "raw")
    assert out.tolist() == scores.tolist()


def test_min3_suppresses_a_single_spike_but_keeps_sustained_scores():
    drives = np.array(["spike"] * 5 + ["steady"] * 5)
    dates = np.concatenate([_dates(5), _dates(5)])
    scores = np.array([0.1, 0.1, 0.9, 0.1, 0.1] + [0.8, 0.8, 0.8, 0.8, 0.8])
    out = smooth_scores(drives, dates, scores, "min3")
    assert out[:5].max() <= 0.1 + 1e-12  # the lone 0.9 never survives min over 3
    assert out[5:7].tolist() == [0.0, 0.0]  # no full window yet
    assert out[7:].tolist() == [0.8, 0.8, 0.8]


def test_median3_needs_two_of_three_high():
    drives = np.array(["a"] * 5)
    scores = np.array([0.9, 0.1, 0.9, 0.1, 0.1])
    out = smooth_scores(drives, _dates(5), scores, "median3")
    assert out.tolist() == [0.0, 0.0, 0.9, 0.1, 0.1]


def test_smoothing_is_causal_per_drive_and_keeps_input_row_order():
    # rows deliberately interleaved and out of date order
    drives = np.array(["a", "b", "a", "b", "a", "b"])
    dates = np.array(
        [
            dt.date(2026, 3, 3),
            dt.date(2026, 3, 1),
            dt.date(2026, 3, 1),
            dt.date(2026, 3, 3),
            dt.date(2026, 3, 2),
            dt.date(2026, 3, 2),
        ]
    )
    scores = np.array([0.6, 0.2, 0.2, 0.9, 0.4, 0.4])
    out = smooth_scores(drives, dates, scores, "mean3")
    # drive a by date: .2 (3/1), .4 (3/2), .6 (3/3) -> means .2, .3, .4
    assert out[2] == pytest.approx(0.2)
    assert out[4] == pytest.approx(0.3)
    assert out[0] == pytest.approx(0.4)
    # drive b by date: .2, .4, .9 -> means .2, .3, .5
    assert out[1] == pytest.approx(0.2)
    assert out[5] == pytest.approx(0.3)
    assert out[3] == pytest.approx(0.5)


def test_analyze_false_alarms_classifies_alerting_healthy_drives():
    rows = []  # (drive, day, y, score, event_type, days_to_event)
    for day in range(3):
        rows.append(("F", day, 1, 0.9, "confirmed_failure", 3 - day))  # a real failure: excluded
        rows.append(("late", day, 0, 0.8 if day == 1 else 0.1, "confirmed_failure", 40 - day))
        rows.append(("pulled", day, 0, 0.9, "preventive_replacement", None))
        rows.append(("ok", day, 0, 0.1, "still_active", None))
        rows.append(("ok2", day, 0, 0.9, None, None))  # missing event_type -> still_active
    d = _dates(3)
    result = analyze_false_alarms(
        np.array([r[0] for r in rows]),
        np.array([d[r[1]] for r in rows]),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]),
        0.5,
        np.array([r[4] for r in rows], dtype=object),
        np.array([r[5] for r in rows], dtype=float),
    )
    assert result["healthy_drives"] == 4  # F is failing -> excluded
    assert result["false_alarm_drives"] == 3  # late, pulled, ok2
    assert result["alerted_by_event_type"] == {
        "confirmed_failure": 1,
        "preventive_replacement": 1,
        "still_active": 1,
    }
    assert result["failed_later_days_after_alert"] == {"31-60d": 1}  # 40 - day(1) = 39
    # 'late' is 1 of 3 alerts but 1 of 4 healthy drives -> lift (1/3)/(1/4)
    assert result["lift_by_event_type"]["confirmed_failure"] == pytest.approx(4 / 3)
