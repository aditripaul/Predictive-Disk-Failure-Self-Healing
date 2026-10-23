import datetime as dt

import matplotlib
import polars as pl

matplotlib.use("Agg")

from src.agent.false_positives import OUTCOME_ORDER, alert_outcomes
from src.reporting.late_hit_figures import (
    BLUE,
    ORANGE,
    plot_alert_outcomes,
    plot_alert_timelines,
    plot_lead_time_histogram,
)

T = dt.date(2026, 4, 5)


def _days(n):
    return T + dt.timedelta(days=n)


def _frame(rows):
    return pl.DataFrame(rows, schema=["drive_id", "date", "event_date", "event_type"], orient="row")


def test_outcomes_split_at_the_horizon_and_after_it():
    flagged = pl.DataFrame(
        {"drive_id": list("abcdefgh"), "date": [T] * 8, "label": [1, 1, 0, 0, 0, 0, 0, 0]}
    )
    events = _frame(
        [
            ("a", T, _days(1), "confirmed_failure"),
            ("b", T, _days(30), "confirmed_failure"),  # day 30 is still inside the label window
            ("c", T, _days(31), "confirmed_failure"),  # day 31 is the first early alert
            ("d", T, _days(60), "failure_followed_by_replacement"),
            ("e", T, _days(61), "confirmed_failure"),
            ("f", T, _days(40), "removed_without_failure"),
            ("g", T, None, "still_active"),
            ("h", T, _days(45), "preventive_replacement"),
        ]
    )
    out = alert_outcomes(flagged, events, horizon_days=30).sort("drive_id")
    assert out["outcome"].to_list() == [
        "correct",
        "correct",
        "early",
        "early",
        "later",
        "removed",
        "active",
        "removed",
    ]
    assert out["days_to_failure"].to_list() == [1, 30, 31, 60, 61, None, None, None]
    assert set(out["outcome"]) <= set(OUTCOME_ORDER)


def test_drive_without_an_event_row_counts_as_still_in_service():
    flagged = pl.DataFrame({"drive_id": ["z"], "date": [T], "label": [0]})
    out = alert_outcomes(
        flagged, _frame([("a", T, _days(5), "confirmed_failure")]), horizon_days=30
    )
    assert out["outcome"].to_list() == ["active"]


def test_histogram_colours_day_30_blue_and_day_31_orange():
    fig = plot_lead_time_histogram([30, 30, 31, 45, 70], horizon_days=30)
    ax = fig.axes[0]
    by_x = {round(p.get_x() + p.get_width() / 2): p for p in ax.patches if p.get_height() > 0}
    assert matplotlib.colors.to_hex(by_x[30].get_facecolor()) == BLUE
    assert matplotlib.colors.to_hex(by_x[31].get_facecolor()) == ORANGE
    assert any(round(p.get_height()) == 1 for p in ax.patches)  # the single ">60" alert
    assert "5 alerts" in fig.texts[1].get_text()
    assert "2 fell inside" in fig.texts[1].get_text()


def test_outcome_figure_reports_the_early_share():
    counts = {"correct": 60, "early": 20, "later": 5, "removed": 5, "active": 110}
    fig = plot_alert_outcomes(counts, horizon_days=30, tier="cordon", snapshots=4)
    title = fig.texts[0].get_text()
    assert "25 of the 140" in title  # early + later of the 30-day false alarms
    assert len(fig.axes[0].patches) == 5


def test_timelines_draw_one_row_per_alert():
    rows = [{"label": f"d{i}", "lead_days": 31 + i} for i in range(6)]
    fig = plot_alert_timelines(rows, horizon_days=30, total_early=40)
    assert "6 of 40" in fig.texts[1].get_text()
    assert len(fig.axes[0].get_yticklabels()) == 6


def test_figures_survive_empty_input():
    plot_lead_time_histogram([], horizon_days=30)
    plot_alert_outcomes({}, horizon_days=30, tier="cordon", snapshots=0)
    plot_alert_timelines([], horizon_days=30, total_early=0)
