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


def test_tier_figures_report_the_credited_gain_and_background_lift():
    from src.reporting.late_hit_figures import plot_later_failure_rate, plot_precision_by_tier

    rows = [
        {
            "tier": "warn",
            "alerts": 100,
            "false_alarms": 80,
            "precision_label": 0.2,
            "precision_credited": 0.3,
            "failed_later_share": 0.1,
        },
        {
            "tier": "drain",
            "alerts": 10,
            "false_alarms": 5,
            "precision_label": 0.5,
            "precision_credited": 0.6,
            "failed_later_share": 0.2,
        },
    ]
    fig = plot_precision_by_tier(rows, horizon_days=30)
    assert "drain rests on 10 alerts" in fig.texts[1].get_text()
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "20.0%" in labels and "30.0%" in labels
    fig = plot_later_failure_rate(rows, background=0.001, horizon_days=30)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert any("100x background" in t for t in labels)  # 0.1 / 0.001
    assert "0.10%" in fig.texts[1].get_text()


def test_profile_figure_draws_one_panel_per_attribute():
    from src.reporting.late_hit_figures import plot_alert_profile

    profile = {g: {"a": 1.0, "b": 2.0} for g in ("correct", "failed_later", "never_failed")}
    fig = plot_alert_profile(profile, attributes={"a": "A", "b": "B"}, counts={"correct": 3})
    assert len(fig.axes) == 2


def test_sequence_figures_render_from_report_shapes():
    from src.reporting.sequence_figures import (
        plot_blend_gain,
        plot_catch_overlap,
        plot_recall_at_budgets,
        plot_training_curves,
    )

    history = [
        {"epoch": 1, "validation_average_precision": 0.5},
        {"epoch": 2, "validation_average_precision": 0.4},
    ]
    fig = plot_training_curves({"s0": history, "s1": history})
    assert "(2)" in fig.axes[0].get_legend().get_texts()[0].get_text()
    budgets = ("0.0001", "0.001")
    rows = [
        {"name": n, "kind": k, "recall_at_far_0.0001": 0.04, "recall_at_far_0.001": 0.3}
        for n, k in (("LightGBM", "tree"), ("LSTM", "lstm"), ("Blend", "blend"))
    ]
    assert len(plot_recall_at_budgets(rows, budgets).axes[0].patches) == 6
    fig = plot_catch_overlap(
        [{"false_alarm_budget": 0.0001, "both": 17, "only_tree": 37, "only_ensemble": 62}]
    )
    assert "17 of LightGBM's 54" in fig.texts[1].get_text()
    gains = [
        {
            "name": "Ensemble",
            "gain": dict.fromkeys(budgets, 0.02),
            "ci": dict.fromkeys(budgets, [0.0, 0.04]),
        }
    ]
    assert len(plot_blend_gain(gains, budgets).axes) == 2
