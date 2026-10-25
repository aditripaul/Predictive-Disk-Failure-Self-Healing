"""Measuring the policy at a date, with intervals that respect repeated drives.

These tests are about the properties that make the numbers trustworthy: a
snapshot contains only drives reporting that day, tier arithmetic matches a
hand count, the bootstrap resamples drives rather than observations, and the
one-look versus two-look comparison is paired and matched on false-alarm rate.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.agent.point_in_time import (
    TierCounts,
    _prepare_far,
    bootstrap_tier_ci,
    compare_looks,
    last_observable_date,
    pair_observations,
    recall_at_far,
    select_snapshot_dates,
    snapshot_observations,
    tier_counts,
    tier_metrics,
)

THRESHOLDS = {"warn": 0.1, "cordon": 0.3, "migrate": 0.6, "drain": 0.9}


def _obs(rows: list[tuple[str, float, int]], day: dt.date = dt.date(2026, 5, 31)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "drive_id": [r[0] for r in rows],
            "date": [day] * len(rows),
            "p_fail": [r[1] for r in rows],
            "label": [r[2] for r in rows],
        },
        schema_overrides={"label": pl.Int8},
    )


# --- dates -----------------------------------------------------------------


def test_snapshot_dates_are_anchored_on_the_end_and_ascending():
    dates = select_snapshot_dates(dt.date(2026, 4, 1), dt.date(2026, 5, 31), 7)
    assert dates[-1] == dt.date(2026, 5, 31)
    assert dates == sorted(dates)
    assert all((b - a).days == 7 for a, b in zip(dates, dates[1:], strict=False))
    assert dates[0] >= dt.date(2026, 4, 1)


def test_a_step_that_never_hits_the_start_simply_stops_short_of_it():
    dates = select_snapshot_dates(dt.date(2026, 4, 1), dt.date(2026, 4, 10), 4)
    assert dates == [dt.date(2026, 4, 2), dt.date(2026, 4, 6), dt.date(2026, 4, 10)]


def test_a_nonpositive_step_is_rejected_rather_than_looping_forever():
    with pytest.raises(ValueError):
        select_snapshot_dates(dt.date(2026, 4, 1), dt.date(2026, 4, 10), 0)


def test_the_last_observable_date_is_one_horizon_before_the_data_ends():
    """After it the label is censored, so rows silently drop out of every count."""
    assert last_observable_date(dt.date(2026, 6, 30), 30) == dt.date(2026, 5, 31)
    assert last_observable_date(dt.date(2026, 6, 30), 7) == dt.date(2026, 6, 23)


# --- snapshot construction ---------------------------------------------------


class _Model:
    """Scores by the row's largest feature, so scores are predictable."""

    def predict_proba(self, x):
        p = np.clip(x.max(axis=1) / 100.0, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def _write_gold(tmp_path: Path) -> tuple[Path, Path]:
    d1, d2 = dt.date(2026, 5, 30), dt.date(2026, 5, 31)
    features = pl.DataFrame(
        {
            "drive_id": ["alive", "alive", "gone", "gone", "new"],
            "date": [d1, d2, d1, d1, d2],
            "smart_5_raw": [10.0, 80.0, 99.0, 99.0, 30.0],
        }
    )
    # "gone" has no row on d2: it left the fleet, so it must not be in d2.
    features = features.filter(~((pl.col("drive_id") == "gone") & (pl.col("date") == d2)))
    features_path = tmp_path / "features.parquet"
    features.write_parquet(features_path)

    labels_path = tmp_path / "labels.parquet"
    pl.DataFrame(
        {
            "drive_id": ["alive", "alive", "gone", "new", "new"],
            "date": [d1, d2, d1, d2, d2],
            "horizon_days": [30, 30, 30, 30, 7],
            "label": [0, 1, 1, None, 1],
        },
        schema_overrides={"label": pl.Int8},
    ).write_parquet(labels_path)
    return features_path, labels_path


def test_a_snapshot_contains_only_drives_reporting_that_day(tmp_path):
    """A fleet at T does not contain drives that already died."""
    features, labels = _write_gold(tmp_path)
    obs = snapshot_observations(features, labels, _Model(), dt.date(2026, 5, 31), horizon_days=30)
    assert "gone" not in obs["drive_id"].to_list()
    assert obs["date"].unique().to_list() == [dt.date(2026, 5, 31)]


def test_a_drive_without_an_observable_label_is_dropped_not_counted_negative(tmp_path):
    features, labels = _write_gold(tmp_path)
    obs = snapshot_observations(features, labels, _Model(), dt.date(2026, 5, 31), horizon_days=30)
    # "new" is alive on the 31st but its 30-day label is null.
    assert obs["drive_id"].to_list() == ["alive"]


def test_the_label_is_the_one_at_the_snapshot_row_and_horizon(tmp_path):
    features, labels = _write_gold(tmp_path)
    obs = snapshot_observations(features, labels, _Model(), dt.date(2026, 5, 31), horizon_days=30)
    row = obs.row(0, named=True)
    assert row["label"] == 1  # the 31st, not the 30th (0), and not horizon 7
    assert row["p_fail"] == pytest.approx(0.8)


def test_a_date_with_no_rows_is_an_empty_typed_frame(tmp_path):
    features, labels = _write_gold(tmp_path)
    obs = snapshot_observations(features, labels, _Model(), dt.date(2027, 1, 1), horizon_days=30)
    assert obs.is_empty()
    assert set(obs.columns) == {"drive_id", "date", "p_fail", "label"}


# --- tier arithmetic ---------------------------------------------------------


def test_tier_metrics_match_a_hand_count():
    # 8 drives: 3 fail. p: A.95 B.7 C.4 D.2 E.05 | F.8 G.15 H.01
    obs = _obs(
        [
            ("A", 0.95, 1),
            ("B", 0.70, 0),
            ("C", 0.40, 1),
            ("D", 0.20, 0),
            ("E", 0.05, 1),
            ("F", 0.80, 0),
            ("G", 0.15, 0),
            ("H", 0.01, 0),
        ]
    )
    metrics = tier_metrics(tier_counts(obs, THRESHOLDS))
    # >= cordon (0.3): A B C F flagged; A and C failed -> 2/4, recall 2/3.
    cordon = metrics["cordon"]
    assert (cordon["flagged"], cordon["caught"]) == (4, 2)
    assert cordon["precision"] == pytest.approx(0.5)
    assert cordon["recall"] == pytest.approx(2 / 3)
    # FAR: 2 false alarms of 5 healthy drives.
    assert cordon["false_alarm_rate"] == pytest.approx(2 / 5)
    assert cordon["base_rate"] == pytest.approx(3 / 8)
    assert cordon["lift"] == pytest.approx(0.5 / (3 / 8))
    # >= drain (0.9): only A, which failed.
    assert metrics["drain"]["precision"] == pytest.approx(1.0)
    assert metrics["drain"]["recall"] == pytest.approx(1 / 3)


def test_weights_act_as_drive_multiplicities():
    obs = _obs([("A", 0.95, 1), ("B", 0.70, 0)])
    counts = tier_counts(obs, THRESHOLDS)
    order = obs["drive_id"].to_list()
    weights = np.array([3.0 if d == "A" else 1.0 for d in order])
    metrics = tier_metrics(counts, weights)["migrate"]
    assert metrics["caught"] == 3 and metrics["flagged"] == 4


# --- the cluster bootstrap -----------------------------------------------------


def _repeated_fleet(n_drives: int, repeats: int, seed: int = 0) -> pl.DataFrame:
    """Each drive appears `repeats` times with the SAME outcome and a similar
    score, as consecutive snapshots of one drive do."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_drives):
        fails = rng.random() < 0.15
        base = rng.beta(5, 2) if fails else rng.beta(2, 5)
        for _ in range(repeats):
            rows.append((f"d{i}", float(np.clip(base + rng.normal(0, 0.02), 0, 1)), int(fails)))
    return _obs(rows)


def test_clustering_widens_the_interval_when_a_drive_repeats():
    """The reason to resample drives. Treating each snapshot as independent
    counts one drive four times as four pieces of evidence. Tested at the
    migrate tier: at cordon nearly every synthetic failing drive clears the
    threshold, recall is pinned at 1.0 and neither interval can vary."""
    obs = _repeated_fleet(300, 4)
    clustered = bootstrap_tier_ci(tier_counts(obs, THRESHOLDS), n_boot=300, seed=1)["migrate"]

    # Naive: one "drive" per observation.
    naive_obs = obs.with_columns(pl.Series("drive_id", [f"o{i}" for i in range(obs.height)]))
    naive = bootstrap_tier_ci(tier_counts(naive_obs, THRESHOLDS), n_boot=300, seed=1)["migrate"]

    clustered_width = clustered["recall"][1] - clustered["recall"][0]
    naive_width = naive["recall"][1] - naive["recall"][0]
    assert clustered_width > 1.4 * naive_width


def test_the_interval_brackets_the_point_estimate():
    obs = _repeated_fleet(300, 2, seed=3)
    counts = tier_counts(obs, THRESHOLDS)
    point, ci = tier_metrics(counts), bootstrap_tier_ci(counts, n_boot=300, seed=2)
    for tier in ("warn", "cordon", "migrate"):
        lo, hi = ci[tier]["precision"]
        assert lo <= point[tier]["precision"] <= hi


def test_the_bootstrap_is_deterministic_for_a_seed():
    counts = tier_counts(_repeated_fleet(100, 2), THRESHOLDS)
    assert bootstrap_tier_ci(counts, n_boot=50, seed=5) == bootstrap_tier_ci(
        counts, n_boot=50, seed=5
    )


def test_a_snapshot_with_no_failures_cannot_be_bootstrapped():
    counts = tier_counts(_obs([("A", 0.5, 0), ("B", 0.2, 0)]), THRESHOLDS)
    with pytest.raises(ValueError, match="failing"):
        bootstrap_tier_ci(counts, n_boot=10)


# --- recall at matched false-alarm rate ----------------------------------------


def test_recall_at_far_never_exceeds_the_budget_and_handles_ties():
    scores = np.array([0.9, 0.8, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1])
    failed = np.array([True, False, False, True, False, True, False, False])
    curve = _prepare_far(scores, failed)
    ones = np.ones(len(scores))
    # 5 negatives. Budget 0.2 allows 1; the two negatives tied at 0.8 cannot
    # be split, so neither is flagged and only the 0.9 positive is caught.
    assert recall_at_far(curve, ones, (0.2,)) == pytest.approx([1 / 3])
    # Budget 0.4 allows 2: both tied negatives, then the positive at 0.5 too.
    assert recall_at_far(curve, ones, (0.4,)) == pytest.approx([2 / 3])
    # Budget 1.0 flags everything.
    assert recall_at_far(curve, ones, (1.0,)) == pytest.approx([1.0])


def test_recall_at_a_zero_budget_flags_only_what_outscores_every_negative():
    scores = np.array([0.95, 0.5, 0.9, 0.3])
    failed = np.array([True, False, True, False])
    curve = _prepare_far(scores, failed)
    assert recall_at_far(curve, np.ones(4), (0.0,)) == pytest.approx([1.0])


# --- one look versus two ---------------------------------------------------------


def _pairs(n: int, *, second_look_helps: bool, seed: int = 0) -> pl.DataFrame:
    """Failing drives score high on BOTH days. Healthy drives occasionally
    spike on one day only (a noisy reading) - which a second look can reject.
    When `second_look_helps` is False the two days are identical."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fails = rng.random() < 0.1
        base = rng.beta(6, 2) if fails else rng.beta(1.5, 6)
        p_cur = base
        p_prev = (
            base
            if not second_look_helps
            else (base if fails else (rng.beta(1.5, 6) if rng.random() < 0.7 else base))
        )
        rows.append((f"d{i}", p_prev, p_cur, int(fails)))
    return pl.DataFrame(
        rows, schema=["drive_id", "p_prev", "p_cur", "label"], orient="row"
    ).with_columns(pl.col("label").cast(pl.Int8))


def test_a_second_look_helps_when_healthy_drives_spike_on_one_day_only():
    result = compare_looks(_pairs(4000, second_look_helps=True), budgets=(5e-3, 2e-2), n_boot=200)
    assert all(row["difference"] > 0 for row in result)
    # The paired interval excludes zero at the looser budget.
    assert result[-1]["difference_ci"][0] > 0


def test_a_second_look_adds_nothing_when_the_two_days_agree():
    """min(p, p) == p, so the policies are identical and the paired difference
    is exactly zero - not merely statistically indistinguishable."""
    result = compare_looks(_pairs(2000, second_look_helps=False), budgets=(1e-2,), n_boot=100)
    assert result[0]["difference"] == pytest.approx(0.0)
    assert result[0]["difference_ci"] == (0.0, 0.0)


def test_pairing_excludes_drives_with_no_earlier_look():
    previous = _obs([("A", 0.5, 0), ("B", 0.6, 1)], dt.date(2026, 5, 24))
    current = _obs([("A", 0.4, 0), ("B", 0.7, 1), ("NEW", 0.9, 1)], dt.date(2026, 5, 31))
    pairs = pair_observations(previous, current)
    assert sorted(pairs["drive_id"].to_list()) == ["A", "B"]
    # The label is the later date's.
    assert pairs.filter(pl.col("drive_id") == "B")["p_prev"][0] == pytest.approx(0.6)
    assert pairs.filter(pl.col("drive_id") == "B")["p_cur"][0] == pytest.approx(0.7)


def test_an_empty_pair_set_compares_nothing():
    assert compare_looks(_pairs(0, second_look_helps=True)) == []


def test_tier_counts_is_a_frozen_value_object():
    counts = tier_counts(_obs([("A", 0.5, 1)]), THRESHOLDS)
    assert isinstance(counts, TierCounts)
    with pytest.raises(AttributeError):
        counts.tiers = ()  # type: ignore[misc]
