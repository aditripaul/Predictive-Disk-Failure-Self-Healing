import numpy as np
import pytest

from pipelines.experiment_model import VARIANTS, _train_survival_bounds, run_variant
from src.models.survival import (
    MIN_SURVIVAL_DAYS,
    risk_from_survival_time,
    survival_bounds,
    train_aft,
)

CUTOFF = np.datetime64("2026-04-15")


def _bounds(dates, event_types, days_to_event, horizon_days=30):
    return survival_bounds(
        np.array(dates, dtype="datetime64[D]"),
        np.array(event_types, dtype=object),
        np.array(days_to_event, dtype=np.float64),
        cutoff_date=CUTOFF,
        horizon_days=horizon_days,
    )


def test_a_failure_inside_the_follow_up_is_an_exact_time():
    lower, upper = _bounds(["2026-04-05"], ["confirmed_failure"], [12])
    assert lower[0] == upper[0] == 12


def test_a_failure_beyond_the_binary_horizon_is_still_an_exact_time():
    """The point of the survival view: day 35 is a negative for a 30-day
    label, but here it is a failure 35 days out (follow-up is 10 + 30 = 40)."""
    lower, upper = _bounds(["2026-04-05"], ["failure_followed_by_replacement"], [35])
    assert lower[0] == upper[0] == 35


def test_a_failure_after_the_follow_up_is_censored_so_no_future_is_leaked():
    lower, upper = _bounds(["2026-04-05"], ["confirmed_failure"], [41])
    assert lower[0] == 40
    assert np.isinf(upper[0])


def test_running_and_removed_drives_are_censored_at_the_follow_up():
    lower, upper = _bounds(
        ["2026-04-15", "2026-03-16", "2026-04-05"],
        ["still_active", "removed_without_failure", "preventive_replacement"],
        [np.nan, np.nan, 5],
    )
    assert lower.tolist() == [30, 60, 40]
    assert np.isinf(upper).all()


def test_times_are_kept_strictly_positive():
    lower, upper = _bounds(["2026-04-05"], ["confirmed_failure"], [0])
    assert lower[0] == upper[0] == MIN_SURVIVAL_DAYS


def test_risk_is_higher_for_sooner_predicted_failure_and_stays_in_zero_one():
    risk = risk_from_survival_time(np.array([0.0, 30.0, 300.0, 1e9]), 30)
    assert risk[0] == 1.0
    assert risk[1] == pytest.approx(0.5)
    assert (np.diff(risk) < 0).all()
    assert ((risk > 0) & (risk <= 1)).all()


def _synthetic(n, seed):
    """One informative feature: the larger it is, the sooner the failure."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0, 1, n)
    x = np.column_stack([signal, rng.normal(size=n)]).astype(np.float32)
    time = np.exp(5.0 - 4.0 * signal + rng.normal(scale=0.3, size=n))
    return x, time


def test_train_aft_ranks_drives_that_fail_sooner_as_riskier():
    x, time = _synthetic(4000, seed=0)
    follow_up = 40.0
    observed = time <= follow_up
    lower = np.where(observed, time, follow_up).astype(np.float32)
    upper = np.where(observed, time, np.inf).astype(np.float32)
    x_eval, time_eval = _synthetic(1000, seed=1)
    y_eval = (time_eval <= 30).astype(np.int8)

    model = train_aft(
        x,
        lower,
        upper,
        eval_x=x_eval,
        eval_y=y_eval,
        horizon_days=30,
        params={"min_child_weight": 1},
        num_boost_round=200,
        early_stopping_rounds=20,
    )
    risk = model.predict_proba(x_eval)[:, 1]
    assert risk[y_eval == 1].mean() > risk[y_eval == 0].mean() + 0.2
    assert model.tree_count >= 1
    gains = model.gain_by_feature(2)
    assert gains[0] > gains[1]  # the informative feature dominates


def _experiment_data(n_train=3000, n_eval=800):
    import polars as pl

    def ids(n, seed, start):
        x, time = _synthetic(n, seed)
        dates = np.datetime64(start) + np.arange(n) % 20
        fails = time <= 60
        return (
            x,
            (time <= 30).astype(np.int8),
            pl.DataFrame(
                {
                    "drive_id": [f"d{seed}_{i}" for i in range(n)],
                    "date": dates.astype("datetime64[D]"),
                    "drive_model": ["M"] * n,
                    "event_type": np.where(fails, "confirmed_failure", "still_active"),
                    "days_to_event": np.where(fails, np.ceil(time), np.nan),
                }
            ),
        )

    x_train, y_train, train_ids = ids(n_train, 0, "2026-03-27")
    x_val, y_val, val_ids = ids(n_eval, 1, "2026-04-16")
    x_test, y_test, test_ids = ids(n_eval, 2, "2026-05-11")
    return {
        "feature_columns": ["signal", "noise"],
        "horizon_days": 30,
        "x_train": x_train,
        "y_train": y_train,
        "train_ids": train_ids,
        "train_drives": train_ids["drive_id"].to_numpy(),
        "x_val": x_val,
        "y_val": y_val,
        "val_ids": val_ids,
        "val_drives": val_ids["drive_id"].to_numpy(),
        "x_test": x_test,
        "y_test": y_test,
        "test_ids": test_ids,
        "test_drives": test_ids["drive_id"].to_numpy(),
        "es_idx": np.arange(n_eval),
    }


@pytest.mark.parametrize("name", ["xgboost_aft", "xgboost_aft_spw"])
def test_survival_variants_run_through_the_experiment_and_beat_chance(name):
    assert name in VARIANTS
    data = _experiment_data()
    result = run_variant(name, data)
    assert result["test_row_auprc"] > 0.9  # base rate is about 0.59
    assert result["trees"] >= 1
    assert result["top_features"][0][0] == "signal"
    val_scores, test_scores = data["_scores"][name]
    assert len(val_scores) == len(data["y_val"])
    assert ((test_scores > 0) & (test_scores <= 1)).all()


def test_survival_variant_explains_a_work_dir_without_time_to_event_columns():
    data = _experiment_data(n_train=50, n_eval=20)
    data["train_ids"] = data["train_ids"].drop(["event_type", "days_to_event"])
    with pytest.raises(SystemExit, match="keep-work-dir"):
        _train_survival_bounds(data)


def test_test_split_stops_at_the_configured_test_end(tmp_path):
    """After test_end only failing drive-days still carry a label, so those
    dates must not be evaluated."""
    import datetime as dt

    import polars as pl

    from pipelines.train_model import _evaluation_period_predicate, _row_group_parts

    config = {"splits": {"test_end": "2026-05-31"}}
    assert _evaluation_period_predicate("train", config) is None
    assert _evaluation_period_predicate("validation", config) is None
    assert _evaluation_period_predicate("test", {"splits": {}}) is None

    frame = pl.DataFrame(
        {
            "drive_id": ["a", "b", "c", "d"],
            "date": [
                dt.date(2026, 5, 20),
                dt.date(2026, 5, 31),
                dt.date(2026, 6, 1),
                dt.date(2026, 5, 1),
            ],
            "split": ["test", "test", "test", "validation"],
            "label": [0, 0, 1, 0],
        }
    )
    frame_path = tmp_path / "frame.parquet"
    frame.write_parquet(frame_path)
    parts = _row_group_parts(
        frame_path,
        split_name="test",
        select_columns=["drive_id", "date", "label"],
        read_columns=["drive_id", "date", "label", "split"],
        extra_predicate=_evaluation_period_predicate("test", config),
        tmp_dir=tmp_path,
    )
    kept = pl.concat([pl.read_parquet(p) for p in parts])
    assert kept["drive_id"].to_list() == ["a", "b"]
