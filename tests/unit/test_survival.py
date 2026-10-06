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


def test_each_split_is_limited_to_its_own_dates(tmp_path):
    """Train stops a horizon before train_end, validation starts after
    train_end, test stops at test_end."""
    import datetime as dt

    import polars as pl

    from pipelines.train_model import _row_group_parts, _split_period_predicate

    config = {"splits": {"train_end": "2026-04-15", "test_end": "2026-05-31"}}
    assert _split_period_predicate("test", {"splits": {}}, 30) is None
    assert _split_period_predicate("train", {"splits": {}}, 30) is None
    off = {
        "splits": {
            **config["splits"],
            "purge_label_window": False,
            "validation_after_train_only": False,
        }
    }
    assert _split_period_predicate("train", off, 30) is None
    assert _split_period_predicate("validation", off, 30) is None
    assert _split_period_predicate("test", off, 30) is not None

    d = dt.date
    frame = pl.DataFrame(
        {
            "drive_id": ["t1", "t2", "t3", "v1", "v2", "v3", "s1", "s2", "s3"],
            "date": [
                d(2026, 3, 16),
                d(2026, 3, 17),
                d(2026, 4, 15),  # train
                d(2026, 2, 1),
                d(2026, 4, 15),
                d(2026, 4, 16),  # validation (v1, v2: held-out drives)
                d(2026, 5, 20),
                d(2026, 5, 31),
                d(2026, 6, 1),  # test
            ],
            "split": ["train"] * 3 + ["validation"] * 3 + ["test"] * 3,
            "label": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    frame_path = tmp_path / "frame.parquet"
    frame.write_parquet(frame_path)

    def kept(split_name, horizon_days):
        out = tmp_path / f"{split_name}_{horizon_days}"
        out.mkdir()
        parts = _row_group_parts(
            frame_path,
            split_name=split_name,
            select_columns=["drive_id", "date", "label"],
            read_columns=["drive_id", "date", "label", "split"],
            extra_predicate=_split_period_predicate(split_name, config, horizon_days),
            tmp_dir=out,
        )
        return pl.concat([pl.read_parquet(p) for p in parts])["drive_id"].to_list()

    assert kept("train", 30) == ["t1"]  # 2026-03-16 + 30 days = train_end
    assert kept("train", 14) == ["t1", "t2"]
    assert kept("validation", 30) == ["v3"]
    assert kept("test", 30) == ["s1", "s2"]


def test_extract_stage_applies_the_split_dates_end_to_end(tmp_path, monkeypatch):
    """The real extract stage on a tiny frame: train is purged of the last
    horizon before train_end, validation drops the held-out drives'
    training-period rows, test stops at test_end, and the ids keep the
    time-to-event columns the survival variants need."""
    import datetime as dt
    import json

    import polars as pl
    import yaml

    import pipelines.train_model as train_model

    config = {
        "model": {"max_train_rows": 1000},
        "splits": {
            "train_end": "2026-04-15",
            "validation_end": "2026-05-10",
            "test_end": "2026-05-31",
        },
    }
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(train_model, "MODEL_CONFIG_PATH", config_path)
    monkeypatch.setattr(train_model, "apply_memory_limit_from_config", lambda: None)

    d = dt.date
    rows = [
        # (drive, date, split, label)
        ("a", d(2026, 3, 1), "train", 0),
        ("a", d(2026, 3, 16), "train", 1),
        ("a", d(2026, 3, 17), "train", 1),  # label window runs past train_end
        ("a", d(2026, 4, 15), "train", 0),  # same
        ("h", d(2026, 3, 1), "validation", 0),  # held-out drive, training period
        ("a", d(2026, 4, 16), "validation", 0),
        ("h", d(2026, 5, 10), "validation", 1),
        ("a", d(2026, 5, 11), "test", 0),
        ("a", d(2026, 5, 31), "test", 1),
        ("a", d(2026, 6, 1), "test", 1),  # after test_end
    ]
    frame = pl.DataFrame(
        {
            "drive_id": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            "split": [r[2] for r in rows],
            "label": pl.Series([r[3] for r in rows], dtype=pl.Int8),
            "drive_model": ["M"] * len(rows),
            "event_type": ["confirmed_failure"] * len(rows),
            "days_to_event": [20] * len(rows),
            "f1": [float(i) for i in range(len(rows))],
        }
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    frame.write_parquet(work_dir / "frame.parquet")
    (work_dir / "frame_meta.json").write_text(
        json.dumps({"feature_columns": ["f1"], "split_counts": {}, "horizon_days": 30})
    )

    for split in ("train", "validation", "test"):
        train_model._stage_extract_split(work_dir, split)

    def dates(name):
        return pl.read_parquet(work_dir / f"{name}_ids.parquet")["date"].to_list()

    assert dates("train") == [d(2026, 3, 1), d(2026, 3, 16)]
    assert dates("validation") == [d(2026, 4, 16), d(2026, 5, 10)]
    assert dates("test") == [d(2026, 5, 11), d(2026, 5, 31)]
    assert np.load(work_dir / "x_train.npy").ravel().tolist() == [0.0, 1.0]
    assert np.load(work_dir / "y_train.npy").tolist() == [0, 1]
    assert np.load(work_dir / "y_val.npy").tolist() == [0, 1]
    assert json.loads((work_dir / "train_meta.json").read_text()) == {
        "row_count": 2,
        "row_count_before_cap": 2,
    }
    assert {"event_type", "days_to_event"} <= set(
        pl.read_parquet(work_dir / "train_ids.parquet").columns
    )
