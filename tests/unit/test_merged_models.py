"""Merged approach: Cox stage, TCN, rank blend and the split/leak checks.
Synthetic data only."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.labels.event_types import CONFIRMED_FAILURE
from src.models.ensemble import (
    assert_no_row_overlap,
    assert_split_dates,
    assert_survival_censored,
    rank_blend,
    validation_rank,
)
from src.models.survival import cox_labels, risk_from_hazard_ratio, survival_bounds, train_cox


def test_cox_labels_sign_marks_censoring() -> None:
    lower = np.array([3.0, 40.0, 0.5])
    upper = np.array([3.0, np.inf, 0.5])
    assert cox_labels(lower, upper).tolist() == [3.0, -40.0, 0.5]


def test_survival_bounds_censor_at_train_cutoff() -> None:
    dates = np.array(["2026-01-01", "2026-01-15", "2026-01-10"], dtype="datetime64[D]")
    events = np.array([CONFIRMED_FAILURE] * 3)
    # Fails 10 days later (seen), 40 days later (after the cutoff), 5 days later (seen).
    days = np.array([10.0, 40.0, 5.0])
    cutoff_row = np.datetime64("2026-01-15")
    lower, upper = survival_bounds(dates, events, days, cutoff_date=cutoff_row, horizon_days=30)
    assert np.isfinite(upper).tolist() == [True, False, True]
    assert lower[1] == 30.0  # censored at the follow-up, not at the true 40 days
    assert_survival_censored(dates, lower, upper, outcome_cutoff=dt.date(2026, 2, 14))
    with pytest.raises(AssertionError):
        assert_survival_censored(dates, lower, upper, outcome_cutoff=dt.date(2026, 2, 13))


def test_train_cox_ranks_failures_higher() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 4)).astype(np.float32)
    risk = x[:, 0]
    fail = rng.random(n) < 1 / (1 + np.exp(-(risk * 3 - 3)))
    lower = np.where(fail, rng.integers(1, 30, n), 60).astype(np.float64)
    upper = np.where(fail, lower, np.inf)
    model = train_cox(
        x[:3000],
        lower[:3000],
        upper[:3000],
        eval_x=x[3000:],
        eval_y=fail[3000:].astype(int),
        num_boost_round=50,
        early_stopping_rounds=10,
    )
    p = model.predict_proba(x[3000:])[:, 1]
    assert ((p > 0) & (p < 1)).all()
    assert p[fail[3000:]].mean() > p[~fail[3000:]].mean()


def test_hazard_ratio_map_is_monotone() -> None:
    hr = np.array([0.01, 0.5, 1.0, 20.0])
    assert np.all(np.diff(risk_from_hazard_ratio(hr)) > 0)


def test_validation_rank_and_blend() -> None:
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert validation_rank(ref, np.array([0.05, 0.2, 0.9])).tolist() == [0.0, 0.5, 1.0]
    assert np.isnan(validation_rank(ref, np.array([np.nan]))[0])
    val = {"a": ref, "b": ref * 100}
    target = {"a": np.array([0.2, 0.4]), "b": np.array([40.0, 20.0])}
    assert rank_blend(val, target, ["a", "b"]).tolist() == [0.75, 0.75]
    with pytest.raises(ValueError):
        rank_blend(val, target, [])


def test_split_assertions() -> None:
    d = dt.date
    good = {
        "train": pl.Series([d(2025, 10, 1), d(2026, 1, 15)]),
        "validation": pl.Series([d(2026, 2, 15), d(2026, 3, 31)]),
        "test": pl.Series([d(2026, 4, 1), d(2026, 5, 31)]),
    }
    windows = dict(
        train_last_row=d(2026, 1, 15),
        validation_start=d(2026, 2, 15),
        validation_end=d(2026, 3, 31),
        test_start=d(2026, 4, 1),
        test_end=d(2026, 5, 31),
    )
    assert assert_split_dates(good, **windows)["test"]["max_date"] == "2026-05-31"
    leaky = {**good, "train": pl.Series([d(2026, 1, 16)])}  # inside the purge gap
    with pytest.raises(AssertionError):
        assert_split_dates(leaky, **windows)
    late_test = {**good, "test": pl.Series([d(2026, 6, 1)])}
    with pytest.raises(AssertionError):
        assert_split_dates(late_test, **windows)


def test_row_overlap_assertion() -> None:
    a = pl.DataFrame({"drive_id": ["x"], "date": [dt.date(2026, 3, 1)]})
    b = pl.DataFrame({"drive_id": ["x"], "date": [dt.date(2026, 3, 2)]})
    assert_no_row_overlap({"validation": a, "test": b})
    with pytest.raises(AssertionError):
        assert_no_row_overlap({"validation": a, "test": a})


def test_tcn_shapes_and_window_causality() -> None:
    torch = pytest.importorskip("torch")
    from src.models.sequence_net import build_sequence_net

    torch.manual_seed(0)
    net = build_sequence_net("tcn", n_channels=5, n_static=3, hidden_size=8)
    net.eval()
    x = torch.randn(4, 30, 5)
    s = torch.randn(4, 3)
    assert net(x, s).shape == (4,)
    with pytest.raises(ValueError):
        build_sequence_net("gru", n_channels=5, n_static=3)
    lstm = build_sequence_net("lstm", n_channels=5, n_static=3, hidden_size=8)
    assert lstm(x, s).shape == (4,)


def test_precomputed_scores_join(tmp_path) -> None:
    import json

    from src.agent.fleet import join_precomputed_scores, load_precomputed_scores

    scores = pl.DataFrame(
        {
            "split": ["test", "test"],
            "drive_id": ["a", "b"],
            "date": [dt.date(2026, 5, 31)] * 2,
            "lgbm_lstm_aft_tcn": [0.9, 0.1],
        }
    )
    path = tmp_path / "merged_scores_h30.parquet"
    scores.write_parquet(path)
    (tmp_path / "agent_thresholds_h30.json").write_text(
        json.dumps(
            {
                "score_column": "lgbm_lstm_aft_tcn",
                "action_tiers": {"warn": 0.5, "cordon": 0.8, "migrate": None, "drain": None},
            }
        )
    )
    fallback = {"warn": 0.3, "cordon": 0.5, "migrate": 0.7, "drain": 0.9}
    loaded, run_id, thresholds = load_precomputed_scores(path, fallback)
    assert run_id == "precomputed:lgbm_lstm_aft_tcn"
    assert thresholds["warn"] == 0.5 and thresholds["migrate"] > 1.0
    snapshot = pl.DataFrame(
        {
            "drive_id": ["a", "c"],
            "date": [dt.date(2026, 5, 31)] * 2,
            "feature_confidence": [1.0, 1.0],
        }
    )
    joined = join_precomputed_scores(snapshot, loaded)
    assert joined["drive_id"].to_list() == ["a"] and joined["p_fail"].to_list() == [0.9]


def test_tag_keeps_untagged_names_and_separates_tagged_runs(tmp_path) -> None:

    from pipelines import train_merged_models as tm

    kwargs = {"horizon_days": 30, "smoke_suffix": ""}
    plain = tm.output_paths(tmp_path, tmp_path, tag=None, **kwargs)
    a = tm.output_paths(tmp_path, tmp_path, tag="h30_seed0", **kwargs)
    b = tm.output_paths(tmp_path, tmp_path, tag="h30_seed1", **kwargs)
    assert plain["report"].name == "merged_evaluation_report.json"
    assert plain["scores"].name == "merged_scores_h30.parquet"
    assert plain["thresholds"].name == "agent_thresholds_h30.json"
    assert a["scores"].name == "merged_scores_h30_h30_seed0.parquet"
    assert not set(a.values()) & set(b.values())
    assert not set(a.values()) & set(plain.values())
    smoke = tm.output_paths(tmp_path, tmp_path, horizon_days=30, smoke_suffix=".smoke", tag="x")
    assert smoke["report"].name == "merged_evaluation_report.smoke_x.json"


def test_tag_rejects_unsafe_names() -> None:
    from pipelines import train_merged_models as tm

    for bad in ("../x", "a/b", "a b", ".hidden"):
        with pytest.raises(SystemExit):
            tm.tag_suffix(bad)
    assert tm.tag_suffix("") == ""


def test_mlflow_metrics_are_flat_and_finite() -> None:
    from pipelines import train_merged_models as tm

    results = {
        "aft": {
            "validation_drive_level": {"precision": 0.5, "recall": 0.1, "auprc": None},
            "test_drive_level": {"precision": float("nan"), "recall": 0.2},
        }
    }
    assert tm.mlflow_metrics(results) == {
        "aft__validation_drive_precision": 0.5,
        "aft__validation_drive_recall": 0.1,
        "aft__test_drive_recall": 0.2,
    }
