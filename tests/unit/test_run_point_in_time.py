"""End-to-end run of `pipelines/run_point_in_time.py` on a synthetic gold table.

`main` wires together date selection, per-date scoring, pooling, the cluster
bootstrap and the one-look/two-look comparison, and it has never run against
anything real. This runs the whole path with the model and MLflow stubbed, so
a wiring mistake shows up here rather than after a long run on real data.
"""

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import yaml

from pipelines import run_point_in_time as runner

DATES = [dt.date(2026, 5, 17), dt.date(2026, 5, 24), dt.date(2026, 5, 31)]
N_DRIVES = 400


class _Model:
    def predict_proba(self, x):
        p = np.clip(x.max(axis=1) / 100.0, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def _gold(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    gold = tmp_path / "gold"
    (gold / "features").mkdir(parents=True)
    (gold / "labels").mkdir(parents=True)

    failing = set(rng.choice(N_DRIVES, size=40, replace=False).tolist())
    feature_rows, label_rows = [], []
    for i in range(N_DRIVES):
        for day_index, day in enumerate(DATES):
            # A few healthy drives leave before the last date.
            if i % 50 == 0 and day_index == 2 and i not in failing:
                continue
            score = rng.uniform(55, 95) if i in failing else rng.uniform(0, 70)
            feature_rows.append({"drive_id": f"d{i:04d}", "date": day, "smart_5_raw": score})
            label_rows.append(
                {
                    "drive_id": f"d{i:04d}",
                    "date": day,
                    "horizon_days": 30,
                    "label": 1 if i in failing else 0,
                }
            )
    pl.DataFrame(feature_rows).write_parquet(gold / "features" / "part.parquet")
    pl.DataFrame(label_rows, schema_overrides={"label": pl.Int8}).write_parquet(
        gold / "labels" / "part.parquet"
    )
    return gold


def _configs(tmp_path: Path, gold: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(Path("configs/data.yaml").read_text())
    data["gold_dir"] = str(gold)
    data["sources"]["backblaze"]["end_date"] = "2026-06-30"
    model = yaml.safe_load(Path("configs/model.yaml").read_text())
    model["splits"].update(
        {"validation_end": "2026-05-10", "test_end": "2026-05-31", "train_end": "2026-04-01"}
    )
    data_path, model_path = tmp_path / "data.yaml", tmp_path / "model.yaml"
    data_path.write_text(yaml.safe_dump(data))
    model_path.write_text(yaml.safe_dump(model))
    return data_path, model_path


@pytest.fixture
def run_report(tmp_path):
    gold = _gold(tmp_path)
    data_path, model_path = _configs(tmp_path, gold)
    out = tmp_path / "report.json"
    accesses: list[dict] = []

    with (
        patch.object(runner, "DATA_CONFIG_PATH", data_path),
        patch.object(runner, "MODEL_CONFIG_PATH", model_path),
        patch.object(runner, "load_scoring_model", lambda *_a, **_k: _Model()),
        patch.object(runner, "latest_run_id_for_horizon", lambda *_a, **_k: "run-test"),
        patch.object(
            runner,
            "action_thresholds_for_run",
            lambda *_a, **_k: {"warn": 0.30, "cordon": 0.55, "migrate": 0.70, "drain": 0.85},
        ),
        patch.object(
            runner, "record_test_access", lambda *a, **k: accesses.append({"args": a, **k})
        ),
        patch("mlflow.set_tracking_uri"),
    ):
        runner.main(["--bootstrap", "40", "--out", str(out), "--start-date", "2026-05-17"])
    return json.loads(out.read_text()), accesses


def test_the_run_snapshots_every_requested_date(run_report):
    report, _ = run_report
    assert report["dates"] == [str(d) for d in DATES]
    assert report["run_id"] == "run-test"
    assert report["horizon_days"] == 30


def test_drives_that_left_are_absent_from_the_dates_after_they_left(run_report):
    report, _ = run_report
    by_date = {row["date"]: row["alive"] for row in report["per_date"]}
    # Healthy drives with i % 50 == 0 that are not failing leave before the 31st.
    assert by_date["2026-05-31"] < by_date["2026-05-24"]


def test_pooled_counts_are_the_sum_of_the_snapshots(run_report):
    report, _ = run_report
    assert report["pooled"]["observations"] == sum(r["alive"] for r in report["per_date"])
    assert report["pooled"]["failing_observations"] == sum(r["failing"] for r in report["per_date"])
    assert report["pooled"]["drives"] == N_DRIVES


def test_the_pooled_precision_has_an_interval_that_brackets_it(run_report):
    report, _ = run_report
    for tier in ("warn", "cordon", "migrate", "drain"):
        point = report["pooled"]["tiers"][tier]["precision"]
        lo, hi = report["pooled"]["ci"][tier]["precision"]
        assert lo <= point <= hi


def test_the_base_rate_is_reported_beside_precision(run_report):
    """Precision lives at a base rate and means nothing without it."""
    report, _ = run_report
    pooled = report["pooled"]
    assert pooled["base_rate"] == pytest.approx(
        pooled["failing_observations"] / pooled["observations"]
    )
    assert pooled["tiers"]["cordon"]["base_rate"] == pytest.approx(pooled["base_rate"])


def test_the_two_look_comparison_runs_on_consecutive_pairs(run_report):
    report, _ = run_report
    assert report["lag_days"] == 7
    assert report["two_look"], "two pairs of consecutive weekly snapshots exist"
    assert report["two_look_pairs"] > 0
    for row in report["two_look"]:
        lo, hi = row["difference_ci"]
        assert lo <= row["difference"] <= hi + 1e-12


def test_a_test_window_run_is_recorded_in_the_access_log(run_report):
    """Fixed-date snapshots inside the test window are test-split reads."""
    _, accesses = run_report
    assert len(accesses) == 1
    assert accesses[0]["variant"] == "agent_point_in_time"
    assert accesses[0]["extra"]["run_id"] == "run-test"


def test_a_date_past_the_last_observable_label_is_refused(tmp_path):
    gold = _gold(tmp_path)
    data_path, model_path = _configs(tmp_path, gold)
    with (
        patch.object(runner, "DATA_CONFIG_PATH", data_path),
        patch.object(runner, "MODEL_CONFIG_PATH", model_path),
        pytest.raises(SystemExit, match="no observable"),
    ):
        runner.main(["--end-date", "2026-06-15", "--out", str(tmp_path / "x.json")])


def test_a_lag_that_is_not_a_multiple_of_the_step_is_refused(tmp_path):
    gold = _gold(tmp_path)
    data_path, model_path = _configs(tmp_path, gold)
    with (
        patch.object(runner, "DATA_CONFIG_PATH", data_path),
        patch.object(runner, "MODEL_CONFIG_PATH", model_path),
        pytest.raises(SystemExit, match="multiple"),
    ):
        runner.main(["--step-days", "7", "--lag-days", "10", "--out", str(tmp_path / "x.json")])
