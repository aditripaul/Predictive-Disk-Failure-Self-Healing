"""The horizon comparison scores every model against every label definition on
the same drive-days, matched on false-alarm rate, not on precision."""

import datetime as dt

import numpy as np
import polars as pl

from src.models.horizon_comparison import common_frame, compare_horizons, comparison_tables

START = dt.date(2026, 7, 1)


def _fleet(n_drives: int = 300, days: int = 40, seed: int = 0):
    """Drives with `days` rows from START; 10% fail on day `days` + 5. The
    30-day label marks the last 30 days before failure, the 60-day one all of
    them. Model "early" rises 50 days before failure, model "late" 15 days."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_drives):
        fails = rng.random() < 0.1
        for d in range(days):
            date = START + dt.timedelta(days=d)
            to_fail = days + 5 - d if fails else None
            rows.append(
                {
                    "drive_id": f"d{i:03d}",
                    "date": date,
                    "days_to_event": to_fail,
                    "label_30": int(fails and to_fail <= 30),
                    "label_60": int(fails and to_fail <= 60),
                    "early": rng.random() * 0.5 + (0.6 if fails and to_fail <= 50 else 0.0),
                    "late": rng.random() * 0.5 + (0.6 if fails and to_fail <= 15 else 0.0),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _inputs(fleet: pl.DataFrame):
    scores = {
        30: fleet.select("drive_id", "date", pl.col("late").alias("score")),
        60: fleet.select("drive_id", "date", pl.col("early").alias("score")),
    }
    labels = pl.concat(
        [
            fleet.select(
                "drive_id",
                "date",
                pl.lit(h).alias("horizon_days"),
                pl.col(f"label_{h}").alias("label"),
                "days_to_event",
            )
            for h in (30, 60)
        ]
    )
    return scores, labels


def test_common_frame_keeps_only_the_shared_window_and_observed_labels():
    fleet = _fleet(n_drives=20, days=10)
    scores, labels = _inputs(fleet)
    labels = labels.with_columns(
        pl.when((pl.col("horizon_days") == 60) & (pl.col("date") == START))
        .then(None)
        .otherwise(pl.col("label"))
        .alias("label")
    )
    frame = common_frame(scores, labels, start=START, end=START + dt.timedelta(days=4))
    assert frame["date"].min() == START + dt.timedelta(days=1)  # day 0: 60-day label unobserved
    assert frame["date"].max() == START + dt.timedelta(days=4)
    assert set(frame.columns) >= {"score_30", "score_60", "label_30", "label_60"}
    assert frame.height == 20 * 4


def test_an_earlier_alerting_model_has_the_longer_lead_and_recall_is_matched_on_far():
    fleet = _fleet()
    scores, labels = _inputs(fleet)
    frame = common_frame(scores, labels, start=START, end=START + dt.timedelta(days=39))
    result = compare_horizons(frame, [30, 60], budgets=(0.01,), n_boot=50, n_boot_auroc=20)
    block = result["labels"][30]
    late = block["models"][30]["at_false_alarm_rate"][0]
    early = block["models"][60]["at_false_alarm_rate"][0]
    assert late["reachable"] and early["reachable"]
    assert late["false_alarm_rate"] <= 0.01 and early["false_alarm_rate"] <= 0.01
    assert early["lead_time"]["median_days"] > late["lead_time"]["median_days"]
    versus = block["models"][60]["versus_reference"]
    assert versus["at_false_alarm_rate"][0]["budget"] == 0.01
    assert "difference_ci" in versus["auroc"]
    # The 60-day labels mark more rows positive: the base rate a precision
    # comparison would be confounded by.
    assert result["labels"][60]["row_positive_rate"] > block["row_positive_rate"]
    text = comparison_tables(
        result,
        {"start": START, "end": START, "n_boot": 50, "n_boot_auroc": 20},
    )
    assert "Scored against 30-day labels" in text and "| 60-day |" in text


def test_compare_horizons_script_end_to_end(tmp_path, monkeypatch):
    """The window ends at observation_end minus the longest horizon."""
    import json

    import yaml

    import scripts.compare_horizons as script

    fleet = _fleet(n_drives=120, days=20)
    scores, labels = _inputs(fleet)
    paths = []
    for h, frame in scores.items():
        path = tmp_path / f"scores_{h}.parquet"
        frame.with_columns(split=pl.lit("test")).write_parquet(path)
        paths.append(f"{h}={path}")
    labels_path = tmp_path / "labels.parquet"
    labels.write_parquet(labels_path)
    config = tmp_path / "model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                # 2026-09-14 minus 60 days = 2026-07-16
                "splits": {"sealed_start": "2026-07-01", "observation_end": "2026-09-14"},
                "evaluation": {"top_k": [5, 10]},
            }
        )
    )
    accesses = []
    monkeypatch.setattr(script, "MODEL_CONFIG_PATH", config)
    monkeypatch.setattr(script, "record_test_access", lambda **kw: accesses.append(kw))
    out = tmp_path / "out"
    script.main(
        [
            "--scores",
            *paths,
            "--labels",
            str(labels_path),
            "--out",
            str(out),
            "--n-boot",
            "20",
            "--n-boot-auroc",
            "10",
            "--budgets",
            "0.02",
        ]
    )
    saved = json.loads((out / "horizon_comparison.json").read_text())
    assert (saved["settings"]["start"], saved["settings"]["end"]) == ("2026-07-01", "2026-07-16")
    assert saved["settings"]["rows"] == 120 * 16
    assert len(accesses) == 1
    assert (out / "tables.md").exists()
    assert (out / "figures" / "labels_30d_precision_recall.png").exists()
    assert (out / "figures" / "labels_60d_confusion.png").exists()
