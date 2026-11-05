"""The merged-run tables, the per-model agent thresholds and the agent driver."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from scripts.run_agent_models import write_thresholds
from src.agent.fleet import load_precomputed_scores
from src.reporting.merged_tables import agent_table, model_label, model_table

FROZEN_REPORT = Path("frozen_runs/fb58d9576c6146ba94c13389005fa15b/merged_evaluation_report.json")

# The table published for the frozen run, row for row.
EXPECTED_ROWS = [
    "| LightGBM two-stage | 59.2% | 57.1% [51.6, 62.6] | 12.0% | 178 / 134 | 0.923 | 0.339 |",
    "| AFT | 64.2% | 68.5% [63.0, 74.1] | 12.4% | 183 / 84 | 0.921 | 0.387 |",
    "| Cox | 60.6% | 62.9% [57.6, 67.9] | 13.7% | 202 / 119 | 0.927 | 0.357 |",
    "| LSTM | 44.3% | 52.7% [48.2, 57.1] | 15.6% | 231 / 207 | 0.923 | 0.281 |",
    "| TCN | 55.5% | 70.5% [65.2, 75.6] | 13.6% | 201 / 84 | 0.900 | 0.348 |",
    "| Blend LightGBM + AFT + TCN | 66.8% | 70.1% [64.9, 75.0] | 13.7% | "
    "202 / 86 | 0.921 | 0.389 |",
    "| Blend LightGBM + LSTM + AFT + TCN (agent) | 64.5% | 69.7% [64.4, 75.0] | 12.6% | "
    "186 / 81 | 0.925 | 0.382 |",
]


def test_model_table_reproduces_the_frozen_runs_published_table() -> None:
    report = json.loads(FROZEN_REPORT.read_text())
    lines = model_table(report, agent_ensemble="lgbm_lstm_aft_tcn").splitlines()
    assert lines[0].startswith("| Model | Val P | Test P [95% CI]")
    assert lines[2:] == EXPECTED_ROWS


def test_labels_name_blends_from_their_members_in_order() -> None:
    ensembles = {"b": ["lightgbm", "tcn", "aft"]}
    assert model_label("b", ensembles, None) == "Blend LightGBM + TCN + AFT"
    assert model_label("b", ensembles, "b") == "Blend LightGBM + TCN + AFT (agent)"
    with pytest.raises(KeyError):
        model_label("unknown", ensembles, None)


def _run(precision: float | None, acted: int = 10, labelled: int = 8, caught: int = 6) -> dict:
    return {
        "candidate_count": 500,
        "sent_to_human_review": 3,
        "scoring": {
            "drives_acted_on": acted,
            "drives_acted_on_with_labels": labelled,
            "caught": caught,
            "agent_precision": precision,
            "agent_recall": 0.04,
            "actions_by_tier": {"warn": 20, "cordon": 7, "migrate": 2, "drain": 1},
        },
    }


def test_agent_table_rows_and_missing_runs() -> None:
    text = agent_table(
        {
            "aft": _run(0.75),
            "cox": None,
            "tcn": {"candidate_count": 9, "scoring": {"skipped": "x"}},
        },
        ensembles={},
    )
    rows = text.splitlines()[2:]
    assert rows[0] == "| AFT | 500 | 10 (8) | 6 / 2 | 75.0% | 4.0% | 20 / 7 / 2 / 1 | 3 |"
    assert rows[1].startswith("| Cox | not run")
    assert "unscored: x" in rows[2]
    assert "n/a" in agent_table({"aft": _run(None)}, ensembles={})  # nothing acted on


def test_write_thresholds_uses_that_models_own_tiers(tmp_path) -> None:
    report = json.loads(FROZEN_REPORT.read_text())
    meta = write_thresholds(report, "aft", tmp_path / "t.json")
    assert meta["score_column"] == "aft"
    assert meta["action_tiers"] == report["models"]["aft"]["action_tiers"]
    assert meta["action_tiers"] != report["models"]["lightgbm"]["action_tiers"]
    assert json.loads((tmp_path / "t.json").read_text()) == meta


def test_precomputed_scores_accept_an_explicit_thresholds_file(tmp_path) -> None:
    pl.DataFrame(
        {"drive_id": ["a"], "date": [dt.date(2026, 5, 31)], "aft": [0.9], "tcn": [0.1]}
    ).write_parquet(tmp_path / "merged_scores_h30_x.parquet")
    thresholds = tmp_path / "thresholds_tcn.json"
    thresholds.write_text(
        json.dumps(
            {
                "score_column": "tcn",
                "action_tiers": {"warn": 0.05, "cordon": None, "migrate": None, "drain": None},
            }
        )
    )
    fallback = {"warn": 0.3, "cordon": 0.5, "migrate": 0.7, "drain": 0.9}
    scores, run_id, tiers = load_precomputed_scores(
        tmp_path / "merged_scores_h30_x.parquet", fallback, thresholds
    )
    assert run_id == "precomputed:tcn" and scores["p_fail"].to_list() == [0.1]
    assert tiers["warn"] == 0.05
    # without it, the paired agent_thresholds_* file is required and absent
    with pytest.raises(FileNotFoundError):
        load_precomputed_scores(tmp_path / "merged_scores_h30_x.parquet", fallback)
