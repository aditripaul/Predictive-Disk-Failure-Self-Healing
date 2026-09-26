import json

import polars as pl

from src.reliability.audit_export import (
    decisions_to_dataframe,
    export_decisions_to_csv,
    export_decisions_to_parquet,
)

_DECISIONS = [
    {
        "decision_id": "d1",
        "action_id": "a1",
        "drive_id": "D-1",
        "trust_score_provisional": 0.9,
        "trust_score_final": None,
        "execution_result": {"success": True},
        "guardrail_result": {"violations": [{"rule_id": "HARD_QUORUM"}]},
    },
    {
        "decision_id": "d2",
        "action_id": "a2",
        "drive_id": "D-2",
        "trust_score_provisional": 0.4,
        "trust_score_final": 0.4,
        "execution_result": None,
        "guardrail_result": None,
    },
]


def test_decisions_to_dataframe_flattens_nested_fields_to_json_strings():
    df = decisions_to_dataframe(_DECISIONS)
    assert df.height == 2
    assert df["trust_score_provisional"].to_list() == [0.9, 0.4]
    execution_result_d1 = df.filter(pl.col("decision_id") == "d1")["execution_result"][0]
    assert json.loads(execution_result_d1) == {"success": True}


def test_decisions_to_dataframe_keeps_null_nested_fields_null():
    df = decisions_to_dataframe(_DECISIONS)
    row_d2 = df.filter(pl.col("decision_id") == "d2")
    assert row_d2["execution_result"][0] is None
    assert row_d2["guardrail_result"][0] is None


def test_decisions_to_dataframe_is_empty_for_no_decisions():
    df = decisions_to_dataframe([])
    assert df.height == 0


def test_export_decisions_to_csv_round_trips():
    csv_bytes = export_decisions_to_csv(_DECISIONS)
    read_back = pl.read_csv(csv_bytes)
    assert read_back.height == 2
    assert set(read_back["decision_id"]) == {"d1", "d2"}


def test_export_decisions_to_parquet_round_trips():
    parquet_bytes = export_decisions_to_parquet(_DECISIONS)
    read_back = pl.read_parquet(parquet_bytes)
    assert read_back.height == 2
    assert set(read_back["action_id"]) == {"a1", "a2"}
