from pipelines.generate_final_report import _data_period, _goal_status

MODEL_CONFIG = {
    "threshold": {"target_precision": 0.95, "target_recall_range": [0.35, 0.5]},
    "splits": {"train_end": "2026-02-22"},
    "primary_horizon_days": 14,
}


def test_goal_status_says_not_met_when_the_target_was_not_reached():
    report = {
        "threshold": {
            "target_met": False,
            "precision": 0.3,
            "recall": 0.35,
            "precision_gap_to_target": 0.65,
        },
        "test_drive_level": {"precision": 0.22, "recall": 0.26},
    }
    status = _goal_status(report, MODEL_CONFIG)
    assert status["target_met"] is False
    assert status["target_precision"] == 0.95
    assert "not met" in status["summary"]


def test_goal_status_without_a_model_report_is_not_yet_available():
    assert _goal_status(None, MODEL_CONFIG)["status"] == "not_yet_available"


def test_data_period_reads_ingest_window_and_splits():
    data_config = {"sources": {"backblaze": {"start_date": "2026-01-01", "end_date": "2026-06-30"}}}
    period = _data_period(data_config, MODEL_CONFIG)
    assert period["ingested_to"] == "2026-06-30"
    assert period["primary_horizon_days"] == 14
