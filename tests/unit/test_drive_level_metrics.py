import numpy as np
import pytest

from src.models.evaluation import drive_level_metrics, drive_level_table

# Drives: A fails (rows y=0,0,1,1), B fails (0,1), C healthy, D healthy, E healthy.
DRIVES = np.array(["A"] * 4 + ["B"] * 2 + ["C"] * 3 + ["D"] * 2 + ["E"])
Y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.3, 0.2, 0.7, 0.1, 0.05, 0.05, 0.4])


def test_drive_level_table_scores_failing_drives_on_their_positive_rows_only():
    labels, scores = drive_level_table(DRIVES, Y, SCORES)
    # A's 0.9 is on a pre-window row (y=0) so it is ignored: max over y=1 rows is 0.8.
    by_drive = {}
    for drive, label, score in zip(sorted(set(DRIVES)), labels, scores, strict=True):
        by_drive[drive] = (int(label), float(score))
    assert by_drive == {
        "A": (1, 0.8),
        "B": (1, 0.3),
        "C": (0, 0.7),
        "D": (0, 0.05),
        "E": (0, 0.4),
    }


def test_drive_level_metrics_counts_caught_and_false_alarm_drives():
    metrics = drive_level_metrics(DRIVES, Y, SCORES, threshold=0.5)
    assert metrics["failing_drive_count"] == 2
    assert metrics["healthy_drive_count"] == 3
    assert metrics["caught_drive_count"] == 1  # A only
    assert metrics["false_alarm_drive_count"] == 1  # C only
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 3)


def test_drive_level_metrics_low_threshold_catches_everything():
    metrics = drive_level_metrics(DRIVES, Y, SCORES, threshold=0.0)
    assert metrics["recall"] == 1.0
    assert metrics["false_alarm_drive_count"] == 3


def test_drive_level_metrics_does_not_overcount_a_drives_many_positive_rows():
    # One failing drive with 14 identical positive rows plus 14 healthy drives.
    drives = np.array(["F"] * 14 + [f"H{i}" for i in range(14)])
    y = np.array([1] * 14 + [0] * 14)
    scores = np.array([0.9] * 14 + [0.9] * 7 + [0.1] * 7)
    row_level_precision = ((scores >= 0.5) & (y == 1)).sum() / (scores >= 0.5).sum()
    metrics = drive_level_metrics(drives, y, scores, threshold=0.5)
    assert row_level_precision == pytest.approx(14 / 21)  # inflated by the 14 duplicates
    assert metrics["precision"] == pytest.approx(1 / 8)  # 1 caught vs 7 false-alarm drives


def test_drive_level_metrics_with_no_failing_drives_is_all_zero_not_an_error():
    metrics = drive_level_metrics(np.array(["a", "b"]), np.array([0, 0]), np.array([0.2, 0.9]), 0.5)
    assert metrics["recall"] == 0.0
    assert metrics["auprc"] == 0.0
    assert metrics["false_alarm_drive_count"] == 1


def _threshold_fixture():
    # 4 failing drives (scores .9 .8 .4 .3), 6 healthy drives (.7 .5 .2 .1 .1 .05)
    drives = np.array([f"F{i}" for i in range(4)] + [f"H{i}" for i in range(6)])
    y = np.array([1] * 4 + [0] * 6)
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.5, 0.2, 0.1, 0.1, 0.05])
    return drives, y, scores


def test_drive_level_threshold_meets_goal_when_reachable():
    from src.models.threshold import tune_drive_level_threshold

    drives, y, scores = _threshold_fixture()
    result = tune_drive_level_threshold(
        drives, y, scores, target_precision=0.6, target_recall_range=(0.5, 0.75)
    )
    # thr .8: caught 2/4, 0 false alarms -> precision 1.0, recall .5 (meets)
    # thr .4: caught 3/4, 2 false alarms -> precision .6, recall .75 (meets)
    # thr .3: caught 4/4, 2 false alarms -> precision .667, recall 1.0 (meets, highest recall)
    assert result["target_met"] is True
    assert result["threshold"] == pytest.approx(0.3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(4 / 6)


def test_drive_level_threshold_falls_back_to_best_precision_inside_the_recall_range():
    from src.models.threshold import tune_drive_level_threshold

    drives, y, scores = _threshold_fixture()
    result = tune_drive_level_threshold(
        drives, y, scores, target_precision=0.99, target_recall_range=(0.75, 1.0)
    )
    # recall >= .75 needs thr <= .4: precision .6 at thr .4, 4/6 at thr .3 -> best is 4/6.
    assert result["target_met"] is False
    assert result["recall"] >= 0.75
    assert result["precision"] == pytest.approx(4 / 6)
    assert result["precision_gap_to_target"] == pytest.approx(0.99 - 4 / 6)
    assert result["level"] == "drive"


def test_drive_level_threshold_never_returns_a_zero_recall_point_when_unreachable():
    from src.models.threshold import tune_drive_level_threshold

    drives, y, scores = _threshold_fixture()
    result = tune_drive_level_threshold(
        drives, y, scores, target_precision=0.999, target_recall_range=(0.5, 0.75)
    )
    assert result["recall"] >= 0.5
