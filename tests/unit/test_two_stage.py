import numpy as np

from pipelines.experiment_model import drive_folds, two_stage_experiment
from src.models.training import early_stopping_subset


def test_drive_folds_keep_every_drive_in_one_fold():
    drives = np.array([f"d{i % 50}" for i in range(1000)])
    folds = drive_folds(drives, 3)
    assert set(folds.tolist()) == {0, 1, 2}
    for drive in np.unique(drives):
        assert len(set(folds[drives == drive].tolist())) == 1
    np.testing.assert_array_equal(folds, drive_folds(drives, 3))  # deterministic


def _split(seed: int, n_drives: int, tmp_path, name: str):
    """Drives with 10 days each; ~15% fail, and failing drives have a shifted
    feature 0 plus a weaker signal in feature 1 on their last 5 days."""
    rng = np.random.default_rng(seed)
    days = 10
    drives = np.repeat([f"{name}{i}" for i in range(n_drives)], days)
    failing = np.repeat(rng.random(n_drives) < 0.15, days)
    last_days = np.tile(np.arange(days) >= 5, n_drives)
    y = (failing & last_days).astype(int)
    x = rng.normal(size=(n_drives * days, 4)).astype(np.float32)
    x[y == 1, 0] += 2.0
    x[y == 1, 1] += 1.0
    path = tmp_path / f"x_{name}.npy"
    np.save(path, x)
    return np.load(path, mmap_mode="r"), y, drives


def test_two_stage_experiment_runs_and_reports_both_models(tmp_path):
    x_train, y_train, train_drives = _split(1, 900, tmp_path, "tr")
    x_val, y_val, val_drives = _split(2, 500, tmp_path, "va")
    x_test, y_test, test_drives = _split(3, 500, tmp_path, "te")
    data = {
        "x_train": np.asarray(x_train),
        "y_train": y_train,
        "train_drives": train_drives,
        "x_val": x_val,
        "y_val": y_val,
        "val_drives": val_drives,
        "x_test": x_test,
        "y_test": y_test,
        "test_drives": test_drives,
        "es_idx": early_stopping_subset(y_val),
    }
    report = two_stage_experiment(data, n_folds=3, candidate_recall=0.5)
    assert report["status"] == "ok", report
    assert report["candidates"]["train_rows"] < len(y_train)  # stage 2 sees a subset
    assert [r["target_recall"] for r in report["stage1"]] == [0.05, 0.10, 0.20, 0.35]
    for row in report["two_stage"]:
        assert row["threshold"] is not None
        assert 0.0 <= row["test"]["precision"] <= 1.0
