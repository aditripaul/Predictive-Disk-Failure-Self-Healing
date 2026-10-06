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


def test_anomaly_stage_experiment_reports_every_comparison(tmp_path):
    from pipelines.experiment_model import (
        _anomaly_scores,
        _print_anomaly_stage,
        anomaly_stage_experiment,
    )

    x_train, y_train, train_drives = _split(1, 900, tmp_path, "tr")
    x_val, y_val, val_drives = _split(2, 500, tmp_path, "va")
    x_test, y_test, test_drives = _split(3, 500, tmp_path, "te")
    x_train = np.array(x_train)
    x_train[0, 2] = np.nan  # the forest must tolerate missing values
    data = {
        "feature_columns": ["f0", "f1", "f2", "f3"],
        "x_train": x_train,
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
    report = anomaly_stage_experiment(data, keep_fractions=(0.5,), fit_rows=2000, n_estimators=25)
    for key in ("pooled", "anomaly_score_alone", "pooled_plus_anomaly_feature"):
        assert [r["target_recall"] for r in report[key]] == [0.05, 0.10, 0.20, 0.35]
    (cascade,) = report["cascade"]
    assert cascade["status"] == "ok", cascade
    assert cascade["test_rows"] < cascade["test_rows_total"]  # the filter removed rows
    assert 0 < cascade["failing_drives_kept"] <= cascade["failing_test_drives"]
    assert 0.0 <= report["anomaly_feature_gain_share"] <= 1.0
    assert np.isnan(data["x_train"][0, 2])  # inputs are left as they were
    _print_anomaly_stage(report)  # formats without error

    # Shifted rows are scored as more unusual than typical ones.
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(n_estimators=25, random_state=0).fit(
        np.random.default_rng(0).normal(size=(500, 2))
    )
    scores = _anomaly_scores(forest, np.array([[0.0, 0.0], [9.0, 9.0]], dtype=np.float32))
    assert scores[1] > scores[0]


# --- src/models/two_stage.py: the module `make train --two-stage` runs --------


class _Fixed:
    """A stand-in classifier that returns scores read from a feature column."""

    def __init__(self, column: int) -> None:
        self.column = column

    def predict_proba(self, x):
        score = np.asarray(x)[:, self.column].astype(float)
        return np.column_stack([1.0 - score, score])


def test_final_score_keeps_stage1_below_the_threshold_and_reranks_above_it(tmp_path):
    from src.models.two_stage import TwoStageModel

    # Column 0 is the Stage 1 score, column 1 what Stage 2 will answer.
    x = np.array(
        [[0.10, 0.9], [0.59, 0.9], [0.60, 0.0], [0.70, 1.0], [0.95, 0.5]], dtype=np.float32
    )
    model = TwoStageModel(_Fixed(0), _Fixed(1), candidate_threshold=0.6)
    final = model.predict_proba(x)[:, 1]

    np.testing.assert_allclose(final[:2], [0.10, 0.59], atol=1e-6)  # not candidates
    np.testing.assert_allclose(final[2:], [0.6, 1.0, 0.8], atol=1e-6)  # 0.6 + 0.4 * stage 2
    assert final[2:].min() >= 0.6 > final[:2].max()  # candidates always rank above
    assert final[3] > final[4]  # Stage 2 reorders the candidates (Stage 1 had 0.70 < 0.95)

    # Same answer from a memory map, scored in more than one chunk.
    import src.models.two_stage as two_stage

    path = tmp_path / "x.npy"
    np.save(path, x)
    mapped = np.load(path, mmap_mode="r")
    original = two_stage.PREDICT_CHUNK_ROWS
    two_stage.PREDICT_CHUNK_ROWS = 2
    try:
        chunked = model.final_scores(mapped, x[:, 0].astype(float))
    finally:
        two_stage.PREDICT_CHUNK_ROWS = original
    np.testing.assert_allclose(chunked, final, atol=1e-6)


def test_threshold_for_drive_recall_takes_the_most_precise_point_meeting_the_recall():
    from src.models.two_stage import threshold_for_drive_recall

    drives = np.array(["a", "b", "c", "d", "e", "f"])
    y = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    assert threshold_for_drive_recall(drives, y, scores, 0.6) == 0.8  # 2 of 3, no false alarm
    assert threshold_for_drive_recall(drives, y, scores, 1.0) == 0.6
    assert threshold_for_drive_recall(drives, np.zeros(6, dtype=int), scores, 0.5) is None


def _fit_real_two_stage(tmp_path, **kwargs):
    from src.models.training import predict_proba_positive
    from src.models.two_stage import fit_second_stage, lightgbm_fitter

    x_train, y_train, train_drives = _split(1, 900, tmp_path, "tr")
    x_val, y_val, val_drives = _split(2, 500, tmp_path, "va")
    fit = lightgbm_fitter(
        {"n_estimators": 200, "min_child_samples": 20, "learning_rate": 0.1},
        positive_weight_power=0.5,
        early_stopping_rounds=20,
    )
    x_train = np.asarray(x_train)
    stage1 = fit(x_train, y_train, eval_x=np.asarray(x_val), eval_y=y_val)
    val_s1 = predict_proba_positive(stage1, x_val)
    model, report = fit_second_stage(
        fit,
        stage1,
        x_train=x_train,
        y_train=y_train,
        train_drives=train_drives,
        x_val=x_val,
        y_val=y_val,
        val_drives=val_drives,
        val_stage1_scores=val_s1,
        **kwargs,
    )
    return model, report, stage1, (x_val, val_s1)


def test_fit_second_stage_trains_on_candidates_only_and_scores_consistently(tmp_path):
    model, report, stage1, (x_val, val_s1) = _fit_real_two_stage(tmp_path)
    assert report["status"] == "ok", report
    assert model is not None and model.stage1 is stage1
    assert 0 < report["candidates"]["train_rows"] < 9000
    assert report["candidates"]["train_positive_rows"] >= 20
    assert report["stage2_trees"] >= 1

    final = model.final_scores(x_val, val_s1)
    below = val_s1 < model.candidate_threshold
    np.testing.assert_allclose(final[below], val_s1[below])
    assert (final[~below] >= model.candidate_threshold).all()
    np.testing.assert_allclose(model.predict_proba(np.asarray(x_val))[:, 1], final)


def test_fit_second_stage_reports_why_it_did_not_train(tmp_path):
    model, report, _, _ = _fit_real_two_stage(tmp_path, min_candidate_positives=10**9)
    assert model is None
    assert report["status"] == "too few candidate failures to train a second stage"


def test_two_stage_model_survives_an_mlflow_round_trip(tmp_path):
    """What `make train --two-stage` logs is what `make score-fleet` loads."""
    import mlflow
    from src.models.two_stage import load_two_stage, log_two_stage

    model, report, stage1, (x_val, val_s1) = _fit_real_two_stage(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("two-stage-test")
    try:
        with mlflow.start_run() as single:
            mlflow.lightgbm.log_model(stage1, name="model")
        with mlflow.start_run() as run:
            mlflow.lightgbm.log_model(stage1, name="model")
            log_two_stage(model, report)

        assert load_two_stage(single.info.run_id, stage1) is None  # single-stage run

        loaded_stage1 = mlflow.lightgbm.load_model(f"runs:/{run.info.run_id}/model")
        loaded = load_two_stage(run.info.run_id, loaded_stage1)
        assert loaded is not None
        assert loaded.candidate_threshold == model.candidate_threshold
        np.testing.assert_allclose(
            loaded.predict_proba(np.asarray(x_val))[:, 1],
            model.final_scores(x_val, val_s1),
            rtol=1e-6,
        )
    finally:
        mlflow.set_tracking_uri("")


def test_seed_summary_counts_the_seeds_where_two_stage_is_better():
    from pipelines.experiment_model import summarize_two_stage_seeds

    def run(p1, r1, p2, r2):
        def row(p, r):
            return {"target_recall": 0.1, "threshold": 0.5, "test": {"precision": p, "recall": r}}

        return {"status": "ok", "stage1": [row(p1, r1)], "two_stage": [row(p2, r2)]}

    reports = [
        run(0.40, 0.10, 0.50, 0.10),  # better
        run(0.40, 0.10, 0.45, 0.09),  # more precise but fewer caught: not counted
        run(0.40, 0.10, 0.38, 0.10),  # worse
        {"status": "candidate recall not reachable on validation"},
    ]
    (row,) = summarize_two_stage_seeds(reports)
    assert row["seeds"] == 3
    assert row["two_stage_better_seeds"] == 1
    np.testing.assert_allclose(row["stage1_precision"], 0.40)
    np.testing.assert_allclose(row["two_stage_precision"], (0.50 + 0.45 + 0.38) / 3)
    np.testing.assert_allclose(
        [row["precision_gain_min"], row["precision_gain_max"]], [-0.02, 0.10]
    )
    assert summarize_two_stage_seeds([{"status": "nope"}]) == []


def test_model_card_states_whether_a_second_stage_was_used():
    from src.models.model_card import _two_stage_lines

    assert "one" in _two_stage_lines(None)[0]
    assert "not trained" in _two_stage_lines({"status": "too few candidate failures"})[0]
    lines = _two_stage_lines(
        {
            "status": "ok",
            "candidate_threshold": 0.91,
            "candidate_recall": 0.5,
            "stage2_trees": 136,
            "candidates": {"train_rows": 9654, "train_positive_rows": 5343},
            "stage1_precision_at_recall": [
                {
                    "target_recall": 0.1,
                    "threshold": 0.96,
                    "test": {
                        "precision": 0.414,
                        "recall": 0.085,
                        "caught_drive_count": 53,
                        "false_alarm_drive_count": 75,
                    },
                }
            ],
        }
    )
    assert "two" in lines[0] and "9654" in lines[0]
    assert "41.4%" in lines[1]
