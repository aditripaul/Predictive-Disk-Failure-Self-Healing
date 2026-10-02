import datetime as dt

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.models.action_tiers import determine_action_tier
from src.models.evaluation import (
    compute_auprc,
    compute_calibration,
    compute_warning_lead_time_days,
    evaluate_at_threshold,
    precision_at_k,
    precision_at_k_fractions,
)
from src.models.features import (
    assemble_training_frame,
    drive_batched_inner_join,
    join_by_native_row_groups,
    select_feature_columns,
)
from src.models.threshold import tune_threshold_for_precision
from src.models.training import predict_proba_positive, train_lightgbm

ACTION_THRESHOLDS = {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}


def test_assemble_training_frame_and_select_feature_columns():
    gold = pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
            "model_family": ["Seagate HDD"] * 3,
            "reallocated_sector_count_7d_mean": [0.0, 1.0, 5.0],
        }
    )
    labels = pl.DataFrame(
        {
            "drive_id": ["A", "A", "B"],
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
            "horizon_days": [14, 14, 14],
            "label": [0, 1, None],
            "split": ["train", "train", "train"],
        }
    )
    frame = assemble_training_frame(gold, labels, horizon_days=14)
    assert frame.height == 2  # the null-label row for B is dropped

    feature_columns = select_feature_columns(frame)
    assert feature_columns == ["reallocated_sector_count_7d_mean"]
    assert "model_family" not in feature_columns
    assert "label" not in feature_columns


def _wide_and_narrow_fixture(n_drives: int = 12, days: int = 10, seed: int = 0):
    """A gold-features-like frame sorted by date (not grouped by drive_id,
    matching src/features/pivot.py's real output ordering) plus a narrow
    label-like frame with some rows deliberately missing (censored), for
    exercising the batched-join functions against something closer to
    the real training-join shape than a handful of hand-picked rows."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "drive_id": f"D{d}",
            "date": dt.date(2024, 1, 1) + dt.timedelta(days=day),
            "f0": float(rng.random()),
        }
        for d in range(n_drives)
        for day in range(days)
    ]
    wide = pl.DataFrame(rows).sort("date")
    keep = rng.random(wide.height) < 0.85
    narrow = pl.DataFrame(
        {
            "drive_id": wide["drive_id"],
            "date": wide["date"],
            "label": (rng.random(wide.height) < 0.5).astype(np.int8),
        }
    ).filter(pl.Series(keep))
    return wide, narrow


def test_drive_batched_inner_join_matches_a_direct_join():
    """Regression test for the bug two commits fixed: an earlier version
    built left.filter(...).join(right.filter(...), ...) as one combined
    lazy plan and it produced a request to allocate ~354 quintillion
    bytes against real data - a query-shape bug, not a real memory need.
    The fix collects each side separately before an eager join; this
    pins that both the fix and the original row-offset chunking it
    replaced give the same rows as an unbatched join."""
    wide, narrow = _wide_and_narrow_fixture()
    direct = wide.join(narrow, on=["drive_id", "date"], how="inner")
    key = ["drive_id", "date"]
    for n_batches in (1, 3, 7, 50):  # 50 > n_drives: exercises empty batches too
        out = drive_batched_inner_join(
            wide.lazy(), narrow.lazy(), on=["drive_id", "date"], n_batches=n_batches
        )
        assert out.sort(key).equals(direct.sort(key).select(out.columns)), n_batches


def test_drive_batched_inner_join_empty_result_keeps_the_joined_schema():
    wide = pl.DataFrame({"drive_id": ["X"], "date": [dt.date(2024, 1, 1)], "f0": [1.0]})
    narrow = pl.DataFrame({"drive_id": ["Y"], "date": [dt.date(2024, 1, 1)], "label": [1]})
    out = drive_batched_inner_join(wide.lazy(), narrow.lazy(), on=["drive_id", "date"], n_batches=4)
    assert out.shape == (0, 4)
    assert set(out.columns) == {"drive_id", "date", "f0", "label"}


def test_drive_batched_inner_join_out_path_matches_the_returned_dataframe(tmp_path):
    """out_path is the actual production path now (pipelines/train_model.py
    always passes it) - the whole point is that the joined result is
    NEVER held as one eager DataFrame, so this must be checked by reading
    the written file back, not by inspecting a return value (out_path
    returns None)."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=15, days=10, seed=4)
    direct = wide.join(narrow, on=["drive_id", "date"], how="inner")
    key = ["drive_id", "date"]

    path = tmp_path / "joined.parquet"
    result = drive_batched_inner_join(
        wide.lazy(), narrow.lazy(), on=["drive_id", "date"], n_batches=5, out_path=path
    )
    assert result is None
    out = pl.read_parquet(path)
    assert out.sort(key).equals(direct.sort(key).select(out.columns))


def test_drive_batched_inner_join_out_path_empty_result_writes_the_joined_schema(tmp_path):
    """Covers _finalize_batched_join's empty-batches branch: when every
    batch is empty, no ParquetWriter is ever opened, so the empty schema
    must still be written explicitly rather than leaving no file at
    all."""
    wide = pl.DataFrame({"drive_id": ["X"], "date": [dt.date(2024, 1, 1)], "f0": [1.0]})
    narrow = pl.DataFrame({"drive_id": ["Y"], "date": [dt.date(2024, 1, 1)], "label": [1]})
    path = tmp_path / "joined.parquet"
    result = drive_batched_inner_join(
        wide.lazy(), narrow.lazy(), on=["drive_id", "date"], n_batches=4, out_path=path
    )
    assert result is None
    out = pl.read_parquet(path)
    assert out.shape == (0, 4)
    assert set(out.columns) == {"drive_id", "date", "f0", "label"}


def test_join_by_native_row_groups_matches_a_direct_join_across_row_group_counts(tmp_path):
    """Covers the two things build_gold_features.py's own writer can
    produce: a single row group (small datasets never split) and
    multiple row groups that do NOT align to drive boundaries (real
    batches routinely exceeded pyarrow's un-configured 1,048,576-row
    default before row_group_size was passed explicitly) - correctness
    must hold either way, since Parquet row groups are always an
    exhaustive, non-overlapping partition of a file's rows regardless of
    where the split falls."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=15, days=12, seed=1)
    direct = wide.join(narrow, on=["drive_id", "date"], how="inner")
    key = ["drive_id", "date"]

    for row_group_size in (wide.height, 25, 7):
        path = tmp_path / f"wide_{row_group_size}.parquet"
        writer = pq.ParquetWriter(path, wide.to_arrow().schema)
        writer.write_table(wide.to_arrow(), row_group_size=row_group_size)
        writer.close()
        assert pq.ParquetFile(path).num_row_groups >= (wide.height // row_group_size)

        out = join_by_native_row_groups(path, narrow.lazy(), on=["drive_id", "date"])
        assert out.sort(key).equals(direct.sort(key).select(out.columns)), row_group_size


def test_join_by_native_row_groups_max_rows_per_join_does_not_change_the_result(tmp_path):
    """max_rows_per_join slices each already-in-memory row group before
    joining, purely to bound peak memory - it must not change which rows
    come out, at a sub-row-group granularity that doesn't evenly divide
    the row group."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=15, days=12, seed=2)
    direct = wide.join(narrow, on=["drive_id", "date"], how="inner")
    key = ["drive_id", "date"]

    path = tmp_path / "wide.parquet"
    wide.write_parquet(path)
    for max_rows_per_join in (None, 1, 37, 10_000):
        out = join_by_native_row_groups(
            path, narrow.lazy(), on=["drive_id", "date"], max_rows_per_join=max_rows_per_join
        )
        assert out.sort(key).equals(direct.sort(key).select(out.columns)), max_rows_per_join


def test_join_by_native_row_groups_empty_result_keeps_the_joined_schema(tmp_path):
    wide = pl.DataFrame({"drive_id": ["X"], "date": [dt.date(2024, 1, 1)], "f0": [1.0]})
    narrow = pl.DataFrame({"drive_id": ["Y"], "date": [dt.date(2024, 1, 1)], "label": [1]})
    path = tmp_path / "wide.parquet"
    wide.write_parquet(path)
    out = join_by_native_row_groups(path, narrow.lazy(), on=["drive_id", "date"])
    assert out.shape == (0, 4)
    assert set(out.columns) == {"drive_id", "date", "f0", "label"}


def test_join_by_native_row_groups_out_path_matches_the_returned_dataframe(tmp_path):
    """Same production-path check as drive_batched_inner_join's out_path
    test, for the row-group-native join path assemble_training_frame
    actually uses in pipelines/train_model.py."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=15, days=12, seed=5)
    direct = wide.join(narrow, on=["drive_id", "date"], how="inner")
    key = ["drive_id", "date"]

    wide_path = tmp_path / "wide.parquet"
    writer = pq.ParquetWriter(wide_path, wide.to_arrow().schema)
    writer.write_table(wide.to_arrow(), row_group_size=25)
    writer.close()

    out_path = tmp_path / "joined.parquet"
    result = join_by_native_row_groups(
        wide_path, narrow.lazy(), on=["drive_id", "date"], out_path=out_path
    )
    assert result is None
    out = pl.read_parquet(out_path)
    assert out.sort(key).equals(direct.sort(key).select(out.columns))


def test_assemble_training_frame_gold_features_path_matches_the_in_memory_path(tmp_path):
    """pipelines/train_model.py always passes gold_features_path now, so
    this is the actual production code path - pins it against the
    simpler in-memory join every other assemble_training_frame test
    exercises, at a scale wide enough to span several row groups."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=20, days=10, seed=3)
    wide = wide.with_columns(pl.lit("Seagate HDD").alias("model_family"))
    labels = narrow.rename({"label": "label"}).with_columns(
        pl.lit(14, dtype=pl.Int64).alias("horizon_days"),
        pl.lit("train").alias("split"),
    )

    baseline = assemble_training_frame(wide, labels, horizon_days=14)

    path = tmp_path / "gold.parquet"
    writer = pq.ParquetWriter(path, wide.to_arrow().schema)
    writer.write_table(wide.to_arrow(), row_group_size=40)  # force several row groups
    writer.close()
    assert pq.ParquetFile(path).num_row_groups > 1

    via_path = assemble_training_frame(
        wide.lazy(),
        labels,
        horizon_days=14,
        chunk_rows=17,
        gold_features_path=path,
    )
    key = ["drive_id", "date"]
    assert baseline.sort(key).equals(via_path.sort(key).select(baseline.columns))


def test_assemble_training_frame_out_path_matches_the_in_memory_path(tmp_path):
    """The exact combination pipelines/train_model.py's _stage_assemble
    now uses in production: gold_features_path (join by native row
    groups) plus out_path (write incrementally, never hold the full
    joined training frame as one eager DataFrame). This is the code path
    that fixed the crash that happened right after all row groups
    processed successfully - every earlier test either returns a
    DataFrame or exercises out_path alone, not both together."""
    wide, narrow = _wide_and_narrow_fixture(n_drives=20, days=10, seed=6)
    wide = wide.with_columns(pl.lit("Seagate HDD").alias("model_family"))
    labels = narrow.rename({"label": "label"}).with_columns(
        pl.lit(14, dtype=pl.Int64).alias("horizon_days"),
        pl.lit("train").alias("split"),
    )

    baseline = assemble_training_frame(wide, labels, horizon_days=14)

    gold_path = tmp_path / "gold.parquet"
    writer = pq.ParquetWriter(gold_path, wide.to_arrow().schema)
    writer.write_table(wide.to_arrow(), row_group_size=40)  # force several row groups
    writer.close()

    frame_path = tmp_path / "frame.parquet"
    result = assemble_training_frame(
        wide.lazy(),
        labels,
        horizon_days=14,
        chunk_rows=17,
        gold_features_path=gold_path,
        out_path=frame_path,
    )
    assert result is None
    key = ["drive_id", "date"]
    via_path = pl.read_parquet(frame_path)
    assert baseline.sort(key).equals(via_path.sort(key).select(baseline.columns))


def _toy_classification_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    return x, y


def test_train_lightgbm_and_predict_proba():
    x, y = _toy_classification_data()
    model = train_lightgbm(x, y, params={"n_estimators": 20})
    scores = predict_proba_positive(model, x)
    assert scores.shape == (200,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert compute_auprc(y, scores) > 0.5


def test_tune_threshold_for_precision_meets_target():
    x, y = _toy_classification_data()
    model = train_lightgbm(x, y, params={"n_estimators": 50})
    scores = predict_proba_positive(model, x)

    result = tune_threshold_for_precision(y, scores, target_precision=0.9)
    assert result["precision"] >= 0.9 or not result["target_met"]

    metrics = evaluate_at_threshold(y, scores, result["threshold"])
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_tune_threshold_falls_back_when_target_unreachable():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # indistinguishable scores
    result = tune_threshold_for_precision(y, scores, target_precision=0.999)
    assert result["target_met"] is False


def test_precision_at_k_picks_highest_scored_rows():
    y_true = np.array([0, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    # top-2 by score: indices 2 (0.9, label 1) and 3 (0.8, label 1) -> precision 1.0
    assert precision_at_k(y_true, y_scores, k=2) == 1.0
    # top-3 by score adds index 4 (0.3, label 0) -> 2/3
    assert abs(precision_at_k(y_true, y_scores, k=3) - 2 / 3) < 1e-9


def test_precision_at_k_caps_k_at_population_size():
    y_true = np.array([1, 0])
    y_scores = np.array([0.9, 0.1])
    assert precision_at_k(y_true, y_scores, k=1000) == 0.5


def test_precision_at_k_fractions_returns_one_entry_per_fraction():
    y_true = np.array([1] * 10 + [0] * 90)
    y_scores = np.linspace(1.0, 0.0, 100)
    result = precision_at_k_fractions(y_true, y_scores, fractions=(0.1,))
    assert result["precision_at_top_10pct"] == 1.0


def test_compute_calibration_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    y_scores = rng.uniform(0, 1, size=2000)
    y_true = (rng.uniform(0, 1, size=2000) < y_scores).astype(int)
    result = compute_calibration(y_true, y_scores, n_bins=10)
    assert result["expected_calibration_error"] < 0.1
    assert len(result["bins"]) == 10
    assert result["brier_score"] >= 0.0


def test_compute_calibration_flags_badly_miscalibrated_scores():
    y_true = np.array([0] * 100)
    y_scores = np.array([0.9] * 100)
    result = compute_calibration(y_true, y_scores, n_bins=10)
    assert result["expected_calibration_error"] > 0.5


def _lead_time_frame():
    return pl.DataFrame(
        {
            "drive_id": ["A", "A", "A", "B", "B", "C"],
            "days_to_event": [10, 5, 1, 3, 1, 2],
            "score": [0.2, 0.9, 0.95, 0.1, 0.1, 0.9],
        }
    )


def test_compute_warning_lead_time_picks_earliest_crossing_per_drive():
    df = _lead_time_frame()
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    # drive A first crosses threshold at days_to_event=5 (not the later 1)
    # drive B never crosses threshold; drive C crosses at days_to_event=2
    assert result["warned_drive_count"] == 2
    assert result["total_failed_drive_count"] == 3
    assert result["max_lead_time_days"] == 5.0
    assert abs(result["mean_lead_time_days"] - (5 + 2) / 2) < 1e-9
    assert abs(result["warning_coverage"] - 2 / 3) < 1e-9


def test_compute_warning_lead_time_handles_no_warnings():
    df = pl.DataFrame({"drive_id": ["A"], "days_to_event": [5], "score": [0.1]})
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    assert result["warned_drive_count"] == 0
    assert result["mean_lead_time_days"] is None
    assert result["warning_coverage"] == 0.0


def test_compute_warning_lead_time_handles_empty_frame():
    df = pl.DataFrame({"drive_id": [], "days_to_event": [], "score": []})
    result = compute_warning_lead_time_days(df, score_column="score", threshold=0.5)
    assert result["total_failed_drive_count"] == 0
    assert result["warning_coverage"] is None


def test_determine_action_tier_downgrades_low_confidence_destructive_action():
    tier = determine_action_tier(
        p_fail=0.95,
        feature_confidence=0.1,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.CORDON


def test_determine_action_tier_allows_destructive_action_with_high_confidence():
    tier = determine_action_tier(
        p_fail=0.95,
        feature_confidence=0.9,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.DRAIN


def test_determine_action_tier_low_risk_is_monitor():
    tier = determine_action_tier(
        p_fail=0.05,
        feature_confidence=0.99,
        feature_maturity=FeatureMaturity.MATURE,
        stale_telemetry=False,
        action_thresholds=ACTION_THRESHOLDS,
        min_confidence_for_destructive_action=0.8,
    )
    assert tier == ActionTier.MONITOR
