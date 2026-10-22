"""Entry point for `make train-sequence`: an LSTM over each drive's daily
history, evaluated exactly like `make train` so the two can be compared.

Why it exists: the tree model sees hand-built 7/14/30-day window features. If
the signal is in the *shape* of the recent history, a recurrent model reading
the raw 30-day window should rank failing drives differently; if not, the
ceiling is in the data, not the feature set. `pipelines/train_lstm.py` could
not answer that - it builds one window per drive at the drive's last day, where
a failing drive's label is censored.

What is shared with `make train`, deliberately:
- the same labelled drive-days: the gold label table for one horizon, rows with
  a non-null label, the label table's `split`, and the same per-split date rules
  (`_split_period_predicate`: purge window, validation after `train_end`, test
  up to `test_end`);
- the same recipe: sqrt positive-class weight (`model.positive_weight_power`),
  early stopping on validation average precision, the capped-negative training
  sample (`model.max_train_rows`);
- the same evaluation: every labelled validation and test drive-day is scored,
  reduced to one score per drive by `drive_level_table`, thresholds picked on
  validation only, and the ROC-space table (`recall_at_false_alarm_table`) that
  does not move with the base rate.

What differs: the model. Inputs are 30 days of 10 SMART attributes (log1p, with
a per-attribute "reported" flag), drive age, vendor and capacity; there are no
rolling statistics, model-family z-scores or two-stage re-ranker.

Writes `data/audit/data_quality_reports/sequence_evaluation_report.json` (never
`model_evaluation_report.json`, which `make score-fleet` reads) and every
validation/test score to `<scratch>/sequence_model/` so the result can be
compared with, or blended into, another model on the same drives. Records one
test-split access in `data/audit/test_access_log.jsonl`.

Needs the torch extra: `uv sync --extra torch`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import average_precision_score

from pipelines.train_model import _split_period_predicate
from src.features.sequence_windows import (
    DenseTelemetry,
    build_dense_telemetry,
    fit_normalization,
    gather_windows,
    static_feature_count,
)
from src.logging_config import configure_logging, get_logger
from src.models.access_log import record_test_access
from src.models.evaluation import (
    compute_auroc,
    drive_level_metrics,
    precision_at_recall_table,
    recall_at_false_alarm_table,
)
from src.models.threshold import tune_drive_level_threshold
from src.models.uncertainty import drive_level_bootstrap_ci
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")

logger = get_logger(__name__)


def _load_labelled_rows(
    labels_path: Path, horizon_days: int, dense: DenseTelemetry, start_date: dt.date
) -> pl.DataFrame:
    """`(drive_idx, day_idx, label, split)` for every observed-label drive-day of
    `horizon_days` whose drive is in `dense`. Integer columns only: the label
    table is tens of millions of rows and the string drive_id is not needed
    once it is mapped."""
    index = pl.LazyFrame(
        {
            "drive_id": dense.drive_ids,
            "drive_idx": pl.Series(np.arange(len(dense.drive_ids)), dtype=pl.Int32),
        }
    )
    return (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("horizon_days") == horizon_days)
            & pl.col("label").is_not_null()
            & pl.col("split").is_in(["train", "validation", "test"])
        )
        .join(index, on="drive_id", how="inner")
        .select(
            "drive_idx",
            (pl.col("date") - start_date).dt.total_days().cast(pl.Int32).alias("day_idx"),
            pl.col("label").cast(pl.Int8),
            "split",
            "date",
        )
        .collect()
    )


def _split_rows(labelled: pl.DataFrame, model_config: dict, horizon_days: int) -> dict:
    out = {}
    for name in ("train", "validation", "test"):
        part = labelled.filter(pl.col("split") == name)
        predicate = _split_period_predicate(name, model_config, horizon_days)
        if predicate is not None:
            part = part.filter(predicate)
        if part.height == 0:
            raise ValueError(f"split {name!r} has no labelled rows for {horizon_days}d")
        out[name] = part
    return out


def _cap_negatives(part: pl.DataFrame, max_negatives: int | None, seed: int) -> pl.DataFrame:
    """All positive rows, plus at most `max_negatives` random negatives."""
    positives = part.filter(pl.col("label") == 1)
    negatives = part.filter(pl.col("label") == 0)
    if max_negatives is not None and negatives.height > max_negatives:
        negatives = negatives.sample(n=max_negatives, seed=seed)
    return pl.concat([positives, negatives]).sort(["drive_idx", "day_idx"])


def _batch_fn(dense: DenseTelemetry, rows: pl.DataFrame, window_days: int):
    drive_idx = rows["drive_idx"].to_numpy()
    day_idx = rows["day_idx"].to_numpy()

    def batch(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gather_windows(dense, drive_idx[selected], day_idx[selected], window_days)

    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon-days", type=int)
    parser.add_argument("--max-train-rows", type=int, help="override sequence_model.max_train_rows")
    parser.add_argument("--max-epochs", type=int, help="override sequence_model.max_epochs")
    parser.add_argument(
        "--drive-fraction",
        type=float,
        default=1.0,
        help="Keep only this deterministic share of drives. For smoke runs; a fraction below 1 "
        "writes to sequence_evaluation_report.smoke.json and is not a result.",
    )
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    import torch  # deferred: the extra is optional

    from src.models.sequence_net import SequenceNet, predict_scores, train_sequence_net

    configure_logging()
    apply_memory_limit_from_config()
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    seq_cfg = model_config["sequence_model"]
    horizon_days = args.horizon_days or model_config["primary_horizon_days"]
    window_days = seq_cfg["window_days"]
    attributes = list(seq_cfg["attributes"])
    seed = seq_cfg.get("seed", 0)
    torch.set_num_threads(args.threads)

    gold_dir = Path(data_config["gold_dir"])
    features_path = gold_dir / "features" / "part.parquet"
    labels_path = gold_dir / "labels" / "part.parquet"
    for path in (features_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `make build-features build-labels`.")
    source = data_config["sources"]["backblaze"]
    start_date = dt.date.fromisoformat(str(source["start_date"]))
    end_date = dt.date.fromisoformat(str(source["end_date"]))

    t0 = time.perf_counter()
    dense = build_dense_telemetry(
        features_path,
        attributes=attributes,
        start_date=start_date,
        end_date=end_date,
        drive_fraction=args.drive_fraction,
    )
    logger.info(
        "sequence_dense_built",
        drives=len(dense.drive_ids),
        days=dense.n_days,
        channels=dense.n_channels,
        gigabytes=round(dense.values.nbytes / 1e9, 2),
        elapsed_seconds=round(time.perf_counter() - t0, 1),
    )

    splits_cfg = model_config["splits"]
    train_end = dt.date.fromisoformat(str(splits_cfg["train_end"]))
    # Normalization statistics from rows the training split may use only.
    fit_normalization(dense, last_day=(train_end - start_date).days - horizon_days)

    labelled = _load_labelled_rows(labels_path, horizon_days, dense, start_date)
    parts = _split_rows(labelled, model_config, horizon_days)
    del labelled
    logger.info(
        "sequence_splits",
        **{f"{name}_rows": part.height for name, part in parts.items()},
        **{f"{name}_positive_rows": int(part["label"].sum()) for name, part in parts.items()},
    )

    max_train_rows = args.max_train_rows or seq_cfg.get("max_train_rows")
    train_positives = int((parts["train"]["label"] == 1).sum())
    max_negatives = None if max_train_rows is None else max(max_train_rows - train_positives, 0)
    train_rows = _cap_negatives(parts["train"], max_negatives, seed)
    early_rows = _cap_negatives(
        parts["validation"], seq_cfg.get("early_stopping_max_negatives"), seed
    )
    y_train = train_rows["label"].to_numpy().astype(np.float32)
    y_early = early_rows["label"].to_numpy()
    train_batch = _batch_fn(dense, train_rows, window_days)
    early_batch = _batch_fn(dense, early_rows, window_days)

    model = SequenceNet(
        n_channels=dense.n_channels,
        n_static=static_feature_count(),
        hidden_size=seq_cfg["hidden_size"],
        num_layers=seq_cfg["num_layers"],
        dropout=seq_cfg["dropout"],
    )

    def validation_average_precision(current: SequenceNet) -> float:
        scores = predict_scores(current, early_batch, early_rows.height)
        return float(average_precision_score(y_early, scores))

    def log_epoch(epoch: int, loss: float, average_precision: float) -> None:
        logger.info(
            "sequence_epoch",
            epoch=epoch,
            loss=round(loss, 5),
            validation_average_precision=round(average_precision, 5),
            elapsed_seconds=round(time.perf_counter() - t0, 1),
        )

    t_train = time.perf_counter()
    model, history = train_sequence_net(
        model,
        train_batch,
        y_train,
        evaluate=validation_average_precision,
        positive_weight_power=model_config["model"].get("positive_weight_power", 0.5),
        learning_rate=seq_cfg["learning_rate"],
        batch_size=seq_cfg["batch_size"],
        max_epochs=args.max_epochs or seq_cfg["max_epochs"],
        patience=seq_cfg["early_stopping_patience"],
        seed=seed,
        on_epoch=log_epoch,
    )
    logger.info(
        "sequence_model_trained",
        train_rows=train_rows.height,
        epochs=len(history),
        elapsed_seconds=round(time.perf_counter() - t_train, 1),
    )

    def score_split(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        part = parts[name].sort(["drive_idx", "day_idx"])
        scores = predict_scores(model, _batch_fn(dense, part, window_days), part.height)
        return part["drive_idx"].to_numpy(), part["label"].to_numpy(), scores

    t_score = time.perf_counter()
    val_ids, y_val, val_scores = score_split("validation")
    logger.info(
        "sequence_validation_scored",
        rows=len(y_val),
        seconds=round(time.perf_counter() - t_score, 1),
    )
    threshold_result = tune_drive_level_threshold(
        val_ids,
        y_val,
        val_scores,
        target_precision=model_config["threshold"]["target_precision"],
        target_recall_range=tuple(model_config["threshold"]["target_recall_range"]),
    )
    test_ids, y_test, test_scores = score_split("test")
    if args.drive_fraction >= 1.0:  # a smoke run is not a test-set evaluation
        record_test_access(variant="lstm", purpose="train_sequence_model headline test metrics")
    logger.info(
        "sequence_test_scored", rows=len(y_test), seconds=round(time.perf_counter() - t_score, 1)
    )

    threshold = threshold_result["threshold"]
    report = {
        "horizon_days": horizon_days,
        "model_type": "lstm",
        "drive_fraction": args.drive_fraction,
        "attributes": attributes,
        "window_days": window_days,
        "channels": dense.n_channels,
        "train_row_count": int(train_rows.height),
        "epochs_run": len(history),
        "history": history,
        "threshold": threshold_result,
        "validation_drive_level": drive_level_metrics(val_ids, y_val, val_scores, threshold),
        "test_drive_level": drive_level_metrics(test_ids, y_test, test_scores, threshold),
        "test_drive_ci": drive_level_bootstrap_ci(test_ids, y_test, test_scores, threshold),
        "precision_at_recall": precision_at_recall_table(
            val_ids, y_val, val_scores, test_ids, y_test, test_scores
        ),
        "test_recall_at_false_alarm": recall_at_false_alarm_table(test_ids, y_test, test_scores),
        "validation_row_auroc": compute_auroc(y_val, val_scores),
        "test_row_auroc": compute_auroc(y_test, test_scores),
    }
    audit_dir = Path(data_config["audit_dir"]) / "data_quality_reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.drive_fraction >= 1.0 else ".smoke"
    report_path = audit_dir / f"sequence_evaluation_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    scratch = Path(
        data_config.get("resource_limits", {}).get("scratch_dir") or gold_dir.parent / "tmp"
    )
    out_dir = scratch / f"sequence_model{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"scores_h{horizon_days}.npz",
        drive_ids=np.array(dense.drive_ids, dtype=str),
        validation_drive_idx=val_ids,
        validation_label=y_val,
        validation_score=val_scores,
        test_drive_idx=test_ids,
        test_label=y_test,
        test_score=test_scores,
    )
    torch.save(model.state_dict(), out_dir / f"lstm_h{horizon_days}.pt")
    logger.info(
        "sequence_report_written",
        path=str(report_path),
        validation_auroc=report["validation_drive_level"]["auroc"],
        test_auroc=report["test_drive_level"]["auroc"],
    )


if __name__ == "__main__":
    main()
