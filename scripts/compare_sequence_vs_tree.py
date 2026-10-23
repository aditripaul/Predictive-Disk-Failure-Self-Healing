"""Do the LSTM and LightGBM catch different failing drives, and does blending help?

Inputs (both written by their training runs, same horizon and splits):
- `make train TRAIN_ARGS=--save-scores`  -> <scratch>/lightgbm_scores_h<H>.parquet
- `make train-sequence`                  -> <scratch>/sequence_model/scores_h<H>.npz

Rows are matched on (drive_id, date). The LSTM file stores drive/row indices in
the order the pipeline scored them, so the (drive, date) keys are rebuilt with
the pipeline's own functions and checked against the stored labels; a mismatch
stops the run rather than comparing misaligned rows.

For each false-alarm budget it reports, among the test split's failing drives,
how many each model catches, how many both do, and how many only one does.
Then a rank-average blend, with the weight chosen on VALIDATION only (recall at
a 5e-4 false-alarm rate) and applied to test, with a paired drive-level
bootstrap interval on blend minus LightGBM. One test-split access is logged.

    uv run --extra torch python scripts/compare_sequence_vs_tree.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import yaml
from scipy.stats import rankdata

from pipelines.train_sequence_model import _load_labelled_rows, _split_rows
from src.models.access_log import record_test_access
from src.models.evaluation import compute_auroc, drive_level_table

BUDGETS = (1e-4, 5e-4, 1e-3)
SELECTION_BUDGET = 5e-4
WEIGHTS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))


def _recall_at_far(labels: np.ndarray, scores: np.ndarray, budget: float) -> tuple[float, float]:
    """`(recall, threshold)`: alert on score > threshold, the strictest cut whose
    false-alarm rate on the healthy drives is at most `budget`."""
    healthy = np.sort(scores[labels == 0])[::-1]
    allowed = int(budget * len(healthy))
    threshold = healthy[allowed] if allowed < len(healthy) else -np.inf
    failing = scores[labels == 1]
    return float((failing > threshold).mean()), float(threshold)


def _drive_table(drive_keys: np.ndarray, y: np.ndarray, scores: np.ndarray):
    labels, drive_scores = drive_level_table(drive_keys, y, scores)
    return labels, drive_scores


def _rank(scores: np.ndarray) -> np.ndarray:
    return rankdata(scores, method="average") / len(scores)


def _load_aligned(scores_dir: Path, tree_path: Path, horizon: int, model_config: dict, data_config):
    seq = np.load(scores_dir / f"scores_h{horizon}.npz")
    source = data_config["sources"]["backblaze"]
    start = dt.date.fromisoformat(str(source["start_date"]))
    dense = SimpleNamespace(drive_ids=[str(d) for d in seq["drive_ids"]])
    gold = Path(data_config["gold_dir"])
    labelled = _load_labelled_rows(gold / "labels" / "part.parquet", horizon, dense, start)
    parts = _split_rows(labelled, model_config, horizon)
    tree = pl.read_parquet(tree_path)
    out = {}
    for split in ("validation", "test"):
        rows = parts[split].sort(["drive_idx", "day_idx"])
        if not np.array_equal(rows["label"].to_numpy(), seq[f"{split}_label"]):
            raise ValueError(f"{split}: rebuilt row order does not match the stored LSTM labels")
        keyed = pl.DataFrame(
            {
                "drive_id": np.array(dense.drive_ids)[rows["drive_idx"].to_numpy()],
                "date": rows["date"],
                "label_seq": rows["label"],
                "lstm": seq[f"{split}_score"],
            }
        )
        joined = keyed.join(
            tree.filter(pl.col("split") == split).select("drive_id", "date", "label", "score"),
            on=["drive_id", "date"],
            how="inner",
        ).rename({"score": "tree"})
        if (
            joined.height != keyed.height
            or joined.height != tree.filter(pl.col("split") == split).height
        ):
            raise ValueError(f"{split}: the two models do not cover the same rows")
        if not (joined["label"] == joined["label_seq"]).all():
            raise ValueError(f"{split}: labels disagree between the two runs")
        out[split] = joined
    return out


def _paired_bootstrap(labels, a, b, budget, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        diffs.append(
            _recall_at_far(labels[pick], a[pick], budget)[0]
            - _recall_at_far(labels[pick], b[pick], budget)[0]
        )
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()
    data_config = yaml.safe_load(Path("configs/data.yaml").read_text())
    model_config = yaml.safe_load(Path("configs/model.yaml").read_text())
    scratch = Path(data_config.get("resource_limits", {}).get("scratch_dir") or "data/tmp")
    h = args.horizon_days
    aligned = _load_aligned(
        scratch / "sequence_model",
        scratch / f"lightgbm_scores_h{h}.parquet",
        h,
        model_config,
        data_config,
    )

    drive = {}
    for split, frame in aligned.items():
        ids = frame["drive_id"].to_numpy()
        y = frame["label"].to_numpy()
        for name in ("tree", "lstm"):
            labels, drive[(split, name)] = _drive_table(ids, y, frame[name].to_numpy())
        drive[(split, "labels")] = labels

    report: dict = {"horizon_days": h}
    val_labels, test_labels = drive[("validation", "labels")], drive[("test", "labels")]
    report["drive_counts"] = {
        "test_failing": int(test_labels.sum()),
        "test_healthy": int((test_labels == 0).sum()),
    }
    report["score_correlation_spearman_test_drives"] = float(
        np.corrcoef(_rank(drive[("test", "tree")]), _rank(drive[("test", "lstm")]))[0, 1]
    )

    overlap = []
    for budget in BUDGETS:
        caught = {}
        for name in ("tree", "lstm"):
            _, thr = _recall_at_far(test_labels, drive[("test", name)], budget)
            caught[name] = (drive[("test", name)] > thr) & (test_labels == 1)
        both = int((caught["tree"] & caught["lstm"]).sum())
        overlap.append(
            {
                "false_alarm_budget": budget,
                "tree_caught": int(caught["tree"].sum()),
                "lstm_caught": int(caught["lstm"].sum()),
                "both": both,
                "only_tree": int((caught["tree"] & ~caught["lstm"]).sum()),
                "only_lstm": int((caught["lstm"] & ~caught["tree"]).sum()),
                "union": int((caught["tree"] | caught["lstm"]).sum()),
            }
        )
    report["overlap_at_matched_false_alarm_rate"] = overlap

    # Blend: weight on the LSTM's rank, chosen on validation only.
    def blended(split: str, w: float) -> np.ndarray:
        return (1 - w) * _rank(drive[(split, "tree")]) + w * _rank(drive[(split, "lstm")])

    chosen = max(
        WEIGHTS,
        key=lambda w: _recall_at_far(val_labels, blended("validation", w), SELECTION_BUDGET)[0],
    )
    report["blend_weight_on_lstm_chosen_on_validation"] = float(chosen)
    rows = []
    for label, w in (("lightgbm", 0.0), ("lstm", 1.0), (f"blend w={chosen}", float(chosen))):
        scores = blended("test", w)
        row = {"model": label, "test_drive_auroc": compute_auroc(test_labels, scores)}
        for budget in BUDGETS:
            row[f"recall_at_far_{budget:g}"] = _recall_at_far(test_labels, scores, budget)[0]
        rows.append(row)
    report["test"] = rows
    report["blend_minus_lightgbm_recall_95ci"] = {
        f"{budget:g}": _paired_bootstrap(
            test_labels, blended("test", float(chosen)), blended("test", 0.0), budget
        )
        for budget in BUDGETS
    }

    record_test_access(variant="lstm+lightgbm blend", purpose="compare_sequence_vs_tree")
    out = Path(data_config["audit_dir"]) / "data_quality_reports" / "sequence_vs_tree.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
