"""Row and positive counts per split at each stage of the 30-day pipeline.

Run on the machine that holds the data, after `make train` with
`--keep-work-dir`:

    uv run python scripts/diagnose_split_counts.py

It answers two questions from the evaluation report:
  1. Does any raw `failure` flag reach the model's input frame?
  2. Where are the positives lost between labeling and evaluation?

Stages compared, per split (horizon 30, label not null):
  labels        gold label table (what the labeler produced)
  frame         train_model's assembled training frame
  eval_ids      the row ids the evaluation scored (work dir *_ids.parquet)
Negatives should match between stages; any mismatch in positives is the
lead to follow.
"""

from __future__ import annotations

import glob
from pathlib import Path

import polars as pl

HORIZON = 30
LABELS = Path("data/gold/labels")
TMP = Path("data/tmp")


def _scan(path: Path) -> pl.LazyFrame | None:
    if path.is_file():
        return pl.scan_parquet(path)
    if path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"))
    return None


def _counts(lf: pl.LazyFrame, split_col: str, label_col: str) -> pl.DataFrame:
    return (
        lf.filter(pl.col(label_col).is_not_null())
        .group_by(split_col)
        .agg(
            pl.len().alias("rows"),
            (pl.col(label_col) == 1).sum().alias("positives"),
            (pl.col(label_col) == 0).sum().alias("negatives"),
        )
        .sort(split_col)
        .collect()
    )


def main() -> None:
    labels = _scan(LABELS)
    if labels is None:
        print(f"missing {LABELS}; run from the repo root after build-labels")
        return
    labels_h = labels.filter(pl.col("horizon_days") == HORIZON)
    print("== labels (30-day, label not null)")
    print(_counts(labels_h, "split", "label"))

    frames = sorted(glob.glob(str(TMP / "train_model_frame_*" / "frame.parquet")))
    if not frames:
        print("\nno frame.parquet under data/tmp; rerun training with --keep-work-dir")
        return
    frame_path = Path(frames[-1])
    frame = pl.scan_parquet(frame_path)
    schema = frame.collect_schema().names()
    print(f"\n== frame: {frame_path}")
    print(_counts(frame, "split", "label"))
    if "failure" in schema:
        print("\n== raw `failure` flag in frame (non-zero rows by split)")
        print(
            frame.group_by("split")
            .agg((pl.col("failure") != 0).sum().alias("failure_nonzero_rows"))
            .sort("split")
            .collect()
        )
    else:
        print("\nraw `failure` flag is NOT a column of the frame")

    for split in ("validation", "test"):
        ids_files = sorted(glob.glob(str(TMP / "train_model_frame_*" / f"{split}_ids.parquet")))
        if not ids_files:
            continue
        ids = pl.read_parquet(ids_files[-1]).select(["drive_id", "date"])
        joined = ids.lazy().join(
            labels_h.select(["drive_id", "date", "label"]), on=["drive_id", "date"], how="left"
        )
        print(f"\n== eval ids, {split}: rows and labels found in the label table")
        print(
            joined.select(
                pl.len().alias("rows"),
                pl.col("label").is_not_null().sum().alias("labeled"),
                (pl.col("label") == 1).sum().alias("positives"),
                (pl.col("label") == 0).sum().alias("negatives"),
            ).collect()
        )


if __name__ == "__main__":
    main()
