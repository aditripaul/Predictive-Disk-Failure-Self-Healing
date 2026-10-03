"""SPIKE, not part of the pipeline: the train-split extraction done in DuckDB.

Compares against `pipelines/train_model.py`'s `--stage extract-split --split
train`, which builds the same arrays with Polars. The current extraction
measured a peak resident memory of about 6.0 GB on the real Q1 data (see
docs/model_status_and_runbook.md section 3). This script answers two
questions on that same data:

1. Does DuckDB produce the same feature matrix? (`--compare`, uncapped, on
   a table small enough to hold twice in memory.)
2. What peak memory and runtime does it need? Run without `--compare` on
   the real frame and compare the printed peak with 6.0 GB.

Design points:

- DuckDB does the filter, projection and cast, and spills to `--temp-dir`
  under `--memory-limit`. Rows stream into a Parquet part file.
- The part file is read back in batches into a preallocated float32 matrix,
  so the matrix is never duplicated, as in the Polars path.
- The negative-row sample uses a DuckDB hash, not Polars' `hash`. The kept
  negatives therefore differ from the Polars pipeline's. Both are
  deterministic; the spike does not reproduce the Polars sample row for row.
  Use `--uncapped` to compare the full split exactly.

Usage (on the real frame, from a kept work directory):

    uv run python pipelines/spike_duckdb_extract.py \
        --work-dir data/tmp/train_model_frame_XXXX --memory-limit 8GB

Writes `x_train.duckdb.npy` and `y_train.duckdb.npy` next to the frame so
the existing arrays are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.models.features import feature_matrix

FLOAT_BATCH_ROWS = 500_000
HASH_BUCKET = 1_000_000


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _sample_predicate(label_counts: dict[int, int], *, max_rows: int | None) -> str | None:
    """SQL twin of `pipelines/train_model.py::_train_sample_predicate`, with
    DuckDB's own hash (see module docstring)."""
    if max_rows is None:
        return None
    n_positive = label_counts.get(1, 0)
    n_negative = label_counts.get(0, 0)
    if n_positive + n_negative <= max_rows:
        return None
    target_negative = max(0, max_rows - n_positive)
    keep_fraction = min(1.0, target_negative / n_negative) if n_negative else 1.0
    cutoff = int(keep_fraction * HASH_BUCKET)
    # Reduce each hash before adding: two raw UINT64 hashes overflow.
    return (
        f"(label = 1 OR (hash(drive_id) % {HASH_BUCKET} + hash(date) % {HASH_BUCKET})"
        f" % {HASH_BUCKET} < {cutoff})"
    )


def extract_train_duckdb(
    frame_path: Path,
    feature_columns: list[str],
    out_dir: Path,
    *,
    memory_limit: str,
    temp_dir: Path,
    max_train_rows: int | None,
) -> dict:
    t0 = time.perf_counter()
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}'")
    con.execute("SET threads = 2")  # bounds per-thread buffers on a small machine

    source = f"read_parquet('{frame_path.as_posix()}')"
    counts = dict(
        con.execute(
            f"SELECT label, count(*) FROM {source} WHERE split = 'train' GROUP BY label"
        ).fetchall()
    )
    predicate = _sample_predicate(counts, max_rows=max_train_rows)
    where = "split = 'train'" + (f" AND {predicate}" if predicate else "")
    select_cols = ", ".join(
        f"COALESCE(CAST({_quote(c)} AS FLOAT), 0.0) AS {_quote(c)}" for c in feature_columns
    )
    part_path = out_dir / "_train_part.duckdb.parquet"
    # Stream the filtered rows to Parquet; DuckDB spills to temp_dir as needed.
    con.execute(
        f"COPY (SELECT {select_cols}, label::FLOAT AS label FROM {source} "
        f"WHERE {where} ORDER BY drive_id, date) TO '{part_path.as_posix()}' "
        "(FORMAT parquet, COMPRESSION zstd)"
    )
    t_written = time.perf_counter()

    row_count = pq.ParquetFile(part_path).metadata.num_rows
    x = np.empty((row_count, len(feature_columns)), dtype=np.float32)
    y = np.empty(row_count, dtype=np.float32)
    offset = 0
    for batch in pq.ParquetFile(part_path).iter_batches(batch_size=FLOAT_BATCH_ROWS):
        n = batch.num_rows
        for j in range(len(feature_columns)):
            x[offset : offset + n, j] = batch.column(j).to_numpy(zero_copy_only=False)
        y[offset : offset + n] = batch.column(len(feature_columns)).to_numpy(zero_copy_only=False)
        offset += n
    part_path.unlink()
    np.save(out_dir / "x_train.duckdb.npy", x)
    np.save(out_dir / "y_train.duckdb.npy", y)
    con.close()

    return {
        "row_count": row_count,
        "sample_predicate": predicate,
        "label_counts": {str(k): v for k, v in counts.items()},
        "duckdb_write_seconds": round(t_written - t0, 2),
        "total_seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }


def compare_with_polars(frame_path: Path, feature_columns: list[str], out_dir: Path) -> dict:
    """Uncapped exact comparison: the full train split, ordered by
    (drive_id, date), built with the pipeline's own `feature_matrix`."""
    reference = (
        pl.scan_parquet(frame_path)
        .filter(pl.col("split") == "train")
        .select(["drive_id", "date", *feature_columns, "label"])
        .sort(["drive_id", "date"])
        .collect()
    )
    x_ref = feature_matrix(reference, feature_columns)
    y_ref = reference["label"].cast(pl.Float32).to_numpy()
    x_duck = np.load(out_dir / "x_train.duckdb.npy")
    y_duck = np.load(out_dir / "y_train.duckdb.npy")
    same_shape = x_ref.shape == x_duck.shape and y_ref.shape == y_duck.shape
    return {
        "reference_shape": list(x_ref.shape),
        "duckdb_shape": list(x_duck.shape),
        "same_shape": same_shape,
        "features_identical": bool(same_shape and np.array_equal(x_ref, x_duck, equal_nan=True)),
        "labels_identical": bool(same_shape and np.array_equal(y_ref, y_duck, equal_nan=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=5_000_000)
    parser.add_argument("--uncapped", action="store_true", help="Keep every train row.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also check the output against Polars, uncapped. Only for data small "
        "enough to hold twice in memory.",
    )
    args = parser.parse_args()

    frame_path = args.work_dir / "frame.parquet"
    feature_columns = json.loads((args.work_dir / "frame_meta.json").read_text())["feature_columns"]
    temp_dir = args.temp_dir or args.work_dir / "_duckdb_spill"
    max_rows = None if (args.uncapped or args.compare) else args.max_train_rows
    result = extract_train_duckdb(
        frame_path,
        feature_columns,
        args.work_dir,
        memory_limit=args.memory_limit,
        temp_dir=temp_dir,
        max_train_rows=max_rows,
    )
    print(json.dumps({"duckdb": result}, indent=2))
    if args.compare:
        print(
            json.dumps(
                {"comparison": compare_with_polars(frame_path, feature_columns, args.work_dir)},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
