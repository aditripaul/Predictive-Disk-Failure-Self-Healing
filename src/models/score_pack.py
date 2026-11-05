"""Pack a `train-merged` score table into parts small enough for GitHub, and rebuild it.

The table (`merged_scores_h*.parquet`) is about 1 GB because every model's score
is a high-entropy float. A frozen run needs those scores, but a file over
GitHub's 100 MB limit cannot be committed, and the table holds redundancy:

* the rank-blend columns are a deterministic function of the member columns and
  the members' VALIDATION scores (`src/models/ensemble.py`), so they are dropped
  and recomputed;
* the sequence nets score in float32, so their float64 columns are stored as
  float32, which is exact (checked, never assumed).

Nothing else is changed: `rebuild` returns a frame equal to the original, column
for column. `pack` verifies that before it reports success.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from src.models.ensemble import rank_blend

MANIFEST = "MANIFEST.json"
PART_PATTERN = "part-{index:02d}.parquet"


def _float32_exact(column: pl.Series) -> bool:
    values = column.to_numpy()
    return bool(
        np.array_equal(values, values.astype(np.float32).astype(np.float64), equal_nan=True)
    )


def add_blends(
    table: pl.DataFrame, ensembles: dict[str, list[str]], *, split_column: str = "split"
) -> pl.DataFrame:
    """Adds one rank-blend column per ensemble, in the given order.

    As `train_merged_models` builds them: each member's score is ranked against
    that member's VALIDATION scores, and the blend is the mean of the ranks,
    summed in member order (floating-point addition is not associative)."""
    validation = table.filter(pl.col(split_column) == "validation")
    out = table
    for name, members in ensembles.items():
        reference = {m: validation[m].to_numpy() for m in members}
        parts = []
        for split in table[split_column].unique(maintain_order=True):
            rows = table.filter(pl.col(split_column) == split)
            blended = rank_blend(reference, {m: rows[m].to_numpy() for m in members}, members)
            parts.append(pl.Series(name, blended))
        out = out.with_columns(pl.concat(parts))
    return out


def pack(
    table: pl.DataFrame,
    out_dir: Path,
    *,
    ensembles: dict[str, list[str]],
    float32_columns: Sequence[str],
    rows_per_part: int,
) -> dict:
    """Writes `part-NN.parquet` files and `MANIFEST.json` to `out_dir`, then proves
    that `rebuild` reproduces `table` exactly. Raises if it does not, or if a
    column listed in `float32_columns` is not exactly representable in float32."""
    for column in float32_columns:
        if not _float32_exact(table[column]):
            raise ValueError(f"column {column!r} is not exactly representable in float32")
    if missing := [m for members in ensembles.values() for m in members if m not in table.columns]:
        raise ValueError(f"ensemble members not in the table: {sorted(set(missing))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = [c for c in table.columns if c not in ensembles]
    slim = table.select(kept).with_columns(*(pl.col(c).cast(pl.Float32) for c in float32_columns))
    parts = []
    for index, start in enumerate(range(0, slim.height, rows_per_part)):
        name = PART_PATTERN.format(index=index)
        chunk = slim.slice(start, rows_per_part)
        chunk.write_parquet(out_dir / name, compression="zstd", compression_level=19)
        parts.append({"file": name, "rows": chunk.height})
    manifest = {
        "columns": table.columns,
        "float32_columns": list(float32_columns),
        "dtypes": {c: str(t) for c, t in table.schema.items()},
        "ensembles": ensembles,
        "rows": table.height,
        "parts": parts,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))

    rebuilt = rebuild(out_dir)
    if not rebuilt.equals(table):
        raise AssertionError("rebuilt table differs from the original; nothing should be committed")
    return manifest


def rebuild(pack_dir: Path) -> pl.DataFrame:
    """The original table, rebuilt from `pack_dir`: parts concatenated, float32
    columns widened back to float64, blend columns recomputed, original column
    order and dtypes restored."""
    manifest = json.loads((pack_dir / MANIFEST).read_text())
    frames = [pl.read_parquet(pack_dir / part["file"]) for part in manifest["parts"]]
    table = pl.concat(frames, how="vertical")
    if table.height != manifest["rows"]:
        raise ValueError(f"{table.height} rows rebuilt, manifest says {manifest['rows']}")
    table = table.with_columns(*(pl.col(c).cast(pl.Float64) for c in manifest["float32_columns"]))
    table = add_blends(table, manifest["ensembles"])
    return table.select(manifest["columns"])
