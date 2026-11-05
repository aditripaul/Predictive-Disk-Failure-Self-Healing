"""Packing a train-merged score table for git and rebuilding it exactly."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.models.ensemble import rank_blend
from src.models.score_pack import pack, rebuild

ENSEMBLES = {"blend_a": ["m1", "m3", "m2"], "blend_b": ["m1", "m2", "m3", "m4"]}


def _table(n_per_split: int = 500, seed: int = 0) -> pl.DataFrame:
    """The shape `train_merged_models` writes: splits stacked, blends built as it builds them."""
    rng = np.random.default_rng(seed)
    splits = {}
    for split in ("validation", "test"):
        n = n_per_split
        splits[split] = pl.DataFrame(
            {
                "drive_id": [f"D{i % 97}" for i in range(n)],
                "date": [dt.date(2026, 3, 1) + dt.timedelta(days=i % 30) for i in range(n)],
                "label": rng.integers(0, 2, n).astype(np.int8),
                "m1": rng.random(n),  # float64, not exactly float32
                "m2": rng.random(n).astype(np.float32).astype(np.float64),  # float32-exact
                "m3": rng.random(n).astype(np.float32).astype(np.float64),
                "m4": rng.random(n),
            }
        )
    reference = {m: splits["validation"][m].to_numpy() for m in ("m1", "m2", "m3", "m4")}
    out = []
    for split, frame in splits.items():
        for name, members in ENSEMBLES.items():
            target = {m: frame[m].to_numpy() for m in members}
            frame = frame.with_columns(pl.Series(name, rank_blend(reference, target, members)))
        out.append(frame.with_columns(pl.lit(split).alias("split")))
    return pl.concat(out)


def test_pack_then_rebuild_reproduces_the_table_exactly(tmp_path) -> None:
    table = _table()
    manifest = pack(
        table, tmp_path, ensembles=ENSEMBLES, float32_columns=["m2", "m3"], rows_per_part=300
    )
    assert len(manifest["parts"]) == 4  # 1000 rows in parts of 300
    assert rebuild(tmp_path).equals(table)
    stored = pl.read_parquet(tmp_path / "part-00.parquet")
    assert "blend_a" not in stored.columns and stored.schema["m2"] == pl.Float32


def test_pack_refuses_a_column_that_is_not_exact_in_float32(tmp_path) -> None:
    with pytest.raises(ValueError, match="m1"):
        pack(_table(), tmp_path, ensembles=ENSEMBLES, float32_columns=["m1"], rows_per_part=300)


def test_rebuild_fails_loudly_if_a_part_is_missing(tmp_path) -> None:
    pack(_table(), tmp_path, ensembles=ENSEMBLES, float32_columns=["m2"], rows_per_part=300)
    (tmp_path / "part-01.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        rebuild(tmp_path)


def test_a_changed_score_changes_the_rebuilt_blend(tmp_path) -> None:
    """The blend is recomputed, not stored: it must follow its members."""
    table = _table()
    pack(table, tmp_path, ensembles=ENSEMBLES, float32_columns=["m2"], rows_per_part=1000)
    part = pl.read_parquet(tmp_path / "part-00.parquet")
    tampered = part.with_columns(pl.col("m1") * 0.5)
    tampered.write_parquet(tmp_path / "part-00.parquet")
    assert not rebuild(tmp_path).equals(table)


def test_script_packs_rebuilds_and_refuses_to_overwrite(tmp_path) -> None:
    from scripts.pack_merged_scores import main

    table = _table()
    scores = tmp_path / "scores.parquet"
    table.write_parquet(scores)
    ensembles = [f"{n}={','.join(m)}" for n, m in ENSEMBLES.items()]
    flags = [x for e in ensembles for x in ("--ensemble", e)]
    pack_dir, rebuilt = tmp_path / "pack", tmp_path / "rebuilt.parquet"

    main(["pack", str(scores), str(pack_dir), "--float32", "m2", "m3", *flags])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["pack", str(scores), str(pack_dir), "--float32", "m2", "m3", *flags])
    main(["rebuild", str(pack_dir), str(rebuilt)])
    assert pl.read_parquet(rebuilt).equals(table)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["rebuild", str(pack_dir), str(rebuilt)])
