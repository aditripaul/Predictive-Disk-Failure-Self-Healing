from pathlib import Path

import polars as pl

from tests.golden.generate_golden_dataset import (
    HISTORY_DAYS,
    N_FAILING_DRIVES,
    N_HEALTHY_DRIVES,
    build_golden_dataset,
)


def test_golden_dataset_builds_in_memory():
    df = build_golden_dataset()
    expected_rows = (N_HEALTHY_DRIVES + N_FAILING_DRIVES) * HISTORY_DAYS
    assert df.height == expected_rows
    assert df["drive_id"].n_unique() == N_HEALTHY_DRIVES + N_FAILING_DRIVES
    assert set(df.columns) >= {
        "drive_id",
        "date",
        "reallocated_sector_count",
        "current_pending_sector_count",
        "power_on_hours",
        "label",
    }


def test_golden_dataset_parquet_roundtrip(tmp_path: Path):
    df = build_golden_dataset()
    out = tmp_path / "golden.parquet"
    df.write_parquet(out, compression="zstd")

    loaded = pl.read_parquet(out)
    assert loaded.equals(df)


def test_failing_drives_have_positive_labels():
    df = build_golden_dataset()
    failing = df.filter(pl.col("drive_id").str.starts_with("failing-"))
    assert failing["label"].sum() > 0

    healthy = df.filter(pl.col("drive_id").str.starts_with("healthy-"))
    assert healthy["label"].sum() == 0
