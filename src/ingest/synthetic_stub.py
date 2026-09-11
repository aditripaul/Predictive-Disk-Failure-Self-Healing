"""Placeholder synthetic dataset for exercising the ingestion pipeline shape.

This is intentionally trivial: a handful of drive-days with the canonical
column names, just enough to validate that downstream Bronze/Silver code can
consume a `source_dataset=synthetic` partition. The real trajectory-morphed
chaos generator (noise injection, timeline compression, correlated multi-drive
failures) is built in Phase 8.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from data_contracts.schemas import SourceDataset
from src.ingest.common import sink_partitioned_by_month, with_ingestion_metadata

N_DRIVES = 5
N_DAYS = 10


def build_synthetic_stub() -> pl.DataFrame:
    start = dt.date(2024, 1, 1)
    rows = [
        {
            "date": start + dt.timedelta(days=day),
            "drive_id": f"synthetic-{i:03d}",
            "drive_model": "SYN-MODEL-1",
            "capacity_bytes": 4_000_000_000_000,
            "failure": 0,
            "smart_5_raw": 0,
            "smart_187_raw": 0,
            "smart_188_raw": 0,
            "smart_197_raw": 0,
            "smart_198_raw": 0,
        }
        for i in range(N_DRIVES)
        for day in range(N_DAYS)
    ]
    return pl.DataFrame(rows)


def ingest_synthetic_stub(bronze_root: Path) -> list[Path]:
    lf = build_synthetic_stub().lazy()
    lf = with_ingestion_metadata(
        lf, source_dataset=SourceDataset.SYNTHETIC, source_file="synthetic_stub"
    )
    return sink_partitioned_by_month(lf, bronze_root=bronze_root)
