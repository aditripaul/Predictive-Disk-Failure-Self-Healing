"""Data profiling for freshly ingested Bronze partitions.

Produces the row-count / missingness / failure-rate report required by
Phase 1's exit criteria, written to data/audit/data_quality_reports/.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl


def profile_bronze_dataset(bronze_root: Path) -> dict:
    parquet_glob = str(bronze_root / "**" / "*.parquet")
    lf = pl.scan_parquet(parquet_glob)
    schema_names = lf.collect_schema().names()

    n_rows = lf.select(pl.len()).collect().item()

    null_pct: dict[str, float] = {}
    if n_rows > 0:
        null_counts = lf.select(
            [pl.col(c).null_count().alias(c) for c in schema_names]
        ).collect()
        null_pct = {c: round(null_counts[c][0] / n_rows, 4) for c in schema_names}

    report: dict = {
        "bronze_root": str(bronze_root),
        "profiled_at": dt.datetime.now(dt.UTC).isoformat(),
        "row_count": n_rows,
        "columns": schema_names,
        "null_percentage_by_column": null_pct,
    }

    if "model" in schema_names or "drive_model" in schema_names:
        model_col = "drive_model" if "drive_model" in schema_names else "model"
        report["drive_model_distribution"] = (
            lf.group_by(model_col).agg(pl.len().alias("count")).collect().to_dicts()
        )

    if "failure" in schema_names and n_rows > 0:
        n_failures = lf.select(pl.col("failure").sum()).collect().item()
        report["failure_rate"] = round((n_failures or 0) / n_rows, 6)

    return report


def write_profiling_report(report: dict, audit_dir: Path, name: str) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / f"{name}_profile.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return out_path
