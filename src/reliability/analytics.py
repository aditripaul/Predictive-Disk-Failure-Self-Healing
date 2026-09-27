"""DuckDB-backed analytics over the gold Parquet layer
(docs/project_plan.md Phase 10 Key Task 3 "Read analytics from DuckDB";
docs/design_goal.md / docs/dataset_strategy.md: DuckDB for out-of-core SQL
over Parquet). Every other read path in this codebase goes through Polars
or the in-memory API store (`src/api/store.py`); this is the one place
DuckDB is used for real, querying Parquet files directly on disk without
loading them into process memory first - the out-of-core analytics role
the design docs describe for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

_FAILURE_RATE_BY_MODEL_FAMILY_SQL = """
    SELECT
        f.model_family AS model_family,
        COUNT(*) AS drive_day_count,
        SUM(l.label) AS failure_count,
        AVG(l.label) AS failure_rate
    FROM read_parquet(?) AS f
    JOIN read_parquet(?) AS l
      ON f.drive_id = l.drive_id AND f.date = l.date
    WHERE l.horizon_days = ? AND l.label IS NOT NULL
    GROUP BY f.model_family
    ORDER BY failure_rate DESC
"""


def failure_rate_by_model_family(
    features_path: Path, labels_path: Path, *, horizon_days: int
) -> list[dict[str, Any]]:
    """Joins gold features to gold labels for one horizon directly in
    DuckDB - both are Parquet files read from disk via `read_parquet`,
    never materialized as a Polars DataFrame in process memory - and
    returns the observed failure rate per drive model family, highest
    first. Returns `[]` if either file doesn't exist yet."""
    if not features_path.exists() or not labels_path.exists():
        return []

    with duckdb.connect(":memory:") as conn:
        cursor = conn.execute(
            _FAILURE_RATE_BY_MODEL_FAMILY_SQL,
            [str(features_path), str(labels_path), horizon_days],
        )
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()

    return [dict(zip(columns, row, strict=True)) for row in rows]
