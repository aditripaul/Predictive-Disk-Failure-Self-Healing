"""Dataset versioning (docs/dataset_strategy.md section 4,
`audit/dataset_versions/`; docs/project_plan.md Phase 6 Key Task 7 "Store
dataset version, feature registry version, threshold policy, and model
card").

A dataset version is an immutable, timestamped snapshot of exactly what
went into one `make build-labels` run: which gold features/labels files,
how many rows, what date range, what config versions, and a content hash
of each file - so a model card's "trained on dataset_version X" claim
points at something concrete and reproducible rather than "whatever
happened to be on disk at training time."
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

DATASET_VERSIONS_SUBDIR = "dataset_versions"


def _content_hash(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """Streams the file through the hash in fixed-size chunks rather than
    `path.read_bytes()`-ing it whole - `gold_features`' `part.parquet` is
    ~10GB for one month of real data, and this is called on it after the
    label table (already several GB) is already resident, so loading the
    entire feature file into one more Python `bytes` object here would add
    another ~10GB on top for no reason other than computing a hash."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_dataset_version_record(
    *,
    features_path: Path,
    labels: pl.DataFrame,
    labels_path: Path,
    feature_registry_version: int,
    model_config_version: int,
) -> dict[str, Any]:
    date_range: list[str | None] = [None, None]
    if "date" in labels.columns and labels.height > 0:
        date_range = [str(labels["date"].min()), str(labels["date"].max())]

    split_counts = (
        labels.group_by("split").len().sort("split").to_dicts() if "split" in labels.columns else []
    )

    return {
        "version_id": dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ"),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "feature_registry_version": feature_registry_version,
        "model_config_version": model_config_version,
        "gold_features": {
            "path": str(features_path),
            "content_hash": _content_hash(features_path),
        },
        "gold_labels": {
            "path": str(labels_path),
            "row_count": labels.height,
            "content_hash": _content_hash(labels_path),
            "date_range": date_range,
            "split_counts": split_counts,
        },
    }


def write_dataset_version(record: dict[str, Any], audit_dir: Path) -> Path:
    versions_dir = audit_dir / DATASET_VERSIONS_SUBDIR
    versions_dir.mkdir(parents=True, exist_ok=True)
    path = versions_dir / f"{record['version_id']}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def latest_dataset_version(audit_dir: Path) -> dict[str, Any] | None:
    """Reads back the most recently written dataset version record (by
    `version_id`, which sorts chronologically) - used by
    `pipelines/train_model.py` to stamp the model card with the dataset
    version it was actually trained against."""
    versions_dir = audit_dir / DATASET_VERSIONS_SUBDIR
    if not versions_dir.exists():
        return None
    files = sorted(versions_dir.glob("*.json"))
    if not files:
        return None
    result: dict[str, Any] = json.loads(files[-1].read_text())
    return result
