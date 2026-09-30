"""Entry point for `make build-features`.

Pivots the silver canonical telemetry into wide form and computes every gold
feature family: rolling aggregates, deltas/slopes/acceleration, event/spike
counts, feature confidence, and lifecycle context. Writes the gold feature
table and a versioned feature registry.

The feature table is far too wide to hold in memory at fleet scale (~950
bytes per row across ~196 columns, so ~10GB for a single month of real
Backblaze data, before counting the intermediates every step allocates on
top). Splitting the work into per-drive batches (see `_write_wide_batches`)
bounds how much *live* data any one step holds, but that alone isn't
enough: Polars is built on jemalloc, which on 64-bit Linux defaults to
`retain`-ing freed virtual memory for reuse instead of returning it to the
OS (`madvise(MADV_DONTNEED)` drops the physical pages, but the address-space
mapping itself stays reserved). `RLIMIT_AS` - the memory cap this pipeline
runs under (src/resource_limits.py) - constrains mapped address space, not
resident memory, so it tracks the *high-water mark of everything this
process has ever allocated*, not what's currently live. `del` and
`gc.collect()` free the data but never lower that high-water mark, which is
why earlier rounds of narrowing joins and freeing intermediates each bought
progress to the next pipeline stage without ever fixing the crash outright.

The only thing that actually resets it is a process boundary: exiting a
process unconditionally unmaps its entire address space, regardless of what
the allocator inside it was retaining. So the batch loop below runs each
batch (and the initial pivot+split) in its own subprocess - `main()` with
`--stage ...` - orchestrated by a parent process that never itself touches
a large Polars frame, so its own address space stays flat for the whole
run.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import yaml

from src.features.confidence import add_feature_confidence
from src.features.cross_vendor import (
    add_attribute_ratios,
    add_model_family_zscores_from_stats,
    model_family_zscore_plan,
    model_family_zscore_stats,
)
from src.features.derivatives import add_acceleration, add_deltas
from src.features.events import add_positive_day_counts, add_spike_counts, add_zero_to_nonzero_flags
from src.features.lifecycle import add_lifecycle_features
from src.features.pivot import pivot_badness_wide
from src.features.registry import build_registry
from src.features.windows import add_rolling_aggregates
from src.logging_config import configure_logging, get_logger
from src.resource_limits import apply_memory_limit_from_config

DATA_CONFIG_PATH = Path("configs/data.yaml")
FEATURES_CONFIG_PATH = Path("configs/features.yaml")

#: Target rows per feature batch. ~2M rows x ~196 columns is ~2GB of final
#: feature table per batch, which leaves plenty of room under a 20GB cap for
#: the intermediates each feature family allocates while computing.
DEFAULT_BATCH_TARGET_ROWS = 2_000_000

#: Manifest/result filenames inside the run's temp directory - how the
#: subprocess stages hand small pieces of information back to the
#: orchestrator, since they can't return Python values across a process
#: boundary.
PREPARE_MANIFEST_NAME = "prepare_manifest.json"
FINALIZE_RESULT_NAME = "finalize_result.json"

logger = get_logger(__name__)


def _log_stage(stage: str, started_at: float, **fields: object) -> None:
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(f"build_gold_features_{stage}", elapsed_seconds=elapsed_seconds, **fields)


def resolve_feature_plan(
    columns: list[str], features_config: dict
) -> tuple[list[str], tuple[int, ...], dict[str, float]]:
    """Which attributes/windows/spike thresholds the configured feature set
    actually applies to, given the columns present. Deterministic in the
    schema alone, so every batch resolves to the same plan."""
    available_attributes = [a for a in features_config["priority_smart_attributes"] if a in columns]
    windows_days = tuple(features_config["windows_days"])
    spike_thresholds = {
        a: t for a, t in features_config["spike_thresholds"].items() if a in available_attributes
    }
    return available_attributes, windows_days, spike_thresholds


def _write_wide_batches(
    wide: pl.DataFrame, tmp_dir: Path, *, target_rows: int
) -> list[Path]:
    """Splits `wide` into batches of whole drives, each written to its own
    Parquet file so the feature pipeline can run on one at a time.

    Batching on a hash of `drive_id` puts every row of a given drive in
    exactly one batch. That makes the batching invisible to the result:
    every feature family except the model-family z-scores is computed
    strictly per drive - `.rolling(index_column="date", group_by="drive_id")`
    in src/features/windows.py and src/features/events.py, `.over("drive_id")`
    in src/features/derivatives.py and src/features/lifecycle.py, and plain
    row-wise arithmetic in src/features/cross_vendor.py and
    src/features/confidence.py - so a drive's features depend only on that
    drive's own rows, which are all present in its batch. The z-scores are
    the single cross-drive computation, and are applied separately from
    fleet-wide statistics (see `model_family_zscore_stats`).

    Filtering preserves relative row order, so each drive's rows stay in
    the ascending date order `pivot_badness_wide` established - which is
    what `.rolling()` requires."""
    n_batches = max(1, math.ceil(wide.height / target_rows))
    batch_of_drive = pl.col("drive_id").hash(seed=0) % n_batches

    paths = []
    for index in range(n_batches):
        t0 = time.perf_counter()
        batch = wide.filter(batch_of_drive == index)
        if batch.is_empty():
            continue
        path = tmp_dir / f"wide_batch_{index}.parquet"
        batch.write_parquet(path, compression="zstd")
        _log_stage("wide_batch_written", t0, batch_index=index, row_count=batch.height)
        paths.append(path)
        del batch
        gc.collect()
    return paths


def build_gold_features(wide: pl.DataFrame, features_config: dict) -> pl.DataFrame:
    """Computes every feature family for one batch of whole drives.

    Does *not* add the model-family z-scores: those are the one feature
    that depends on other drives, so they are applied once, globally, after
    every batch has been computed (see `_stage_finalize`)."""
    available_attributes, windows_days, spike_thresholds = resolve_feature_plan(
        wide.columns, features_config
    )

    t0 = time.perf_counter()
    gold = add_rolling_aggregates(wide, available_attributes, windows_days=windows_days)
    _log_stage("rolling_aggregates_added", t0)

    t0 = time.perf_counter()
    gold = add_deltas(gold, available_attributes, windows_days=windows_days)
    _log_stage("deltas_added", t0)

    t0 = time.perf_counter()
    gold = add_acceleration(
        gold,
        available_attributes,
        short_window_days=windows_days[0],
        long_window_days=windows_days[-1],
    )
    _log_stage("acceleration_added", t0)

    t0 = time.perf_counter()
    gold = add_positive_day_counts(gold, available_attributes, windows_days=windows_days)
    _log_stage("positive_day_counts_added", t0)

    t0 = time.perf_counter()
    if spike_thresholds:
        gold = add_spike_counts(gold, spike_thresholds, windows_days=windows_days)
    _log_stage("spike_counts_added", t0)

    t0 = time.perf_counter()
    gold = add_zero_to_nonzero_flags(gold, available_attributes)
    _log_stage("zero_to_nonzero_flags_added", t0)

    t0 = time.perf_counter()
    gold = add_lifecycle_features(gold)
    _log_stage("lifecycle_features_added", t0)

    t0 = time.perf_counter()
    gold = add_attribute_ratios(gold)
    _log_stage("attribute_ratios_added", t0)

    t0 = time.perf_counter()
    gold = add_feature_confidence(
        gold,
        available_attributes,
        recency_tau_hours=features_config["feature_confidence"]["recency_tau_hours"],
    )
    _log_stage(
        "feature_confidence_added", t0, row_count=gold.height, column_count=len(gold.columns)
    )

    return gold


def _load_configs() -> tuple[dict, dict]:
    data_config = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    features_config = yaml.safe_load(FEATURES_CONFIG_PATH.read_text())
    return data_config, features_config


def _stage_prepare(tmp_dir: Path) -> None:
    """Subprocess stage: reads Silver, pivots to wide, and splits into
    per-drive batches. Runs in its own process so that once it exits, its
    entire address space - including whatever jemalloc is still holding
    onto from reading ~62M rows of canonical_long/drive_day and pivoting
    them - is actually released back to the OS, not just logically freed
    within a process that keeps running."""
    configure_logging()
    apply_memory_limit_from_config()
    data_config, _ = _load_configs()

    silver_dir = Path(data_config["silver_dir"])
    canonical_path = silver_dir / "canonical_telemetry" / "part.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(f"{canonical_path} not found; run `make build-silver` first.")
    drive_day_path = silver_dir / "drive_day" / "part.parquet"
    if not drive_day_path.exists():
        raise FileNotFoundError(f"{drive_day_path} not found; run `make build-silver` first.")

    t0 = time.perf_counter()
    canonical_long = pl.read_parquet(canonical_path)
    _log_stage("canonical_telemetry_read", t0, row_count=canonical_long.height)

    t0 = time.perf_counter()
    drive_day = pl.read_parquet(drive_day_path)
    _log_stage("drive_day_read", t0, row_count=drive_day.height)

    t0 = time.perf_counter()
    wide = pivot_badness_wide(canonical_long, drive_day=drive_day)
    _log_stage("pivoted_to_wide", t0, row_count=wide.height, column_count=len(wide.columns))

    del canonical_long, drive_day
    gc.collect()

    batch_target_rows = int(
        data_config.get("resource_limits", {}).get(
            "feature_batch_target_rows", DEFAULT_BATCH_TARGET_ROWS
        )
    )
    t0 = time.perf_counter()
    wide_batches = _write_wide_batches(wide, tmp_dir, target_rows=batch_target_rows)
    _log_stage("wide_batched", t0, batch_count=len(wide_batches))
    if not wide_batches:
        raise ValueError("Pivoted feature frame is empty; nothing to build features from.")

    (tmp_dir / PREPARE_MANIFEST_NAME).write_text(
        json.dumps({"wide_batch_paths": [str(p) for p in wide_batches]})
    )


def _stage_batch(wide_batch_path: Path, gold_batch_path: Path) -> None:
    """Subprocess stage: computes every feature family for one batch of
    whole drives. Deliberately one process per batch, not one process for
    the whole loop - see the module docstring for why reusing a single
    process across batches would let jemalloc's retained memory pile up
    across them exactly as it does within a single batch's joins."""
    configure_logging()
    apply_memory_limit_from_config()
    _, features_config = _load_configs()

    t0 = time.perf_counter()
    batch = pl.read_parquet(wide_batch_path)
    gold_batch = build_gold_features(batch, features_config)
    gold_batch.write_parquet(gold_batch_path, compression="zstd")
    _log_stage(
        "gold_batch_built",
        t0,
        row_count=gold_batch.height,
        column_count=len(gold_batch.columns),
    )


def _stage_finalize(tmp_dir: Path, gold_batch_paths: list[Path], out_path: Path) -> None:
    """Subprocess stage: applies the model-family z-scores (the one feature
    that needs the whole fleet) and writes the final output, one batch at a
    time via a Parquet row-group writer rather than concatenating the
    batches into a single DataFrame first - the full feature table is
    ~10GB for one month of real data, and nothing in this pipeline ever
    needs it all at once."""
    configure_logging()
    apply_memory_limit_from_config()
    _, features_config = _load_configs()
    attributes, windows_days, spike_thresholds = resolve_feature_plan(
        pl.scan_parquet(gold_batch_paths[0]).collect_schema().names(), features_config
    )

    # The model-family z-scores are the only feature that depends on drives
    # outside its own batch, so they are the only thing that needs a
    # fleet-wide pass. The statistics themselves are tiny (one row per
    # model family) and are aggregated straight off the batch files, so
    # nothing has to be materialized to compute them.
    t0 = time.perf_counter()
    batch_columns = pl.scan_parquet(gold_batch_paths[0]).collect_schema().names()
    zscore_plan = model_family_zscore_plan(batch_columns, attributes, windows_days=windows_days)
    zscore_stats = (
        model_family_zscore_stats(
            pl.concat([pl.scan_parquet(p) for p in gold_batch_paths], how="vertical"),
            zscore_plan,
        )
        if zscore_plan
        else None
    )
    _log_stage(
        "model_family_zscore_stats_computed",
        t0,
        zscore_count=len(zscore_plan),
        family_count=0 if zscore_stats is None else zscore_stats.height,
    )

    t0 = time.perf_counter()
    row_count = 0
    column_count = 0
    writer = None
    try:
        for path in gold_batch_paths:
            batch = pl.read_parquet(path)
            if zscore_plan and zscore_stats is not None:
                batch = add_model_family_zscores_from_stats(batch, zscore_stats, zscore_plan)
            row_count += batch.height
            column_count = len(batch.columns)
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            del batch, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    _log_stage("gold_batches_written", t0, row_count=row_count, column_count=column_count)

    (tmp_dir / FINALIZE_RESULT_NAME).write_text(
        json.dumps(
            {
                "row_count": row_count,
                "column_count": column_count,
                "attributes": attributes,
                "windows_days": list(windows_days),
                "spike_thresholds": spike_thresholds,
            }
        )
    )


def _run_stage(*args: str) -> None:
    """Runs this same script as a fresh subprocess for one stage. Each
    stage gets a brand-new process (and therefore a brand-new address
    space) regardless of what the previous stage's allocator left mapped -
    see the module docstring for why that's the actual fix, not just
    another round of freeing things sooner within one long-lived process."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "batch", "finalize"])
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--wide-batch-path", type=Path)
    parser.add_argument("--gold-batch-path", type=Path)
    parser.add_argument("--gold-batch-paths", type=Path, nargs="*")
    parser.add_argument("--out-path", type=Path)
    args = parser.parse_args()

    if args.stage == "prepare":
        _stage_prepare(args.tmp_dir)
        return
    if args.stage == "batch":
        _stage_batch(args.wide_batch_path, args.gold_batch_path)
        return
    if args.stage == "finalize":
        _stage_finalize(args.tmp_dir, args.gold_batch_paths, args.out_path)
        return

    # No --stage: this is the top-level orchestrator. It never touches a
    # large Polars frame itself - only the subprocesses it spawns do - so
    # its own address space stays flat for the whole run regardless of how
    # many batches there are.
    configure_logging()
    data_config, features_config = _load_configs()

    gold_dir = Path(data_config["gold_dir"]) / "features"
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "part.parquet"

    tmp_dir = Path(tempfile.mkdtemp(prefix="build_gold_features_"))
    try:
        _run_stage("--stage", "prepare", "--tmp-dir", str(tmp_dir))
        manifest = json.loads((tmp_dir / PREPARE_MANIFEST_NAME).read_text())
        wide_batch_paths = [Path(p) for p in manifest["wide_batch_paths"]]

        gold_batch_paths = []
        for index, wide_batch_path in enumerate(wide_batch_paths):
            gold_batch_path = tmp_dir / f"gold_batch_{index}.parquet"
            _run_stage(
                "--stage",
                "batch",
                "--wide-batch-path",
                str(wide_batch_path),
                "--gold-batch-path",
                str(gold_batch_path),
            )
            gold_batch_paths.append(gold_batch_path)
            logger.info(
                "build_gold_features_batch_done",
                batch_index=index,
                batch_count=len(wide_batch_paths),
            )

        _run_stage(
            "--stage",
            "finalize",
            "--tmp-dir",
            str(tmp_dir),
            "--gold-batch-paths",
            *[str(p) for p in gold_batch_paths],
            "--out-path",
            str(out_path),
        )
        result = json.loads((tmp_dir / FINALIZE_RESULT_NAME).read_text())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    attributes = result["attributes"]
    windows_days = tuple(result["windows_days"])
    spike_thresholds = result["spike_thresholds"]

    registry = build_registry(attributes, windows_days, spike_thresholds)
    registry_dir = Path(data_config["audit_dir"]) / "feature_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"v{features_config['version']}.json"
    registry_path.write_text(
        json.dumps([e.model_dump() for e in registry], indent=2, default=str)
    )

    logger.info(
        "gold_features_written",
        row_count=result["row_count"],
        column_count=result["column_count"],
        path=str(out_path),
    )
    logger.info(
        "feature_registry_written", entry_count=len(registry), path=str(registry_path)
    )


if __name__ == "__main__":
    main()
