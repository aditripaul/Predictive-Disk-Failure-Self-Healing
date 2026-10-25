"""Memory bounds for the agent fleet run.

A 10,000-cycle run exhausted 20 GB. Three compounding allocations, none of
them the model:

  * `score_snapshot` built the whole fleet as one float64 matrix - 356,696 x
    246 x 8B = 0.7 GB - and `fill_null` then `to_numpy` each copied it.
  * `apply_hysteresis` rebuilds `drive_risk_state` as `{**state, drive: ...}`
    every cycle and never removes an entry, so the dict grew one entry per
    drive ever decided and every copy went into a checkpoint.
  * the checkpoint store defaulted to `:memory:`, which retains every
    checkpoint, so that growth was held in RAM for the whole run.

These tests pin the bounds rather than the symptom: scoring is batched,
pruning holds the risk state at roughly one entry, and the default
checkpoint is a file.
"""

import datetime as dt
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.agent.fleet import DEFAULT_SCORE_BATCH_ROWS, score_snapshot


class _CountingModel:
    """Records the shape of every batch handed to it."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_proba(self, x):
        self.batch_sizes.append(x.shape[0])
        p = np.full(x.shape[0], 0.5)
        return np.column_stack([1 - p, p])


def _snapshot(n_rows: int, n_features: int = 8) -> pl.DataFrame:
    data = {"drive_id": [f"d{i}" for i in range(n_rows)], "date": [dt.date(2026, 6, 1)] * n_rows}
    for f in range(n_features):
        data[f"smart_{f}_raw"] = [float(i + f) for i in range(n_rows)]
    return pl.DataFrame(data)


def test_scoring_is_batched_so_the_peak_does_not_scale_with_the_fleet():
    model = _CountingModel()
    score_snapshot(_snapshot(1000), model, batch_rows=100)
    assert model.batch_sizes == [100] * 10


def test_batching_does_not_change_the_scores():
    """Every row is scored independently, so the batch size must be an
    implementation detail and not a result."""
    snapshot = _snapshot(500)
    unbatched = score_snapshot(snapshot, _CountingModel(), batch_rows=10_000)
    batched = score_snapshot(snapshot, _CountingModel(), batch_rows=37)
    assert unbatched["p_fail"].to_list() == batched["p_fail"].to_list()
    assert batched.height == 500


def test_the_default_batch_bounds_the_matrix_well_under_a_gigabyte():
    bytes_per_batch = DEFAULT_SCORE_BATCH_ROWS * 250 * 8
    assert bytes_per_batch < 200e6, f"{bytes_per_batch / 1e6:.0f} MB per batch"


def test_an_empty_snapshot_scores_without_calling_the_model():
    model = _CountingModel()
    scored = score_snapshot(_snapshot(0), model)
    assert scored.height == 0
    assert "p_fail" in scored.columns
    assert model.batch_sizes == []


def test_a_partial_final_batch_is_scored():
    model = _CountingModel()
    scored = score_snapshot(_snapshot(250), model, batch_rows=100)
    assert model.batch_sizes == [100, 100, 50]
    assert scored["p_fail"].null_count() == 0


def _prune(risk_state: dict, alive: set[str]) -> dict:
    """The runner's pruning step, isolated."""
    return {k: v for k, v in risk_state.items() if k in alive}


def test_pruning_holds_the_risk_state_at_the_live_pool():
    """Unpruned this grows one entry per drive ever decided, and
    apply_hysteresis copies the whole dict every cycle."""
    risk_state = {f"d{i}": {"consecutive_escalation_cycles": 1} for i in range(5000)}
    alive = {"d4999"}
    assert len(_prune(risk_state, alive)) == 1


def test_a_two_cycle_drive_leaves_nothing_behind():
    """The real pattern: WARN on the first selection, an action on the
    second, then out of the pool - so at most one drive's counters are live
    at a time regardless of how many cycles run."""
    pool = [{"drive_id": f"d{i}", "p_fail": 0.9 - i * 0.01} for i in range(50)]
    risk_state: dict[str, dict] = {}
    peak = 0
    for _ in range(100):
        if not pool:
            break
        top = max(pool, key=lambda d: d["p_fail"])["drive_id"]
        risk_state = {**risk_state, top: {"consecutive_escalation_cycles": 1}}
        # Second selection acts and removes it; first only warns.
        if top in risk_state and risk_state[top].get("seen"):
            pool[:] = [d for d in pool if d["drive_id"] != top]
        else:
            risk_state[top]["seen"] = True
        risk_state = _prune(risk_state, {d["drive_id"] for d in pool})
        peak = max(peak, len(risk_state))
    assert peak <= 2, f"risk state peaked at {peak} entries"


def test_the_default_checkpoint_is_a_file_not_memory():
    """:memory: retains every checkpoint in RAM; a 10,000-cycle run holds
    its whole history."""
    from pipelines.run_agent_fleet import parse_args

    assert parse_args([]).checkpoint_path is None  # None means a temp FILE

    captured: dict[str, str] = {}

    class _Ctx:
        def __enter__(self):
            raise RuntimeError("stop before the graph runs")

        def __exit__(self, *_):
            return False

    def fake_compiled_agent(_deps, *, checkpoint_path, plan_kwargs=None):
        captured["path"] = checkpoint_path
        return _Ctx()

    with (
        patch("pipelines.run_agent_fleet.build_fleet_dependencies") as build,
        patch("pipelines.run_agent_fleet.compiled_agent", fake_compiled_agent),
    ):
        build.return_value.snapshot_id = "s"
        build.return_value.plan_kwargs = {}
        build.return_value.candidates = []
        with pytest.raises(RuntimeError, match="stop before"):
            from pipelines.run_agent_fleet import main

            main(["--cycles", "1"])

    assert captured["path"] != ":memory:"
    assert captured["path"].endswith("checkpoints.sqlite")


def test_the_wide_scan_reaches_the_join_unfiltered():
    """The invariant that makes the snapshot bounded.

    chunked_inner_join slices its left side with left.slice(offset, n).
    Against a plain scan_parquet that pushes into the Parquet reader. Against
    a filtered scan Polars must evaluate the filter across the whole scan to
    resolve row offsets, so every slice materializes the entire window at
    full width - 43 GB for a 61-day window at 246 columns. Filtering the wide
    side is therefore not a performance detail; it is the bug.
    """
    import datetime as dt
    from pathlib import Path

    import polars as pl

    from src.agent import fleet

    captured: dict[str, object] = {}

    def fake_join(left, right, *, on, chunk_rows):
        captured["left_plan"] = left.explain() if hasattr(left, "explain") else ""
        captured["chunk_rows"] = chunk_rows
        return pl.DataFrame(
            {"drive_id": ["d0"], "date": [dt.date(2026, 5, 1)], "smart_5_raw": [1.0]}
        )

    with (
        patch.object(fleet, "chunked_inner_join", fake_join),
        patch.object(
            fleet.pl,
            "read_parquet",
            lambda *_a, **_k: pl.DataFrame({"drive_id": ["d0"], "feature_maturity": ["MATURE"]}),
        ),
        patch.object(
            fleet.pl,
            "scan_parquet",
            lambda *_a, **_k: pl.DataFrame(
                {
                    "drive_id": ["d0", "d0"],
                    "date": [dt.date(2026, 4, 1), dt.date(2026, 5, 1)],
                    "smart_5_raw": [1.0, 2.0],
                }
            ).lazy(),
        ),
    ):
        fleet.load_fleet_snapshot(
            Path("features.parquet"),
            Path("metadata.parquet"),
            start_date=dt.date(2026, 4, 1),
            end_date=dt.date(2026, 5, 31),
        )

    plan = str(captured["left_plan"]).upper()
    assert "FILTER" not in plan, f"the wide scan was filtered:\n{captured['left_plan']}"


def test_the_snapshot_chunk_is_smaller_than_the_training_join():
    """Each slice is read at full width, so the training join's 2,000,000
    would be 3.9 GB per slice for an output of one row per drive."""
    from src.agent.fleet import DEFAULT_SNAPSHOT_CHUNK_ROWS

    assert DEFAULT_SNAPSHOT_CHUNK_ROWS < 2_000_000
    assert DEFAULT_SNAPSHOT_CHUNK_ROWS * 246 * 8 < 1.5e9
