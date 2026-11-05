"""The overwrite guard shared by train-sequence, train-merged and agent-fleet."""

from __future__ import annotations

import pytest

from pipelines import train_merged_models as tm
from src.outputs import refuse_overwrite


def test_guard_passes_when_nothing_exists(tmp_path) -> None:
    refuse_overwrite([tmp_path / "a.json", tmp_path / "b.parquet"])


def test_guard_names_every_existing_file(tmp_path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        refuse_overwrite([a, b])
    assert str(a) in str(exc.value) and str(b) not in str(exc.value)
    a.write_text("kept")
    with pytest.raises(SystemExit):
        refuse_overwrite([a])
    assert a.read_text() == "kept"  # a refusal leaves the existing file as it was


def test_guard_allows_explicit_overwrite(tmp_path) -> None:
    a = tmp_path / "a.json"
    a.write_text("{}")
    refuse_overwrite([a], overwrite=True)


def test_every_merged_output_is_guarded(tmp_path) -> None:
    """Each output file on its own is enough to stop a run."""
    paths = tm.output_paths(tmp_path, tmp_path, horizon_days=30, smoke_suffix="", tag="h30_seed0")
    refuse_overwrite(paths.values())  # nothing exists yet
    for path in paths.values():
        path.write_text("x")
        with pytest.raises(SystemExit) as exc:
            refuse_overwrite(paths.values())
        assert str(path) in str(exc.value)
        path.unlink()
