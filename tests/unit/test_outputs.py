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
    a.write_text("{}")  # untouched by the refusal
    assert a.read_text() == "{}"


def test_guard_allows_explicit_overwrite(tmp_path) -> None:
    a = tmp_path / "a.json"
    a.write_text("{}")
    refuse_overwrite([a], overwrite=True)


def test_every_merged_output_is_guarded(tmp_path) -> None:
    paths = tm.output_paths(tmp_path, tmp_path, horizon_days=30, smoke_suffix="", tag="h30_seed0")
    for name, path in paths.items():
        path.write_text("x")
        with pytest.raises(SystemExit):
            refuse_overwrite(paths.values())
        path.unlink()
        refuse_overwrite(paths.values())  # clean again, so each file is checked alone
        assert name in paths
