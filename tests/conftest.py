"""Shared fixtures."""

from __future__ import annotations

import pytest

from src.models import access_log


@pytest.fixture(autouse=True)
def _isolate_test_access_log(tmp_path_factory: pytest.TempPathFactory, monkeypatch):
    """Send every test-split access recorded during a test to a throwaway file.

    `data/audit/test_access_log.jsonl` is the append-only count of real
    test-split looks (src/models/access_log.py), which a reported result is
    discounted by. Unit tests that exercise the experiment, merged or sequence
    pipelines call `record_test_access` with its default path and would
    otherwise inflate that count on every `make test`.

    The default is a keyword default fixed when the function is defined, so
    patching the module constant would not reach it; the defaults are replaced.
    """
    log = tmp_path_factory.mktemp("access_log") / "test_access_log.jsonl"
    monkeypatch.setitem(access_log.record_test_access.__kwdefaults__, "log_path", log)
    monkeypatch.setattr(access_log.count_test_accesses, "__defaults__", (log,))
