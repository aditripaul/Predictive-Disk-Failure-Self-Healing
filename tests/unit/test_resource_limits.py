from unittest.mock import MagicMock, patch

import pytest

import src.resource_limits as resource_limits
from src.resource_limits import (
    _watch_rss,
    apply_memory_limit_from_config,
    apply_memory_limit_gb,
    current_rss_bytes,
)


@pytest.fixture(autouse=True)
def _never_start_a_real_watchdog():
    """Every test runs with the watchdog thread class mocked and the
    module's "already started" state reset, so no test can ever arm a
    real RSS watchdog that would kill the pytest process."""
    with (
        patch.object(resource_limits, "_watchdog", None),
        patch("src.resource_limits.threading.Thread") as mock_thread,
    ):
        yield mock_thread


def test_current_rss_bytes_reads_a_plausible_value_on_linux():
    rss = current_rss_bytes()
    assert rss is not None
    assert 1024**2 < rss < 1024**4


def test_current_rss_bytes_is_none_when_statm_is_unreadable():
    with patch.object(resource_limits, "_STATM_PATH", "/nonexistent/statm"):
        assert current_rss_bytes() is None


def test_apply_memory_limit_gb_is_a_noop_when_none_zero_or_negative(_never_start_a_real_watchdog):
    apply_memory_limit_gb(None)
    apply_memory_limit_gb(0)
    apply_memory_limit_gb(-5)
    _never_start_a_real_watchdog.assert_not_called()


def test_apply_memory_limit_gb_starts_a_daemon_watchdog_with_the_cap_in_bytes(
    _never_start_a_real_watchdog,
):
    apply_memory_limit_gb(12)

    _never_start_a_real_watchdog.assert_called_once()
    kwargs = _never_start_a_real_watchdog.call_args.kwargs
    assert kwargs["target"] is _watch_rss
    assert kwargs["args"] == (12 * 1024**3,)
    assert kwargs["daemon"] is True
    _never_start_a_real_watchdog.return_value.start.assert_called_once()


def test_apply_memory_limit_gb_starts_only_one_watchdog_per_process(_never_start_a_real_watchdog):
    _never_start_a_real_watchdog.return_value.is_alive.return_value = True
    apply_memory_limit_gb(12)
    apply_memory_limit_gb(12)
    _never_start_a_real_watchdog.assert_called_once()


def test_apply_memory_limit_gb_is_a_noop_where_rss_cannot_be_read(_never_start_a_real_watchdog):
    with patch("src.resource_limits.current_rss_bytes", return_value=None):
        apply_memory_limit_gb(12)  # must not raise (e.g. macOS/Windows)
    _never_start_a_real_watchdog.assert_not_called()


def test_watch_rss_fires_once_rss_exceeds_the_cap():
    readings = iter([100, None, 200, 1001])
    on_exceeded = MagicMock()
    _watch_rss(1000, read_rss=lambda: next(readings), on_exceeded=on_exceeded, interval=0)
    on_exceeded.assert_called_once_with(1001, 1000)


def test_watch_rss_does_not_fire_at_exactly_the_cap():
    readings = iter([1000, 1000, 1001])
    on_exceeded = MagicMock()
    _watch_rss(1000, read_rss=lambda: next(readings), on_exceeded=on_exceeded, interval=0)
    on_exceeded.assert_called_once_with(1001, 1000)


def test_rss_ignores_virtual_reservations_that_never_become_resident():
    """The point of measuring RSS: reserving (but never touching) a large
    address range - what jemalloc, glibc arenas and thread pools do all
    the time - must not count against the cap. RLIMIT_AS counted it."""
    import mmap

    before = current_rss_bytes()
    reservation = mmap.mmap(-1, 1024**3)  # 1GB, untouched
    try:
        assert current_rss_bytes() - before < 64 * 1024**2
    finally:
        reservation.close()


def test_apply_memory_limit_from_config_reads_configs_data_yaml():
    config = {"resource_limits": {"max_memory_gb": 4}}
    with (
        patch("src.resource_limits.load_yaml", return_value=config),
        patch("src.resource_limits.apply_memory_limit_gb") as mock_apply,
    ):
        apply_memory_limit_from_config()
    mock_apply.assert_called_once_with(4)


def test_apply_memory_limit_from_config_defaults_to_none_when_section_absent():
    with (
        patch("src.resource_limits.load_yaml", return_value={}),
        patch("src.resource_limits.apply_memory_limit_gb") as mock_apply,
    ):
        apply_memory_limit_from_config()
    mock_apply.assert_called_once_with(None)


def test_rss_counts_allocated_memory_but_not_memory_mapped_file_pages(tmp_path):
    """Validation/test matrices are memory-mapped and read through; those
    file-backed pages must not count against the cap, real allocations must."""
    import numpy as np

    size = 96 * 1024**2
    path = tmp_path / "big.npy"
    np.save(path, np.ones(size // 8))

    before = current_rss_bytes()
    mapped = np.load(path, mmap_mode="r")
    assert float(mapped.sum()) == size // 8  # touches every page of the file
    after_mapping = current_rss_bytes()
    assert after_mapping - before < size // 4

    allocated = np.ones(size // 8)  # the same amount, really allocated
    assert current_rss_bytes() - after_mapping > size // 2
    del allocated, mapped
