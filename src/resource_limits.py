"""Process-wide memory cap (docs/dataset_strategy.md section 5's
RAM-conscious pipeline discipline, made an enforced hard limit rather than
just careful code). Every pipeline/service entry point calls
`apply_memory_limit_from_config()` once, as early as possible in `main()`:
it reads `configs/data.yaml`'s `resource_limits.max_memory_gb` and starts a
watchdog that kills *this process* if its **resident** memory (RSS - the
physical RAM it actually occupies, counting what it allocated and not
memory-mapped file pages) goes over the cap.

If a fleet larger than expected (or a bug) would blow past the configured
cap, the process is stopped with a clear `memory_limit_exceeded` log line
instead of the OS silently OOM-killing an unrelated process or thrashing
the whole machine into swap.

**Why resident memory, not `RLIMIT_AS`.** This used to be enforced with
`RLIMIT_AS` (`ulimit -v`), which caps *virtual address space*. That is
the wrong quantity for this codebase: Polars (jemalloc), numpy/glibc
malloc (per-thread arenas), and LightGBM/OpenMP worker threads all
reserve address space far beyond what they ever touch, and jemalloc keeps
freed ranges mapped for reuse. On real data, pipelines aborted on
allocations of a few hundred KB while measured peak RSS was ~6-10GB under
a 20GB cap - the cap was tripping on reservations, not on memory use, and
the gap grows with core count. Watching RSS measures what the cap is
meant to protect: the machine's physical RAM.

It's a polling watchdog (every `_POLL_INTERVAL_SECONDS`), so a single very
fast allocation burst can overshoot the cap briefly before the process is
stopped - size the cap with some headroom below physical RAM.

Linux only (reads `/proc/self/statm`); elsewhere it's a no-op, logged
once as a warning.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

from src.config import load_yaml
from src.logging_config import get_logger

logger = get_logger(__name__)

#: How often the watchdog samples this process's RSS.
_POLL_INTERVAL_SECONDS = 0.25

#: Exit status when the watchdog stops a process over its cap (distinct
#: from Python's 1 and from a signal death, so a parent/Makefile can tell).
MEMORY_LIMIT_EXIT_CODE = 86

_STATM_PATH = "/proc/self/statm"

_watchdog: threading.Thread | None = None


def current_rss_bytes() -> int | None:
    """This process's ANONYMOUS resident memory in bytes - what it has
    allocated and touched (numpy arrays, Polars buffers) - or `None` where it
    can't be read (non-Linux).

    File-backed pages are left out (`statm`'s "shared" field: memory-mapped
    files and shared memory). The pipelines map multi-GB `.npy` matrices
    read-only and Polars maps Parquet files; those pages are page cache the
    kernel drops under pressure, so counting them would stop a healthy run for
    reading a large file."""
    try:
        with open(_STATM_PATH) as statm:
            fields = statm.read().split()
            resident_pages, file_backed_pages = int(fields[1]), int(fields[2])
    except (OSError, ValueError, IndexError):
        return None
    return max(0, resident_pages - file_backed_pages) * os.sysconf("SC_PAGE_SIZE")


def current_rss_breakdown_gb() -> dict[str, float]:
    """RSS split by kind, from /proc/self/status: anonymous memory (what the
    process allocated: numpy arrays, Polars buffers) versus file-backed
    pages (memory-mapped files such as Parquet, which the kernel can drop
    under pressure). The watchdog's limit counts both; this says which one
    is growing, so a file-cache effect is not mistaken for allocations."""
    breakdown: dict[str, float] = {}
    try:
        with open("/proc/self/status") as status:
            for line in status:
                for key in ("RssAnon", "RssFile", "RssShmem"):
                    if line.startswith(key + ":"):
                        breakdown[key] = round(int(line.split()[1]) / 1024**2, 2)
    except (OSError, ValueError, IndexError):
        pass
    return breakdown


def _abort_over_limit(rss_bytes: int, max_bytes: int) -> None:
    logger.error(
        "memory_limit_exceeded",
        rss_gb=round(rss_bytes / 1024**3, 2),
        max_memory_gb=round(max_bytes / 1024**3, 2),
        **current_rss_breakdown_gb(),
        hint="lower the data volume / batch sizes, or raise "
        "configs/data.yaml resource_limits.max_memory_gb",
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(MEMORY_LIMIT_EXIT_CODE)


def _watch_rss(
    max_bytes: int,
    *,
    read_rss: Callable[[], int | None] = current_rss_bytes,
    on_exceeded: Callable[[int, int], None] = _abort_over_limit,
    interval: float = _POLL_INTERVAL_SECONDS,
) -> None:
    """Watchdog loop: returns only after calling `on_exceeded` (which, by
    default, never returns)."""
    while True:
        rss = read_rss()
        if rss is not None and rss > max_bytes:
            on_exceeded(rss, max_bytes)
            return
        time.sleep(interval)


def apply_memory_limit_gb(max_memory_gb: float | None) -> None:
    """Stops this process if its resident memory exceeds `max_memory_gb`
    gigabytes. `None` or `<= 0` is a no-op (no limit configured). Only the
    first call in a process starts a watchdog; later calls are no-ops."""
    global _watchdog

    if not max_memory_gb or max_memory_gb <= 0:
        return

    if current_rss_bytes() is None:
        logger.warning(
            "memory_limit_unsupported_on_this_platform",
            requested_max_memory_gb=max_memory_gb,
            reason=f"{_STATM_PATH} is unreadable (Linux only)",
        )
        return

    if _watchdog is not None and _watchdog.is_alive():
        return

    max_bytes = int(max_memory_gb * (1024**3))
    _watchdog = threading.Thread(
        target=_watch_rss, args=(max_bytes,), name="rss-memory-watchdog", daemon=True
    )
    _watchdog.start()
    logger.info("memory_limit_applied", max_memory_gb=max_memory_gb, enforced_on="rss")


def apply_memory_limit_from_config() -> None:
    """Reads `configs/data.yaml`'s `resource_limits.max_memory_gb` and
    applies it via `apply_memory_limit_gb`. Call once, as early as
    possible, at the top of every pipeline/service entry point's
    `main()` - before any data is loaded."""
    data_config = load_yaml("data.yaml")
    max_memory_gb = data_config.get("resource_limits", {}).get("max_memory_gb")
    apply_memory_limit_gb(max_memory_gb)
