"""Process-wide memory cap (docs/dataset_strategy.md section 5's
RAM-conscious pipeline discipline, made an enforced hard limit rather than
just careful code). Every pipeline/service entry point calls
`apply_memory_limit_from_config()` once, as early as possible in `main()`:
it reads `configs/data.yaml`'s `resource_limits.max_memory_gb` and applies
it as a hard OS-level cap (`RLIMIT_AS`, the same mechanism as `ulimit -v`)
on *this process's* total virtual memory.

If a fleet larger than expected (or a bug) would blow past the configured
cap, the process fails at that allocation instead of the OS silently
OOM-killing an unrelated process, thrashing the whole machine into swap,
or a runaway pipeline slowly consuming everything else running on a
shared machine. **Because Polars/PyArrow allocate memory in native Rust/
C++ code, hitting the cap usually aborts the process outright (SIGABRT/
SIGSEGV) rather than raising a catchable Python `MemoryError`** - this is
a hard safety net for the rest of the system, not a graceful in-process
recovery mechanism.

POSIX only (`RLIMIT_AS` doesn't exist on Windows) - a no-op there, logged
once as a warning.
"""

from __future__ import annotations

from src.config import load_yaml
from src.logging_config import get_logger

logger = get_logger(__name__)

try:
    import resource
except ImportError:  # Windows - RLIMIT_AS/setrlimit don't exist
    resource = None  # type: ignore[assignment]


def apply_memory_limit_gb(max_memory_gb: float | None) -> None:
    """Caps this process's virtual memory (`RLIMIT_AS`) to `max_memory_gb`
    gigabytes. `None` or `<= 0` is a no-op (no limit configured) - the
    project ships with no limit by default; set
    `configs/data.yaml`'s `resource_limits.max_memory_gb` to enable one."""
    if not max_memory_gb or max_memory_gb <= 0:
        return

    if resource is None:
        logger.warning(
            "memory_limit_unsupported_on_this_platform",
            requested_max_memory_gb=max_memory_gb,
            reason="the resource module is POSIX-only (no effect on Windows)",
        )
        return

    max_bytes = int(max_memory_gb * (1024**3))
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Never try to raise an existing, more restrictive hard limit (e.g. one
    # set by a container/cgroup or a parent shell's `ulimit -Hv`) -
    # setrlimit would just fail; only ever tighten it further.
    new_hard = hard if hard != resource.RLIM_INFINITY and hard < max_bytes else max_bytes
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, new_hard))
    except (ValueError, OSError) as exc:
        logger.warning(
            "memory_limit_could_not_be_applied",
            requested_max_memory_gb=max_memory_gb,
            error=str(exc),
        )
        return

    logger.info("memory_limit_applied", max_memory_gb=max_memory_gb)


def apply_memory_limit_from_config() -> None:
    """Reads `configs/data.yaml`'s `resource_limits.max_memory_gb` and
    applies it via `apply_memory_limit_gb`. Call once, as early as
    possible, at the top of every pipeline/service entry point's
    `main()` - before any data is loaded."""
    data_config = load_yaml("data.yaml")
    max_memory_gb = data_config.get("resource_limits", {}).get("max_memory_gb")
    apply_memory_limit_gb(max_memory_gb)
