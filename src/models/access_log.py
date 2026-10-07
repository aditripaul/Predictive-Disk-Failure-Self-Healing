"""Append-only record of every computation on the test split.

Reporting test metrics for many model variants inflates the apparent result:
each look is another comparison. Logging every look makes the number of
comparisons a fact, so a reported gain can be discounted for it. Entries are
only appended, never rewritten.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path("data/audit/test_access_log.jsonl")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def record_test_access(
    variant: str,
    purpose: str,
    *,
    log_path: Path = DEFAULT_LOG,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Appends one line describing a test-split computation and returns it."""
    entry: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "variant": variant,
        "purpose": purpose,
        "git_sha": _git_sha(),
        **(extra or {}),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def count_test_accesses(log_path: Path = DEFAULT_LOG) -> int:
    """Number of recorded test-split computations; 0 when no log exists."""
    if not log_path.exists():
        return 0
    with log_path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
