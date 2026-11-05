"""Guard against a pipeline silently overwriting an earlier run's outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def refuse_overwrite(paths: Iterable[Path], *, overwrite: bool = False) -> None:
    """Raises SystemExit naming every path that already exists, unless `overwrite`.

    Call it before any expensive work, so a collision fails in seconds rather
    than after a multi-hour run (or, worse, succeeds and destroys the results
    a later comparison needs).
    """
    if overwrite:
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        listed = "\n  ".join(existing)
        raise SystemExit(
            f"refusing to overwrite existing outputs:\n  {listed}\n"
            "Use a different --tag (or --out), or pass --overwrite to replace them."
        )
