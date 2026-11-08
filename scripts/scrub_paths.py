"""Rewrite absolute paths inside this repository as repo-relative paths.

Third-party messages (MLflow, uv) and Python warnings print the repository's
absolute location, so a run log cannot be committed as it is. This rewrites
`<repository root>/x` as `x` and the bare root as `./`; any other absolute path
it finds is replaced by `<outside-repo>` and counted.

    cmd 2>&1 | python3 scripts/scrub_paths.py > run.log     # as a launcher writes a log
    python3 scripts/scrub_paths.py --in-place a.log b.log   # an existing log

A launcher keeps the command's exit status with `set -o pipefail`. Standard
library only, so it runs without the project's environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_repo_paths = _load("repo_paths", _HERE.parent / "src" / "repo_paths.py")
_check = _load("check_no_absolute_paths", _HERE / "check_no_absolute_paths.py")

OUTSIDE = "<outside-repo>"


def scrub_line(line: str) -> tuple[str, int]:
    """(`line` with repo paths made relative and other absolute paths replaced,
    how many other absolute paths there were). A line carrying the checker's
    waiver is only made relative."""
    line = _repo_paths.relativize_text(line)
    if _check.WAIVER in line:
        return line, 0
    line, outside = _check.ABSOLUTE_PATH.subn(OUTSIDE, line)
    return line, outside


def scrub_text(text: str) -> tuple[str, int]:
    out, outside = [], 0
    for line in text.splitlines(keepends=True):
        scrubbed, n = scrub_line(line)
        out.append(scrubbed)
        outside += n
    return "".join(out), outside


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in-place",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="rewrite these files instead of filtering stdin to stdout",
    )
    args = parser.parse_args(argv)

    if not args.in_place:
        for line in sys.stdin:  # line by line, so a running job's log stays current
            sys.stdout.write(scrub_line(line)[0])
            sys.stdout.flush()
        return 0

    for path in args.in_place:
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        scrubbed, outside = scrub_text(text)
        if scrubbed != text:
            path.write_text(scrubbed, encoding="utf-8", errors="surrogateescape")
        changed = sum(a != b for a, b in zip(text.splitlines(), scrubbed.splitlines(), strict=True))
        note = f", {outside} outside the repository" if outside else ""
        print(f"{path}: {changed} line(s) rewritten{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
