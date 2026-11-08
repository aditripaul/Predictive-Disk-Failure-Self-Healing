"""Write paths inside this repository as repo-relative paths.

Logs and reports are committed, and a path that starts at the filesystem root
names the machine and user that produced it (`scripts/check_no_absolute_paths.py`
refuses such a commit). `relativize_text` rewrites every mention of the
repository root in a piece of text: `<root>/data/x` becomes `data/x` and the
root on its own becomes `./`.

Standard library only, so `scripts/scrub_paths.py` and the git hooks can use it
without the project's environment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root_spellings() -> list[str]:
    """The repository root as it can appear in text: resolved, as reached through
    a symlink (the unresolved module path, `$PWD`), longest first so a longer
    spelling is replaced before a shorter one it contains."""
    spellings = {str(REPO_ROOT), os.path.abspath(Path(__file__).parents[1])}
    pwd = os.environ.get("PWD")
    if pwd and os.path.isabs(pwd):
        try:
            if Path(pwd).resolve() == REPO_ROOT:
                spellings.add(pwd.rstrip("/"))
        except OSError:
            pass
    return sorted((s for s in spellings if s not in ("", "/")), key=len, reverse=True)


def _pattern(spellings: list[str]) -> re.Pattern[str]:
    # The root, then either a path below it or the end of the path. A root that
    # is only the start of a longer name (`repo.git-old`, `repo2`) is left alone;
    # a full stop that ends a sentence still counts as the end of the path.
    roots = "|".join(re.escape(s) for s in spellings)
    return re.compile(rf"(?:{roots})(?:/(?=[^\s/])|(?![\w\-/]|\.[\w\-]))")


_PATTERN = _pattern(_root_spellings())


def relativize_text(text: str) -> str:
    """`text` with every mention of the repository root made repo-relative."""
    return _PATTERN.sub(lambda m: "" if m.group(0).endswith("/") else "./", text)


def repo_relative(path: str | os.PathLike[str]) -> str:
    """`path` relative to the repository root when it is inside it, else as given."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO_ROOT).as_posix() or "."
        except ValueError:
            return relativize_text(str(path))
    return str(path)
