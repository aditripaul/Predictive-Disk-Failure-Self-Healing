"""Fail when a commit would add an absolute filesystem path.

Machine-specific paths (a home directory, a scratch directory, a username)
make a committed file differ by machine and leak where the work was done.
Commit repo-relative paths (`data/audit/...`) instead.

    python3 scripts/check_no_absolute_paths.py                  # what is staged
    python3 scripts/check_no_absolute_paths.py --message FILE   # a commit message
    python3 scripts/check_no_absolute_paths.py --range A..B     # commits already made

Only ADDED lines of text files are checked, so an old file that already has such a
path does not block an unrelated edit. A binary file is searched whole. A line that
is meant to show a path can carry `abs-path-ok`.

Wired in as the pre-commit and commit-msg hooks by `make install-hooks`
(`scripts/githooks/`). Standard library only, so a hook can run it anywhere.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator

#: A path starting at the filesystem root under a directory that is machine- or
#: user-specific, or a username/host that appears in this repo's paths. The
#: look-behind keeps repo-relative `data/tmp/x`, `./tmp/x` and URLs out.
ABSOLUTE_PATH = re.compile(
    r"(?<![\w.~:$-])/(?:home|Users|root|tmp|mnt|media|opt|var/folders)/[^\s`'\")\]>,;|]*"
    r"|(?<![\w.~:$-])/run/user/\d+[^\s`'\")\]>,;|]*"
    r"|data\.arp"
)
WAIVER = "abs-path-ok"


def find_violations(text: str) -> list[tuple[int, str]]:
    """(line number from 1, the matched text) for every absolute path in `text`."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if WAIVER in line:
            continue
        found.extend((number, match.group(0)) for match in ABSOLUTE_PATH.finditer(line))
    return found


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """(file, line number in the new file, text) for each added line of a -U0 diff."""
    path, number = "", 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/")
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            number = int(match.group(1)) if match else 0
        elif line.startswith("+") and not line.startswith("+++"):
            yield path, number, line[1:]
            number += 1


def check_diff(diff_args: list[str], blob_prefix: str) -> list[str]:
    """Violations in the added lines of `git diff <diff_args>` and in the binary
    files it touches (searched whole, read from `<blob_prefix><path>`)."""
    problems = []
    diff = _git("diff", "--no-color", "-U0", "--text", *diff_args).decode("utf-8", "replace")
    for path, number, text in _added_lines(diff):
        if WAIVER in text:
            continue
        problems.extend(f"{path}:{number}: {m.group(0)}" for m in ABSOLUTE_PATH.finditer(text))
    numstat = _git("diff", "--numstat", "-z", *diff_args).decode("utf-8", "replace")
    binary = [
        entry.split("\t", 2)[2] for entry in numstat.split("\0") if entry.startswith("-\t-\t")
    ]
    for path in binary:
        blob = _git("show", f"{blob_prefix}{path}")
        if ABSOLUTE_PATH.search(blob.decode("latin-1")):
            problems.append(f"{path}: binary file contains an absolute path")
    return problems


def report(problems: Iterable[str]) -> int:
    problems = list(problems)
    if not problems:
        return 0
    print("Absolute paths must not be committed; use repo-relative paths:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(f"(a line that must show one can carry `{WAIVER}`)", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--message", metavar="FILE", help="check a commit message file")
    group.add_argument("--range", metavar="A..B", help="check commits already made")
    args = parser.parse_args(argv)

    if args.message:
        with open(args.message, encoding="utf-8", errors="replace") as handle:
            text = "\n".join(
                line for line in handle.read().splitlines() if not line.startswith("#")
            )
        return report(f"commit message:{n}: {path}" for n, path in find_violations(text))
    if args.range:
        end = args.range.split("..")[-1] or "HEAD"
        return report(check_diff([args.range], f"{end}:"))
    return report(check_diff(["--cached"], ":"))


if __name__ == "__main__":
    sys.exit(main())
