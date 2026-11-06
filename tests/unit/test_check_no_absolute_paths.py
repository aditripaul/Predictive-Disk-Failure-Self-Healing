"""scripts/check_no_absolute_paths.py: what counts as an absolute path, and the git wiring."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_absolute_paths.py"
spec = importlib.util.spec_from_file_location("check_no_absolute_paths", SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize(
    "line",
    [
        "see /home/someone/repo/data/audit/x.json",  # abs-path-ok
        "cwd=/tmp/scratch/run1",  # abs-path-ok
        "root /Users/someone/work",  # abs-path-ok
        "socket /run/user/1000/app.sock",  # abs-path-ok
        "path /mnt/disk/data",  # abs-path-ok
        "user data.arp in the path",  # abs-path-ok
    ],
)
def test_absolute_paths_are_found(line):
    assert check.find_violations(line)


@pytest.mark.parametrize(
    "line",
    [
        "data/tmp/lightgbm_scores_h30.parquet",
        "see `./tmp/x` and ../home/y",
        "https://example.com/home/page and https://example.com/tmp/file",
        "plain text with no paths at all",
        "a fraction 1/2/3 and a ratio and/or home/tmp words",
        "/home/x/y  # abs-path-ok (shown on purpose)",
    ],
)
def test_relative_paths_urls_and_waived_lines_pass(line):
    assert not check.find_violations(line)


def test_line_numbers_and_matches_are_reported():
    text = "ok\nbad /tmp/a/b here\nok\n/home/u/x\n"  # abs-path-ok
    assert check.find_violations(text) == [(2, "/tmp/a/b"), (4, "/home/u/x")]  # abs-path-ok


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "a@b.c")
    _git(tmp_path, "config", "user.name", "n")
    return tmp_path


def _run(tmp_path, *args):
    return subprocess.run(
        ["python3", str(SCRIPT), *args], cwd=tmp_path, capture_output=True, text=True
    )


def test_staged_check_blocks_only_added_lines_and_binaries(tmp_path):
    repo = _repo(tmp_path)
    (repo / "old.md").write_text("legacy /home/u/old/path\n")  # abs-path-ok
    _git(repo, "add", "old.md")
    _git(repo, "commit", "-qm", "base")

    # An edit elsewhere in a file that already has a legacy path is not blocked.
    (repo / "old.md").write_text(
        "legacy /home/u/old/path\nnew clean line data/tmp/x\n"  # abs-path-ok
    )
    _git(repo, "add", "old.md")
    assert _run(repo).returncode == 0

    (repo / "new.md").write_text("fine\nbad /tmp/scratch/out.json\n")  # abs-path-ok
    (repo / "blob.bin").write_bytes(b"\x00\x01 /home/u/secret \x02")  # abs-path-ok
    _git(repo, "add", "new.md", "blob.bin")
    result = _run(repo)
    assert result.returncode == 1
    assert "new.md:2: /tmp/scratch/out.json" in result.stderr  # abs-path-ok
    assert "blob.bin: binary file contains an absolute path" in result.stderr
    assert "old.md" not in result.stderr


def test_message_check_ignores_comment_lines(tmp_path):
    repo = _repo(tmp_path)
    ok = repo / "ok.txt"
    ok.write_text("Subject\n\n# on branch /home/u/x\nBody with data/tmp/x\n")  # abs-path-ok
    assert _run(repo, "--message", str(ok)).returncode == 0
    bad = repo / "bad.txt"
    bad.write_text("Subject\n\nran in /tmp/work/dir\n")  # abs-path-ok
    result = _run(repo, "--message", str(bad))
    assert result.returncode == 1 and "commit message:3" in result.stderr


def test_range_check_finds_a_path_in_an_existing_commit(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("clean\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "one")
    (repo / "a.txt").write_text("clean\npath /home/u/leak\n")  # abs-path-ok
    _git(repo, "commit", "-qam", "two")
    assert _run(repo, "--range", "HEAD~1..HEAD").returncode == 1
    assert _run(repo, "--range", "HEAD~1..HEAD~1").returncode == 0
