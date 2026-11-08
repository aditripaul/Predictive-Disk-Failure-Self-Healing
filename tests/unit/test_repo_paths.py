"""Paths inside the repository are written repo-relative: src/repo_paths.py, the
logging hooks in src/logging_config.py and scripts/scrub_paths.py."""

import importlib.util
import json
import logging
import warnings
from pathlib import Path

from src.logging_config import configure_logging, get_logger
from src.repo_paths import REPO_ROOT, relativize_text, repo_relative

ROOT = str(REPO_ROOT)

_spec = importlib.util.spec_from_file_location(
    "scrub_paths", REPO_ROOT / "scripts" / "scrub_paths.py"
)
scrub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrub)


def test_paths_below_the_root_become_repo_relative():
    assert relativize_text(f"path={ROOT}/data/tmp/x.parquet") == "path=data/tmp/x.parquet"
    assert relativize_text(f"{ROOT}/.venv/lib/a.py:7: Warning") == ".venv/lib/a.py:7: Warning"


def test_the_bare_root_becomes_dot_slash_also_at_a_sentence_end():
    assert relativize_text(f"found uv.lock in {ROOT}") == "found uv.lock in ./"
    assert relativize_text(f"project at {ROOT}. Exporting") == "project at ./. Exporting"


def test_a_longer_name_that_starts_with_the_root_is_left_alone():
    for text in (f"{ROOT}-old/x", f"{ROOT}2/x", f"{ROOT}.bak/x"):
        assert relativize_text(text) == text


def test_root_inside_colour_codes_is_rewritten():
    coloured = f"\x1b[36mpath\x1b[0m=\x1b[35m{ROOT}/data/tmp/w\x1b[0m"
    assert relativize_text(coloured) == "\x1b[36mpath\x1b[0m=\x1b[35mdata/tmp/w\x1b[0m"


def test_repo_relative_handles_paths_inside_outside_and_relative():
    assert repo_relative(REPO_ROOT / "data" / "audit") == "data/audit"
    assert repo_relative(REPO_ROOT) == "."
    assert repo_relative("data/tmp/x") == "data/tmp/x"


def test_structlog_events_carry_repo_relative_paths(capsys):
    configure_logging(json_output=True)
    get_logger("test.paths").info(
        "work_dir_kept", path=REPO_ROOT / "data" / "tmp" / "w", files=[f"{ROOT}/a"]
    )
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["path"] == "data/tmp/w"
    assert payload["files"] == ["a"]


def test_stdlib_log_messages_and_warnings_are_rewritten(caplog):
    configure_logging()
    with caplog.at_level(logging.INFO, logger="mlflow.test"):
        logging.getLogger("mlflow.test").info("Detected uv project at %s", ROOT)
    assert caplog.records[-1].getMessage() == "Detected uv project at ./"

    text = warnings.formatwarning("old", DeprecationWarning, f"{ROOT}/scripts/x.py", 3)
    assert text.startswith("scripts/x.py:3: DeprecationWarning")


def test_scrubber_rewrites_repo_paths_and_masks_others():
    line, outside = scrub.scrub_line(f"at {ROOT}/a and /home/someone/b\n")  # abs-path-ok
    assert line == "at a and <outside-repo>\n"
    assert outside == 1


def test_scrubber_in_place_leaves_no_absolute_path(tmp_path: Path, capsys):
    log = tmp_path / "run.log"
    log.write_text(f"INFO found uv.lock in {ROOT}\nplain line\n{ROOT}/s.py:1: W\n")
    scrub.main(["--in-place", str(log)])
    assert log.read_text() == "INFO found uv.lock in ./\nplain line\ns.py:1: W\n"
    assert "2 line(s) rewritten" in capsys.readouterr().out
    assert not scrub._check.find_violations(log.read_text())
