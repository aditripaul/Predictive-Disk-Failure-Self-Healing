"""Structured logging configuration (docs/design_goal.md / docs/project_plan.md
"Logging / audit: structlog / JSON logs" — "Per-decision rationale,
guardrail status, explainability... query-friendly, append-only").
`structlog` was a declared but entirely unused dependency; every pipeline
script and `src/agent/demo.py` used plain `print()` instead, which can't
be filtered, queried, or shipped to a log aggregator as structured events.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

from src.repo_paths import relativize_text


def _relativize(value: Any) -> Any:
    if isinstance(value, str):
        return relativize_text(value)
    if isinstance(value, os.PathLike):
        return relativize_text(os.fspath(value))
    if isinstance(value, dict):
        return {k: _relativize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_relativize(v) for v in value)
    return value


def relativize_paths(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """structlog processor: every path inside the repository is written
    repo-relative, so a log can be committed (scripts/check_no_absolute_paths.py)."""
    return {key: _relativize(value) for key, value in event_dict.items()}


def _install_stdlib_hooks() -> None:
    """Third-party loggers (MLflow, uv detection) and warnings print the
    repository's absolute location; rewrite it there too. Installed once."""
    if getattr(logging, "_repo_relative_paths", False):
        return
    make_record = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = make_record(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:  # a malformed record is left for logging to report
            return record
        relative = relativize_text(message)
        if relative != message:
            record.msg, record.args = relative, None
        return record

    logging.setLogRecordFactory(record_factory)

    format_warning = warnings.formatwarning

    def formatwarning(message, category, filename, lineno, line=None):
        return relativize_text(format_warning(message, category, filename, lineno, line))

    warnings.formatwarning = formatwarning
    logging._repo_relative_paths = True  # type: ignore[attr-defined]


def configure_logging(*, json_output: bool = False) -> None:
    """Configures the process-wide structlog pipeline. Safe to call more
    than once (e.g. once per pipeline entry point + once in tests) -
    `structlog.configure` simply overwrites the prior configuration.
    `json_output=True` renders newline-delimited JSON, the audit-friendly
    format the design docs describe; the default console renderer is
    friendlier for interactive `make` runs. Paths inside the repository are
    written repo-relative, in our events and in third-party log messages and
    warnings (`src/repo_paths.py`)."""
    _install_stdlib_hooks()
    renderer = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            relativize_paths,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)
