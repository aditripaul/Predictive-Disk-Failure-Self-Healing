"""Structured logging configuration (docs/design_goal.md / docs/project_plan.md
"Logging / audit: structlog / JSON logs" — "Per-decision rationale,
guardrail status, explainability... query-friendly, append-only").
`structlog` was a declared but entirely unused dependency; every pipeline
script and `src/agent/demo.py` used plain `print()` instead, which can't
be filtered, queried, or shipped to a log aggregator as structured events.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(*, json_output: bool = False) -> None:
    """Configures the process-wide structlog pipeline. Safe to call more
    than once (e.g. once per pipeline entry point + once in tests) -
    `structlog.configure` simply overwrites the prior configuration.
    `json_output=True` renders newline-delimited JSON, the audit-friendly
    format the design docs describe; the default console renderer is
    friendlier for interactive `make` runs."""
    renderer = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)
