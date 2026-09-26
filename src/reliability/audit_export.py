"""Audit trail export to CSV/Parquet (docs/project_plan.md Phase 10 Key
Task 6: "Implement audit export to CSV/Parquet"). Lets an operator or
compliance reviewer pull the decision audit trail out of the in-memory
store (`src/api/store.py::InMemoryAuditStore`) as a file, rather than only
being able to view it through `GET /api/v1/audit/decisions`.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

#: Decision fields that are already flat scalars (see
#: AgentOrchestrator._handle_result's `add_decision` payload) and should be
#: written as-is; everything else (nested dicts/lists, e.g.
#: `guardrail_result`, `execution_result`) is JSON-serialized to a string so
#: a single row schema stays representable in both CSV and Parquet even
#: when different decisions have different nested shapes (or `None`).
_SCALAR_FIELDS = frozenset(
    {
        "decision_id",
        "action_id",
        "drive_id",
        "date",
        "horizon_days",
        "timestamp",
        "proposed_action",
        "human_review_required",
        "human_decision",
        "override_reason_code",
        "trust_score_provisional",
        "trust_score_final",
        "safety_violation",
        "guardrail_severity",
        "cycle_duration_seconds",
    }
)


def _flatten_decision(decision: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in decision.items():
        if key in _SCALAR_FIELDS or value is None:
            flat[key] = value
        else:
            flat[key] = json.dumps(value, default=str)
    return flat


def decisions_to_dataframe(decisions: list[dict[str, Any]]) -> pl.DataFrame:
    if not decisions:
        return pl.DataFrame()
    return pl.DataFrame([_flatten_decision(d) for d in decisions])


def export_decisions_to_csv(decisions: list[dict[str, Any]]) -> bytes:
    df = decisions_to_dataframe(decisions)
    return df.write_csv().encode("utf-8")


def export_decisions_to_parquet(decisions: list[dict[str, Any]]) -> bytes:
    import io

    df = decisions_to_dataframe(decisions)
    buffer = io.BytesIO()
    df.write_parquet(buffer, compression="zstd")
    return buffer.getvalue()
