"""Streamlit dashboard: fleet health, trust trend, approval queue, and audit
trail (README.md section 19, docs/design_goal.md section 15). Reads from
the FastAPI service, so `make api` must be running (or set API_BASE_URL).

Not covered by automated tests: Streamlit apps are rendered in a browser
session and aren't meaningfully exercisable via pytest. Run `make dashboard`
and click through the tabs manually against a live API instance to verify
behavior.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Disk Failure Self-Healing Agent", layout="wide")
st.title("Disk Failure Self-Healing Agent — Fleet & Trust Dashboard")


def _get(path: str) -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}{path}: {exc}")
        return {}


def _post(path: str, body: dict) -> dict:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=body, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return {}


tab_fleet, tab_trust, tab_approvals, tab_audit = st.tabs(
    ["Fleet Health", "Trust & Reliability", "Approval Queue", "Audit Trail"]
)

with tab_fleet:
    fleet_state = _get("/api/v1/fleet/state")
    drives = fleet_state.get("drives", [])
    if drives:
        by_state: dict[str, int] = {}
        for d in drives:
            by_state[d["state"]] = by_state.get(d["state"], 0) + 1
        st.metric("Total drives", len(drives))
        st.bar_chart(by_state)
        st.dataframe(drives)
    else:
        st.info("No fleet state reported yet.")

with tab_trust:
    trend = _get("/api/v1/reliability/trust-trend").get("trust_trend", [])
    if trend:
        st.line_chart(
            {
                "provisional": [t["trust_score_provisional"] for t in trend],
                "final": [t.get("trust_score_final") or 0 for t in trend],
            }
        )
        st.dataframe(trend)
    else:
        st.info("No trust scores recorded yet.")

    violations = _get("/api/v1/guardrails/violations").get("violations", [])
    st.subheader("Guardrail violations")
    st.dataframe(violations) if violations else st.info("No guardrail violations recorded.")

with tab_approvals:
    pending = _get("/api/v1/actions/pending").get("actions", [])
    if not pending:
        st.info("No actions pending human review.")
    for action in pending:
        with st.expander(f"{action['drive_id']} — {action['proposed_action']}"):
            st.json(action)
            operator_id = st.text_input("Operator ID", key=f"op_{action['action_id']}")
            reason_code = st.text_input("Reason code", key=f"reason_{action['action_id']}")
            comment = st.text_area("Comment (optional)", key=f"comment_{action['action_id']}")
            col_approve, col_reject = st.columns(2)
            if col_approve.button("Approve", key=f"approve_{action['action_id']}"):
                _post(
                    f"/api/v1/actions/{action['action_id']}/approve",
                    {"operator_id": operator_id, "reason_code": reason_code, "comment": comment},
                )
                st.rerun()
            if col_reject.button("Reject", key=f"reject_{action['action_id']}"):
                _post(
                    f"/api/v1/actions/{action['action_id']}/reject",
                    {"operator_id": operator_id, "reason_code": reason_code, "comment": comment},
                )
                st.rerun()

with tab_audit:
    decisions = _get("/api/v1/audit/decisions").get("decisions", [])
    if decisions:
        st.dataframe(decisions)
    else:
        st.info("No decisions recorded yet.")
