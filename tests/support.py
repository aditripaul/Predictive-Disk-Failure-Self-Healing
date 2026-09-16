"""Shared test constants (not a conftest.py - plain importable helpers)."""

#: Most agent tests exist to prove something about guardrails/execution/API
#: wiring, not hysteresis/cooldown itself - passing this to
#: `compiled_agent(deps, plan_kwargs=...)` restores immediate single-cycle
#: escalation (hysteresis_cycles_required=1) with no cooldown interference,
#: matching pre-hysteresis test expectations. Hysteresis/cooldown get their
#: own dedicated tests (tests/unit/test_hysteresis.py,
#: tests/integration/test_agent_graph.py's hysteresis-specific cases).
IMMEDIATE_ESCALATION_PLAN_KWARGS = {"hysteresis_cycles_required": 1, "cooldown_seconds": 0}
