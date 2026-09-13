"""Guardrail engine: evaluates every rule against a RuleContext and produces
a GuardrailResult (docs/design_goal.md section 14). Target latency is
<500ms (docs/project_plan.md Phase 7 exit criteria).
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable

from data_contracts.schemas import GuardrailResult, GuardrailSeverity, GuardrailViolation
from src.guardrails.context import RuleContext
from src.guardrails.rules import ALL_RULES

_SEVERITY_ORDER = {GuardrailSeverity.HARD: 0, GuardrailSeverity.SOFT: 1, GuardrailSeverity.NONE: 2}


RuleFn = Callable[[RuleContext], GuardrailViolation | None]


class GuardrailEngine:
    def __init__(self, rules: tuple[RuleFn, ...] = ALL_RULES):
        self.rules = rules

    def evaluate(self, ctx: RuleContext) -> GuardrailResult:
        start = time.perf_counter()

        violations = [v for rule in self.rules if (v := rule(ctx)) is not None]
        violations.sort(key=lambda v: _SEVERITY_ORDER[v.severity])

        has_hard_violation = any(v.severity == GuardrailSeverity.HARD for v in violations)
        latency_ms = (time.perf_counter() - start) * 1000

        # Safety > Operational > Efficiency: any hard violation blocks the
        # action outright (routed to human review upstream); soft violations
        # do not block, but are recorded for the trust-score guardrail
        # multiplier (Phase 9).
        final_action = ctx.proposed_action

        return GuardrailResult(
            action_id=ctx.action_id,
            evaluated_at=dt.datetime.now(dt.UTC),
            passed=not has_hard_violation,
            violations=violations,
            final_action=final_action,
            evaluation_latency_ms=latency_ms,
        )
