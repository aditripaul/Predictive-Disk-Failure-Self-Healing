"""Guardrail engine: evaluates every rule against a RuleContext and produces
a GuardrailResult (docs/design_goal.md section 14). Target latency is
<500ms (docs/project_plan.md Phase 7 exit criteria).
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from typing import TypeVar

from data_contracts.schemas import GuardrailResult, GuardrailSeverity, GuardrailViolation
from src.guardrails.context import PostActionContext, RuleContext
from src.guardrails.rules import ALL_RULES, POST_ACTION_RULES

_SEVERITY_ORDER = {GuardrailSeverity.HARD: 0, GuardrailSeverity.SOFT: 1, GuardrailSeverity.NONE: 2}

CtxT = TypeVar("CtxT", RuleContext, PostActionContext)
RuleFn = Callable[[CtxT], GuardrailViolation | None]


class GuardrailEngine:
    def __init__(
        self,
        rules: tuple[RuleFn[RuleContext], ...] = ALL_RULES,
        post_action_rules: tuple[RuleFn[PostActionContext], ...] = POST_ACTION_RULES,
    ):
        self.rules = rules
        self.post_action_rules = post_action_rules

    def evaluate(self, ctx: RuleContext) -> GuardrailResult:
        violations, latency_ms = self._run_rules(self.rules, ctx)
        has_hard_violation = any(v.severity == GuardrailSeverity.HARD for v in violations)

        # Safety > Operational > Efficiency: any hard violation blocks the
        # action outright (routed to human review upstream); soft violations
        # do not block, but are recorded for the trust-score guardrail
        # multiplier (Phase 9).
        return GuardrailResult(
            action_id=ctx.action_id,
            evaluated_at=dt.datetime.now(dt.UTC),
            passed=not has_hard_violation,
            violations=violations,
            final_action=ctx.proposed_action,
            evaluation_latency_ms=latency_ms,
        )

    def evaluate_post_action(self, ctx: PostActionContext) -> GuardrailResult:
        """POST_DATA_INTEGRITY / POST_SERVICE_CONTINUITY (docs/design_goal.md
        section 14), evaluated in the Validate node after execution. A
        failure here is what the fleet simulator's compensating action is
        meant to remedy; this records it for the audit trail and trust
        score regardless of whether the simulator already rolled back."""
        violations, latency_ms = self._run_rules(self.post_action_rules, ctx)
        has_hard_violation = any(v.severity == GuardrailSeverity.HARD for v in violations)

        return GuardrailResult(
            action_id=ctx.action_id,
            evaluated_at=dt.datetime.now(dt.UTC),
            passed=not has_hard_violation,
            violations=violations,
            final_action=ctx.proposed_action,
            evaluation_latency_ms=latency_ms,
        )

    @staticmethod
    def _run_rules(rules: tuple[RuleFn, ...], ctx) -> tuple[list[GuardrailViolation], float]:
        start = time.perf_counter()
        violations = [v for rule in rules if (v := rule(ctx)) is not None]
        violations.sort(key=lambda v: _SEVERITY_ORDER[v.severity])
        latency_ms = (time.perf_counter() - start) * 1000
        return violations, latency_ms
