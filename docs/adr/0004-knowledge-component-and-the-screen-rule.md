# ADR 0004: Give the agent a Knowledge component, and let it change decisions only through a pre-registered screen

Status: accepted, 2026-11-07. Implemented; no knowledge rule is enabled, because none passed.

## Context

The agent is a MAPE-K loop with no K. Its "knowledge" was configuration, trained thresholds, a
hysteresis counter and two ledgers; no outcome fed back into a decision, and the final
trust-score and drift code was called only by tests. ADR 0003 had shown that re-filtering the
model's flags does not raise precision, and named the one way it could be reopened: evidence
the model has not seen. A drive's own history over months is such evidence, since the model's
features cover at most 30 days.

## Decision

1. **The Knowledge component is a time-indexed store of facts, queried as of a date**
   (`src/agent/knowledge/`). It stores candidate days and their scores, presence, failure days,
   cohort counts, recorded decisions and final outcomes; everything else (episodes, a drive's
   context, outcomes) is derived for the requested date. A view as of a date never sees a later
   fact, and building day by day equals building in one pass. Columns computed over the whole
   data range are never read. "Censored" is never a final outcome, because a silent drive can
   come back.
2. **It is read-only context in the loop.** Analyze attaches a knowledge context to each
   candidate, Plan carries it in the proposal, Validate records the decision. With no rule
   enabled the agent's decisions are identical with and without the store (tested).
3. **A rule may change a decision only after passing a screen whose rule is written down and
   committed before any screening code or result exists** (`docs/agent_knowledge_base.md`
   section 8). Amendments are appended, dated and justified, and also made before any result.
   It is judged on validation drives point-in-time against "no rule" and against a plain
   stricter threshold at matched recall and false-alarm rate, with a paired drive bootstrap and a
   power floor. A passing rule earns exactly one logged, descriptive look at a test period.
4. **The control-arm fraction stays 0.0 while the executor is simulated.** It is set (0.03
   recommended) in the same commit that connects a real executor, because real actions censor
   their own outcomes and the store's decisions and outcomes are what make that measurable.

## Consequences

- The screen ran on the 60-day stacked Stage 2 (2026-11-07; runbook 1.1.7). No rule passed:
  shrinking toward the drive model's failure rate gave no precision gain (as in runbook 4.10),
  demoting chronic cordon drives would cost 4-10 points of recall, because 138 of the 222
  episodes with a week or more above cordon ended in failure, and the remaining rules had too
  little history to fire or be judged. **A drive's own long-run history carries information,
  but it points toward failure, not toward a false alarm.** ADR 0003's conclusion stands.
- The store is still useful: outcomes and final trust scores can be resolved from recorded
  decisions, the daily Stage 1 pass share has a drift monitor, and the loop has a tested place
  for a rule.
- The screen's selection criterion (recall at the tier's own false-alarm rate) cannot see a
  pure tier rule, because a demoted drive keeps its rank above the rest of its new tier. A
  future screen of tier rules should select on precision and recall at the fixed threshold.

## What would reopen this

A rule built on information the store holds but the model does not (for example a drive's
history across an untouched later period long enough to resolve episodes at the horizon, or
cohort and location data the ingest currently drops), passing the same kind of pre-registered
screen.
