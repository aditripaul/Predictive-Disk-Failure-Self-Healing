# ADR 0005: Set the action tiers from costs, and state what the lift targets already assume

Status: accepted, 2026-11-10. Implemented. No costs have been supplied, so the agent still acts on the lift tiers.

## Context

The agent's four action tiers (warn, cordon, migrate, drain) are cut from the score at lift targets
(`configs/model.yaml threshold.action_tier_lift`). Those targets are placeholders, carried over from
precision targets that were never derived from anything. The precision goal (68% at 10% or more recall) is not met.
Above about 60% recall, precision falls to single digits because of what daily SMART telemetry can see, not because
of the model (report Section 14). Below that ceiling, whether an operating point is acceptable is a cost question:
an action pays when `precision x P(save) x C_failure > C_action` (runbook Section 4).

`src/models/threshold.py::tune_cost_tiers` (2026-11-07, runbook 1.1.7) already placed each tier where its
action pays for itself. It takes one ratio `r` per action, "how many unnecessary actions one prevented failure
is worth", whose values (100 / 20 / 5 / 2) were placeholders too. Nobody could supply `r` directly: it folds
together the cost of a failure, the cost of the action, and how often the action actually prevents the failure.
Two lines of the runbook still said that no cost model existed.

## Decision

1. **Costs are the inputs and `r` is derived from them.** `threshold.cost_model` takes `failure_cost`, and an
   `action_cost` and `p_save` per tier. An action on a drive that fails within the horizon is worth
   `p_save x failure_cost - action_cost`; on one that does not fail, it costs `action_cost`. The net value
   of a flagged set is therefore `action_cost x (r x caught - false_alarms)` with
   `r = p_save x failure_cost / action_cost - 1`, and flagging pays while the marginal precision is at
   least `1 / (1 + r)` (`cost_ratios_from_costs`). Two input sets are rejected:
   - an action that never pays (`r <= 0`);
   - costs that make a more drastic action cheaper per prevented failure than a milder one (`r` increasing
     from warn to drain).
2. **One resolver decides which ratios are in force** (`resolve_cost_ratios`).
   - All cost inputs null (the shipped default): the placeholder `action_cost_ratio`.
   - All set: the derived ratios.
   - Partly filled: an error, so real inputs are never mixed with placeholders.

   `scripts/freeze_stack_scorer.py` and `scripts/cost_tier_report.py` both call it. The thresholds file
   the agent reads and the report therefore cannot disagree, and the file records `cost_ratio_source`.
3. **The lift tiers stay the default** (`threshold.tier_rule: lift`) until costs are supplied. Switching
   takes three steps: set `tier_rule: cost`, re-freeze the scorer, and judge the result point in time
   against the lift tiers.
4. **State what the current tiers already assume** (`implied_cost_ratio`). For a given threshold, it reports
   the range of `r` for which that threshold is the cost-optimal cut. It reads this from the upper convex hull
   of the validation drive curve, where every optimum of `r x caught - false_alarms` lies. Operations then has
   four numbers to confirm or replace, not a blank form.
5. **Validation only, and late hits are not credited.** On the chronological split, days 31-60 after a
   validation snapshot fall in the test period, so crediting late hits would mean reading test-period
   failures.

## Consequences

- **Result** (2026-11-10, `make chrono-cost-tiers`, runbook 1.9, `data/audit/cost_tiers_chrono_h30/`):
  - The lift tiers are cost-optimal for about r = 35 (warn), 16-20 (cordon), 5-7 (migrate) and 2-4 (drain),
    the same on all four chronological arms. The placeholder ratios agree except at warn, where the
    placeholder (100) is three times higher.
  - At the placeholders, the cost tiers move warn from 59-60% to 68-69% drive recall (precision 10.9% to
    6.5-7.3%) and cordon from 49% to 51-55%.
  - The net value gained is 8-9% at warn and 2-4% at cordon, measured on the drives both rules were
    tuned on. Point in time, the cordon net is within 2% either way.
- **The ratio is the decision, not the rule.** Lift and cost tiers with the same implied ratio sit in the
  same place. Costs decide where on the curve each action sits; the ranking, and with it the 68% goal at
  10% recall, does not change.
- **`make train` still writes only the lift tiers** (`action_tiers.json`). Cost tiers are written only
  into frozen scorers' thresholds files (`action_tiers_cost`, by `freeze_stack_scorer.py`).
  `tier_rule: cost` therefore takes effect only when the agent reads a precomputed scorer
  (`src/agent/fleet.py`).
- **The cost optimum depends on the failure rate.** A tier tuned at validation's 0.19% flags slightly
  more than is optimal at test's 0.23%. On the drive-holdout split the two rates were twice as far apart.
- **What would change this:**
  - Operations supplies costs: fill in `threshold.cost_model`, re-freeze, and judge `tier_rule: cost`
    point in time on a period nothing has seen.
  - A data period whose days 31-60 lie outside any test window: credit late hits for the cheap tiers.

## Alternatives considered

- **Ask operations for `r` directly.** Rejected: `r` mixes three quantities, and nobody estimates it as
  one number.
- **Put the cost inputs in `configs/agent.yaml`**, where the runbook once said a cost model would go.
  Rejected: the tier code reads `configs/model.yaml threshold`, next to the lift targets and the
  placeholder ratios, and one place is easier to keep consistent.
- **Measure marginal precision over a window of drives around the cut**, not the convex hull. Rejected:
  the hull is exact and is the set of cuts `tune_cost_tiers` can return; a window is noisy at the strict
  end, where drain has a few hundred drives.
- **Switch the agent to the cost tiers now.** Rejected: with placeholder costs this would replace one
  placeholder with another, and the measured gain is small and in-sample.
