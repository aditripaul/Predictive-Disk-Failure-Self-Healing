# ADR 0003: Judge the flagging policy point-in-time, and accept that the loop cannot raise precision

Status: accepted, 2026-10-18. Implemented; the cost model and the weak-family
features it points to are not done.

## Context

Two questions came up once the agent could run on the real fleet
(`make agent-fleet`, `src/agent/fleet.py`):

1. What do precision and recall look like for the policy the agent actually
   runs - the score, the tier thresholds, the evidence gates, hysteresis -
   rather than for the model's score alone?
2. Can the agent loop reduce the model's false positives, and if so how?

The first agent backtest answered the first question with **61.8% precision
at 48.3% recall** (Apr-May 2026, 30-day horizon). Neither number was real.
Its per-tier breakdown was the wrong way round: migrate and drain were 0 for
71, while the drives the evidence gates had downgraded to cordon were 90%
correct. Two look-aheads explained it, and a diagnostic on the saved run
confirmed both:

- **`feature_maturity` is assigned from the dataset's last date.** A drive
  whose final row is earlier than the data's end is DECOMMISSIONED, checked
  before every other state, and the migrate/drain gate requires MATURE. In a
  window ending a month before the data does, "MATURE" means "still reporting
  a month after the snapshot", and such a drive cannot have failed inside the
  horizon. Maturity separated the flagged list almost perfectly (88.8% of
  DECOMMISSIONED drives failed, 0.2% of MATURE) - it was reading the outcome.
- **"Latest row per drive" picks a dying drive's last day.** 59% of the acted
  drives had left the fleet mid-window, and 84.9% of the drives the agent
  "caught" were scored on the day they died: median notice 0 days. A fleet
  scored at a fixed date does not contain drives that already died.

Measured cleanly, at a single date, the same policy gave 17.9% precision at
26.6% recall.

## Decision

**1. Any precision or recall figure for the flagging policy comes from
`make agent-pit`**, which takes fixed-date snapshots: every drive with a
feature row on that date, scored once, judged on the label at that same row.
Drives that had already gone are absent and no maturity label is consulted.
`make agent-fleet --score` remains for exercising the loop and is not a
source of quotable numbers; it warns when it scores a window that ends before
the data does.

**2. Precision is quoted with its base rate, and runs are compared in ROC
space.** Across nine weekly snapshots the 30-day failure rate fell 1.76x and
cordon-and-above precision fell from 26.8% to 17.9% while recall stayed
between 26% and 33% and the likelihood ratio (recall / false-alarm rate) had
no trend. A single date can land anywhere in a 14-point precision range for a
model that did not change. Intervals resample whole drives, because weekly
snapshots share most of their drives and a failing drive is positive in about
four of them.

**3. The loop gets no new precision mechanism.** Everything that could be
measured at the policy layer was, point-in-time, and none of it raises recall
at a matched false-alarm rate:

| Mechanism | Result |
|---|---|
| Confidence and maturity gates | They downgrade migrate/drain to cordon, which still acts; they never prevent an action. Zero effect. |
| Hysteresis (require a second look) | Flagging only when a drive scores high at T-7 and T costs 0.5 to 3.3 points of recall at matched false-alarm rate on the test window, intervals clear of zero at three of five budgets, and again -3.1 and -2.0 on the validation window. A fifth persistence refutation. |
| Other two-look combinations (max, mean, rising) | `max` within a point of one look, never clear of zero; `mean` and `min` worse; `rising` worse by up to 17 points. |
| Suppressing chronic flaggers | Precision is flat by flag streak (18.6% to 22.2%). No such group. |
| Filtering on SMART attributes | Each priority attribute is non-zero in nearly the same share of true and false positives (83% / 80%, 88% / 81%, ...). A false positive looks like a failing drive. |
| Per-family recalibration with shrinkage, fitted on validation and judged on test | Recall changes of -1.4 to +0.6 points across six budgets; five intervals straddle zero and the one that excludes it is negative. |
| Per-family thresholds | Lost on test (earlier work, ADR 0002). |

**4. `hysteresis_cycles_required` should be 1.** The evidence supports it. It
is a policy setting, so it is recorded here as a recommendation and was not
changed.

## Consequences

- The honest policy-level numbers for the quarter-aligned model are, pooled
  over nine test weeks: cordon and above **23.4% precision [21, 25] at 29.3%
  recall**; migrate and above 35.7% [32, 40] at 10.3%; drain 40.3% [29, 52] at
  1.6%, with the validation window agreeing (22.6% at 28.4%). Drain is the
  unstable tier - 67% on 64 flags in validation, 40% on 149 in test.
- 12.8% of false alarms fail in days 31 to 60 and are early, not wrong.
  Crediting them lifts cordon-and-above precision from 28.7% to 37.8% on the
  weeks where that window is observed. It is reported beside the 30-day
  figure and never instead of it; the goal stays pinned to 30 days.
- Ranking quality varies 73-fold across drive families (PLR 22 to 1,583
  against 250 pooled). That is discrimination *within* a family, which no
  policy can repair; six families below 100 hold 33% of false alarms. The fix
  is features for those families, which is model work.
- Choosing the operating point is a cost decision. A cordon pays for itself
  when `precision x P(save) x C_failure > C_action`, so at 23% precision it
  must cost under roughly a quarter of an unplanned failure, scaled by how
  often it prevents one. The cost model does not exist; until it does there is
  no principled place to put the thresholds.
- Real actions censor their own outcomes: drain a drive about to fail and its
  label is 0. `control_fraction` (`configs/agent.yaml`, default 0.0) holds back
  a stable-hash share of flagged drives so some labels stay unbiased. It must
  be set before a real executor is connected; `src/reliability/checks.py`
  documents which of its metrics assume actions do not change outcomes.

## What would reopen this

Evidence the model has not seen - a probe, host I/O errors and latency, RAID
scrub results - or features that change within-family ranking. Reordering or
filtering the flags the current model produces will not.

## Alternatives considered

- **Patch the maturity gate and keep the latest-row backtest.** Rejected: the
  latest-row selection is biased on its own, independently of maturity.
- **Treat the first backtest as an upper bound.** Rejected: an upper bound
  nobody can reach is still a number that gets quoted.
- **Add a persistence or novelty rule to the loop.** Rejected on the evidence
  above, five times.
