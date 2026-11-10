# System summary: a self-healing disk agent that can be trusted

Status: 2026-11-10 (current model tested on Q3 2026; stacked Stage 2, deep
dive and cluster policy, runbook Sections 1.1.4 to 1.1.6 and 1.9). The config now holds the
chronological split of runbook 1.1.9, which has not been run: the model described below is still the
drive-holdout run `c459046e`.
This is the entry point for a reviewer. It describes what
the project built, how a decision flows through it, what was measured on real
data, and what is not done. Detailed results are in
`docs/model_status_and_runbook.md`; the code map is in
`docs/developer_guide.md`.

## 1. What the project is

A system that watches the SMART health readings of every drive in a server
fleet, predicts which drives are about to fail, and takes a protective action
(warn, cordon, migrate, drain) before the failure. A second layer then audits
each of those automated decisions and scores how far it can be trusted.

The project's question is not only "can we predict failures?" but "can an
autonomous system act on imperfect predictions safely?". The design answers
that with layers: a prediction alone never drains a drive.

## 2. How one decision flows

```text
SMART telemetry (Backblaze, daily, per drive)
   -> data pipeline: ingest -> harmonize -> features -> labels
   -> failure model: probability that the drive fails within 30 days
   -> action tier: warn / cordon / migrate / drain, each with its own cut-off
   -> guardrail engine: 11 rules; a hard violation always blocks
   -> execute in the fleet simulator, or pause for human approval
   -> validate the outcome; roll back (compensate) if the action failed
   -> reliability checker: trust score, veto on any safety violation
   -> dashboard and audit trail
```

| Layer | What it does | Where |
|---|---|---|
| Data pipeline | Turns daily Backblaze telemetry (62.6 million drive-days from 363,548 drives in Q1 + Q2 2026; the data now runs from Q4 2025 to Q3 2026) into 246 features per drive-day, within a 20 GB memory cap | `pipelines/`, `src/features/` |
| Failure model | LightGBM classifier whose net keeps 80% of failing drives on validation, with a second LightGBM that re-ranks the 5-6% of drives it passes; outputs a failure score per drive-day | `pipelines/train_model.py`, `src/models/two_stage.py` |
| Action tiers | Maps the probability to an action. Mild actions accept more false alarms; destructive ones need more confidence | `src/models/threshold.py` |
| Agent | A Monitor - Analyze - Plan - Execute loop (LangGraph), checkpointed so a crash cannot repeat an action | `src/agent/` |
| Guardrails | Six hard pre-action rules (score cut-off, feature confidence, fresh telemetry, never the last healthy node, keep replication quorum, limit concurrent drains), three soft ones (high I/O, maintenance window, rate limit), and two hard post-action checks (data integrity, service continuity) | `src/guardrails/` |
| Human review | Blocked or uncertain destructive actions wait for an operator to approve or reject | `src/api/` |
| Fleet simulator | Drives, nodes and replication groups with injected faults, so actions and rollbacks can be exercised without real hardware | `src/simulator/` |
| Reliability checker | Scores every decision on correctness, necessity and timeliness; any safety or hard-guardrail violation sets the trust score to zero | `src/reliability/` |
| Dashboard | Fleet health, approval queue, trust trend, audit trail | `src/dashboards/` |

## 3. Why the system does not need a perfect model

Disk failure is rare: in the Q3 2026 test period 1,105 of about 369,000 drives
failed, roughly 1 in 330 (1 in 570 in the retired model's test, 2026-05-11 to 05-31). At that rate any predictor produces false alarms, so the
system is built to make a false alarm cheap and a wrong destructive action
unlikely:

- **Graduated actions.** A low score only raises a warning. Draining a drive
  needs the highest score.
- **Two independent checks before a destructive action.** Low feature
  confidence or stale telemetry downgrades the action when it is planned, and
  the guardrail engine blocks it again if that first check is bypassed.
- **Fleet-safety rules that ignore the model.** However confident the model
  is, the agent may not drain the last healthy node in a failure domain or
  break replication quorum.
- **A human in the loop** for anything blocked or uncertain.
- **Rollback.** A failed action is compensated, and a crash cannot cause the
  same action to run twice.
- **An audit of every decision**, with a veto when a safety rule was broken.

## 4. What the model achieves on real data

**The current model (2026-11-06)** is the 30-day two-stage LightGBM trained on
2026-01-01 to 05-01 (80% of drives; the other 20% are the validation holdout)
and tested on Q3 2026, a quarter it never saw, frozen as `c459046e` (runbook
1.1.4, `docs/q3_horizon_and_cascade_run.md`). Per drive, thresholds chosen on
validation, 1,105 failing and 367,583 healthy test drives:

| Operating point | Precision | Recall | Caught / missed / false alarms |
|---|---|---|---|
| Stage 1 net (filter, cutoff 0.235) | 3.8% | 79.9% | 883 / 222 / 22,463 |
| Warn tier | 17.3% | 60.9% | 673 / 432 / 3,208 |
| Cordon tier | 34.8% | 42.4% | 469 / 636 / 877 |
| **Primary threshold** | **59.0%** [52.3, 65.5] | **12.5%** [10.5, 14.3] | 138 / 967 / 96 |
| Migrate = drain (fallback cutoff) | 33.3% | 0.2% | 2 / 1,103 / 4 |

- **The goal (68% precision at 10% or more recall on the stacked cascade, Stage 1 LightGBM and Stage 2 LightGBM + Cox; 90% until 2026-11-10) is not met on test:** the stack gives 64.9% at 11.6% (3-seed mean, runbook 1.1.4a) and the shipped LightGBM two-stage 59.0% at 12.5%. Drive AUROC is
  0.926 and recall at a 0.1% false-alarm rate 30.4%.
- **Validation reads 15-20 points higher, and that is the base rate** (0.59% of
  validation drives fail, 0.30% of test drives): AUROC is the same on both
  (0.922 and 0.926), and test precision restated at the validation rate is
  73.9% against 75.0% (runbook 1.1.5).
- **Migrate and drain do not operate on this model.** Their lift targets
  convert at the validation failure rate to 88.9% and 133% precision: migrate's
  is met only by the two highest-scored validation drives and drain's not at
  all, so both share the cutoff 0.926 and flag 6 drives in the 62-day Q3 test window. The agent's effective tiers are warn and cordon (runbook 1.9).
- **The fifth of failing drives that Stage 1 never passes is silent:** 98.6% of
  them show no rise in any core SMART counter in their last 30 days, with
  complete telemetry, at a median age of 2.9 years. That caps recall near 80%.
- **False alarms are damaged drives:** 98-99.8% of those at warn and above
  have a non-zero core counter (8.9% of the healthy fleet), and 35.4% of the
  primary threshold's false alarms were removed from service anyway.
- Longer label horizons (44, 60 days) rank drives no better once the base rate
  is removed; survival Stage 2 models (Cox matches LightGBM, AFT is worse) and
  a stack of LightGBM and Cox (a small gain in the top 500 drives, not
  confirmed) change nothing above about 20-35% recall. Section 5 lists them.

**Precision depends on how rare failures are in the test set**, so the
retired model's operating points below are shown twice: at the fleet's own
failure rate, which is what an operator would experience, and restated for a
test set in which 15% of drives fail, which is how most published results are
reported and the figure to compare with the literature (Amram et al., 2021,
used about 15% positives). The model and its alerts are identical either way;
only the mix of drives it is judged on changes. That retired model's setup:
Backblaze Q1 + Q2 2026, training to 2026-03-16, validation 2026-04-16 to
05-10, test 2026-05-11 to 05-31, 621 failing and 353,328 healthy test drives.

**The previous model of record**, a frozen training run, was **retired on
2026-11-06**: its artifacts
are not available to this repository, so it can never be scored on the sealed
quarter or the cross-vendor set, and Q3 2026 is now used as a test set
(`docs/q3_horizon_and_cascade_run.md`). Its results, kept as a historical
record:

| Operating point | Precision, real fleet (0.18% fail) | Failing drives caught | Precision, 15% fail |
|---|---|---|---|
| **Primary threshold** | **48.6%**, 95% CI [38.9%, 58.2%] | **8.7%** | 99.0% |
| Broader | 39.7% | 20.3% | - |
| Broadest | 29.5% | 33.3% | - |

(The two broader rows cannot be re-checked: the run's artifacts are gone, and
the runbook's Section 7 summarises that run as 49.6% at 10.5% recall and 28.7%
at 32.4%. The primary row is the figure of record.)

At the primary threshold an alerted drive is about 277 times more likely to
fail than one picked at random, and fewer than 1 healthy drive in 6,000 is
alerted. The interval is a 1000-draw bootstrap that resamples whole drives,
so a drive's many near-identical rows move together.

**An earlier freeze is void, not merely superseded.** A prior run,
`0ee06c01ff2a427ca76011fb1afb8ce3`, was trained before two bugs in
`feature_confidence` were fixed (see Section 7 and ADR 0002); its own
feature columns are now far outside the distribution the rebuilt data
produces, so it may not be scored on any held-out period or SMART-Z even
though every one of its expected columns still exists in the new frame.

**The fuller breakdown below** - drive counts, the action tiers and the
warning lead time - comes from that earlier, now-void run, on an earlier
data build with 212 features (46.2% precision at 9.7% recall at the primary
threshold). It is kept because the breakdown was never re-measured for the
current freeze, not because the run itself is still trustworthy.

| Operating point | Failing drives caught | Healthy drives wrongly alerted | Precision, real fleet (0.18% fail) | Precision, 15% fail | Precision, 50% fail |
|---|---|---|---|---|---|
| Strictest | 18 of 621 (2.9%) | 22 of 353,328 (0.006%) | 45.0% | 98.8% | 99.8% |
| Primary threshold | 60 of 621 (9.7%) | 70 (0.020%) | 46.2% | 98.9% | 99.8% |
| Broader | 129 of 621 (20.8%) | 164 (0.046%) | 44.0% | 98.7% | 99.8% |
| Broadest | 207 of 621 (33.3%) | 506 (0.143%) | 29.0% | 97.6% | 99.6% |
| Warn tier | about 388 of 621 (62.5%) | about 3,520 (1.0%) | 9.9% | 91.7% | 98.4% |

The 15% and 50% columns are computed from the measured catch rate and
false-alarm rate (precision = r·p / (r·p + f·(1 − p)) for catch rate r,
false-alarm rate f and failure rate p). They are not a separate experiment.

Other measures at the primary threshold of the earlier (now-void) run:

- **Lift:** an alerted drive is about 263 times more likely to fail than a
  drive picked at random (277 times for the retired frozen run above).
- **Warning time:** median 14 days before the failure (mean 17.9).
- **False-alarm rate:** 70 of 353,328 healthy drives, about 1 in 5,000.
- **Validation agrees with test once the failure rate is held fixed:** at
  15% failing, precision at the primary threshold is 99.0% on validation and
  98.9% on test.

**Against the project's stated goal.** The original target was 95% precision
at 35-50% recall on the real fleet; it was later set to 90% at 10% recall.
Neither is met: the current model gives 59.0% at 12.5% recall on Q3, interval
[52.3%, 65.5%], and the retired one 48.6% at 8.7% recall, interval
[38.9%, 58.2%]; neither is near 90%.

The earlier run also scored the whole fleet (363,548 drives) and proposed an
action for 3,973 of them (`make score-fleet`) - but, per Section 7, every one
of those proposals was downgraded before a destructive action could fire,
regardless of how confident the model was. The current fleet has not yet
been re-scored with the fix in place.

### 4.1 What the flagging policy does, measured at a fixed date

The figures above score the model. This is the policy the agent runs - the
score, the tier thresholds and the loop around them - measured the way it
would run: the fleet as of a date, judged on what happens next
(`make agent-pit`, ADR 0003). It is for the quarter-aligned model `0c183df0`
(the earlier quarter-aligned window, Q2 2026 as test), which is a different run from the frozen one
above, on a longer test window and a different drive-level convention, so its
precision does not compare directly with the table above.
That window's model is frozen as run `839d92bc` (runbook Section 1.1.1), a
retrain that reproduces `0c183df0` exactly.

Pooled over nine weekly test snapshots (3.14M drive-snapshots, 3,832 failing,
base rate 0.122%):

| Flagged at | Precision (95% CI) | Recall | Likelihood ratio |
|---|---|---|---|
| warn and above | 14.9% [14, 16] | 44.4% | 143 |
| cordon and above | **23.4%** [21, 25] | 29.3% | 250 |
| migrate and above | 35.7% [32, 40] | 10.3% | 456 |
| drain | 40.3% [29, 52] | 1.6% | 552 |

- **Precision follows the base rate; the model does not move.** Over the nine
  dates the failure rate fell 1.76x and cordon-and-above precision went from
  26.8% to 17.9%, while recall stayed between 26% and 33%. Quote precision with
  its base rate.
- **A validation-window run agrees**: cordon and above 22.6% at 28.4% recall.
- **12.8% of false alarms are early, not wrong** - they fail in days 31 to 60.
  Crediting them gives 37.8% against 28.7% on the weeks where that window is
  observed. Reported beside the 30-day figure, never instead of it.
- **An earlier agent backtest reported 61.8% precision at 48.3% recall. That
  figure is withdrawn.** It looked ahead twice (ADR 0003) and measured 17.9%
  once corrected.

Against the goal this is 23.4% at 29.3% recall for cordon and above, and no
tier reaches 90%. Reaching it would need a likelihood ratio near 7,400 at the
pooled base rate, against 250 for cordon and above.

## 5. What was tried to raise precision

Each row was measured on real data and compared with the standard model on
the same drives.

| Approach | Result |
|---|---|
| Fixing a training collapse (class weighting, regularization, early stopping on average precision) | Necessary; produced the working model |
| Hyperparameter variants (tree size, weights, regularization) | Plateau |
| XGBoost in place of LightGBM | Same |
| One model per drive family | Worse than the pooled model for all three largest families |
| A second quarter of data (twice the failing drives) | Same precision, tighter estimate |
| More SMART attributes and velocity, recency and age features | No measurable change |
| Second-stage model on the hard cases | No gain at 14 days. At 30 days, a five-seed repeat showed a repeatable gain in every seed (42.4% to 47.7% at the primary threshold). A direct paired-bootstrap test disagrees between runs: the void run gave a gap of 10.7 points, 95% CI [3.2, 18.5] (excludes 0); the current run gives 3.6 points, 95% CI [-5.5, 12.8] (includes 0). Kept as the default on the five-seed evidence; **not** claimed as a confirmed gain |
| One fleet-wide threshold per action tier, against one threshold per drive model | Measured on test, 2026-10-09: worse. 7 of 81 models cleared the 50-failing-drive gate; against those models' own thresholds they lost 7 to 8 points of recall for under 1 point of precision at the lenient tiers, and at drain lost on both axes. Families sit at different operating points under one threshold because a calibrated score does that to populations of differing risk, not because the threshold is mis-set (ADR 0002) |
| Requiring several consecutive high-score days (as a feature) | Lower precision |
| Requiring a second look, as a decision: flag only if the drive also scored high a week earlier | Costs 0.5 to 3.3 points of recall at matched false-alarm rate, and again on a second window. Combining the two looks by max, mean or rising score does no better. A fifth persistence refutation (ADR 0003) |
| Evidence gates (confidence, maturity) and cooldown | No effect: they downgrade a destructive tier to cordon, which still acts, so they never prevent an action |
| Filtering false positives by SMART attribute, by flag streak or by drive age | Nothing separates them: each priority attribute is non-zero in nearly the same share of true and false positives |
| Per-family recalibration with shrinkage, fitted on validation and judged on test | No gain: recall changes of -1.4 to +0.6 points at matched false-alarm rate, intervals mostly straddling zero. Ranking quality does differ 73-fold by family, but that is within-family discrimination, which a calibration offset cannot repair |
| 30-day horizon in place of 14 days | The one clear gain: 34% to 46% precision near 10% recall, same test period and same split rules |
| Survival model (time to failure, XGBoost AFT) | Same as the classifier |
| Anomaly detection (isolation forest): alone, as a filter, as a feature | Alone far worse; the others within noise |
| Label horizon of 7, 14 and 30 days, same data and splits | Drive-level test AUROC is flat (0.925, 0.924, 0.923): the 14-to-30-day precision gain is the higher base rate, not better separation. A 7-day horizon is no more precise (46% against 57% near 10% recall), so a short-horizon alert model does not reach high precision on this data |
| Survival models, a TCN and rank blends beside LightGBM (merged run `fb58d957`, April-May test) | AFT, the TCN and the blends look more precise at the 10%-recall point (68-70% against 57.1%), one seed, not confirmed; but every model misses 84-88% of failing drives there (178-231 caught of 1,479). For at least 35% recall every model but the LSTM gives 37-41% recall at 46-52% precision (runbook 1.1.2, 1.1.3) |
| A wider Stage 1 net (0.5 to 0.8 of failing drives) | On April-May, about 330 fewer false alarms at 60% recall for 2 fewer catches; the rule fixed in advance kept 0.5, and 0.8 was adopted post hoc. On Q3, +5.1 points at 10% recall for 30 days (1 of 3 seeds), nothing above 20% recall; not confirmed (runbook 4.12, 1.1.4) |
| 44- and 60-day horizons beside 30, identical training rows, Q3 test | Same ranking: AUROC within 0.003, recall at a matched false-alarm rate within 1.3 points, median lead 14-15 days for all; precision rises with the label only (runbook 1.1.4) |
| Cox or AFT as the Stage 2 | Cox matches the LightGBM Stage 2 within about a point; AFT is 2-8 points worse in every horizon (runbook 1.1.4) |
| A stack of the LightGBM and Cox Stage 2 | At the same number flagged, 4 to 11 more failing drives caught than LightGBM at K = 100-500 in 3 of 3 seeds, every horizon; Cox alone does as well at K up to 250; no difference above about 35% recall. Not adopted (runbook 1.1.4a) |
| Action by (tier, failure-mode cluster) instead of tier alone | Rule not passed: precision +4.6 to +8.0 points, recall -6.2 to -8.8 points, on or below the score's own curve (runbook 1.1.6) |
| LSTM over each drive's raw 30-day history, scored on the same drives | Alone slightly worse than LightGBM (AUROC 0.915 against 0.923; recall at a 1-in-1,000 false-alarm rate 20.1% against 26.8%); it overfits after one or two epochs whatever the regularization. It does catch some different drives (14 of 54 shared at the strictest budget), and a rank blend with LightGBM adds a small recall gain (about 2 points at 1-in-2,000 to 1-in-1,000 false alarms; positive in all four seeds, intervals optimistic). Precision at those points is 55% to 68%. Not confirmed; Q3 was used for the LightGBM runs, so it needs a new untouched period (runbook 5.9 to 5.12) |

The consistent ceiling across model families, feature sets, framings and now
policy mechanisms indicates the limit is the information in daily SMART data,
not the model or the loop. On those attributes a false positive looks like a
failing drive.
Most false alarms are drives with abnormal readings that kept running: of 531
healthy test drives alerted by the single model at the broadest threshold, 500
were still in service when the data ended.

## 6. How the system is verified

- Automated tests: unit, property-based, golden-dataset, integration (agent
  with guardrails and simulator, crash recovery, approval and rejection), and
  chaos tests (missing telemetry, action failures, guardrail latency). Lint
  and type checks run with them (`make ci`).
- The data pipeline and model were run end to end on the real two-quarter
  data set, and the trained model scored the whole fleet (`make score-fleet`);
  the current model was then tested on Q3 2026 with a method written before
  the runs (`docs/q3_horizon_and_cascade_run.md`).
- Headline test numbers carry a bootstrap interval over whole drives, and the
  second stage was judged by a paired interval on shared drives rather than
  by comparing two point estimates.
- Every computation touching the test split appends a line to
  `data/audit/test_access_log.jsonl`, so the number of comparisons behind a
  reported gain is recorded rather than recalled.
- A final quarter (Q3 2026) was sealed until 2026-11-06, when its single-shot
  evaluation was skipped by decision and it became the test set of the Q1+Q2
  -> Q3 runs; `make evaluate-frozen` now refuses it, and nothing is sealed at
  present.
- A raw-attribute screen (`scripts/screen_raw_attributes.py`) measured
  single-attribute separation and spatial clustering directly against
  Backblaze's own CSVs, independent of the gold pipeline, before any of
  Section 7's attribute or confidence changes were made.
- The agent, guardrails, rollback and trust scoring were exercised in the
  simulator, not on real hardware.

## 7. Limitations

- **A safety floor was silently unreachable fleet-wide, now fixed.** Two
  multiplicative bugs in `feature_confidence` each capped it below the 0.80
  floor `FEATURE_CONFIDENCE` enforces before any migrate or drain: an
  attribute-coverage term divided by all five priority attributes
  regardless of what a drive's own model reports (capping non-Seagate
  drives at 0.60), and a recency term charged a full day of staleness for
  perfectly fresh daily telemetry (capping every drive-day at 0.6065). Measured
  on the real build, **0% of 62.6 million drive-days cleared the floor**
  before the fix, meaning no migrate or drain was ever reachable, fleet-wide,
  regardless of how confident the model was. Both are fixed (one factor now
  scored per drive model, the other charged only beyond the expected telemetry
  cadence); 86.1% now clear the floor, with nothing newly blocked. Full
  account: `docs/adr/0002-attribute-selection-and-vendor-aware-confidence.md`.
  No automated test caught this, because every guardrail and chaos fixture
  builds drives with every attribute already populated - the bug was only
  visible against real data. `build_gold_features` now refuses to produce a
  fleet where the floor is unreachable by anyone, as a standing check.
- **The agent runs on the real fleet only in simulation, and the API and demo
  still use a demonstration fleet.** `make agent-fleet` binds the loop to the
  real feature table and the trained model with a simulated executor, and the
  agent, the API and batch scoring now all read the trained tier thresholds
  from one place (`src/models/trained_thresholds.py`) rather than the
  hand-picked cut-offs. The API and `make agent-demo` still run against a
  two-drive demonstration fleet.
- **No cost model.** Where to put the thresholds is an economic question - a
  cordon pays when precision x P(save) x C_failure exceeds C_action - and the
  costs have not been supplied. Until they are, the thresholds are operating
  points rather than decisions.
- **Real actions would censor their own outcomes.** Drain a drive about to
  fail and its label is 0, which teaches the next model the wrong thing. The
  control arm (`control_fraction` in `configs/agent.yaml`) holds back a share
  of flagged drives to keep unbiased labels, but defaults to 0.0 and must be
  set before a real executor is connected.
- **Two fleet-safety guardrails need real topology data.** "Never the last
  healthy node" and "keep quorum" are implemented and tested, but in the
  default demonstration wiring nothing supplies them with node and
  replication state, so they cannot fire there.
- **Validation reads higher than test.** On the current model (75.0% against
  59.0% at the primary threshold) about 90% of the gap is the base rate; the
  ranking transfers. On the earlier, now-void run only about half of a smaller
  gap was explained.
- **The two-stage gain is not confirmed.** A direct, paired significance test
  disagrees between the void run (gap excludes 0) and the earlier frozen run
  (gap includes 0), and on Q3 the gain appears only below about 20% recall;
  the second stage stays the default on a five-seed repeat, not on any paired
  test alone.
- **Migrate and drain do not operate on the current model.** Their lift
  targets convert at the validation failure rate to precision met only at the
  very top of the curve (migrate) or not at all (drain), so both flag 6 drives
  a quarter; the agent warns and cordons. Setting those
  targets from the cost of each action is the decision that would change it.
- **Simulation only.** No action touches real hardware, and the API has no
  authentication of any kind. The approval queue, decision trail and
  guardrail violations are now written through to SQLite and survive a
  restart; the last fleet snapshot and prediction list are not, since each
  cycle rebuilds them.
- **No timeout sweep.** The safe-fallback logic for a review that misses its
  SLA exists and is tested, but nothing scans the queue for overdue reviews,
  so an un-answered approval waits indefinitely.
- **One data source.** All results are Backblaze drives. SMART-Z is supported
  in code and has a split reserved for it, but no cross-vendor result exists
  yet, so generalization beyond Backblaze is unmeasured.
- **Action-tier targets are placeholders.** They are expressed as lift, so
  they at least compare across splits, but the numbers themselves should come
  from the real cost of a false alarm for each action.
- **Limited test periods.** The frozen run's figures come from one
  three-week window. The policy measurements use nine weekly snapshots over a
  61-day window, which show base-rate drift alone moving precision between
  17.6% and 31.3%. Q3 2026 gave a second period for the LightGBM two-stage model
  (59.0% precision at 12.5% recall, against 57.1% at 12.0%); the survival
  models, TCN, LSTM and blends were not rerun on it, it has since been read
  many times (horizons, cascade, stack, deep dive, cluster policy), and no
  further untouched period exists. Failures per month fell about 40% across the test
  window, for a reason not yet identified.

## 8. Where to look

| Topic | Document |
|---|---|
| The full project report | `docs/project_report.md` |
| All measured results, run history, how to reproduce | `docs/model_status_and_runbook.md` |
| The Q1+Q2 -> Q3 runs: method, results, threats to validity | `docs/q3_horizon_and_cascade_run.md` |
| The wide Stage 1 net and the cluster action policy | `docs/cascade_wide_net.md`, `docs/cluster_action_policy.md` |
| What the model sees and why | `docs/feature_engineering.md` |
| Commands for every pipeline stage | `docs/pipeline_usage.md` |
| Code structure, agent, guardrails, known gaps | `docs/developer_guide.md` |
| Original goals and plan | `docs/design_goal.md`, `docs/project_plan.md` |
| Why `make train` runs as four processes | `docs/adr/0001-training-memory-isolation.md` |
| The confidence-floor bug and the attribute screen | `docs/adr/0002-attribute-selection-and-vendor-aware-confidence.md` |
| How the flagging policy is measured, and what the loop cannot do | `docs/adr/0003-point-in-time-evaluation-and-the-limits-of-the-loop.md` |
