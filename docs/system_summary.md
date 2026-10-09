# System summary: a self-healing disk agent that can be trusted

Status: 2026-10-09 (confidence-floor fix and attribute screen, ADR 0002).
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
| Data pipeline | Turns 62.6 million drive-days from 363,548 drives (Backblaze Q1 + Q2 2026) into 246 features per drive-day, within a 20 GB memory cap | `pipelines/`, `src/features/` |
| Failure model | LightGBM classifier, with a second LightGBM that re-ranks its highest-scoring drive-days; outputs a failure score per drive-day | `pipelines/train_model.py`, `src/models/two_stage.py` |
| Action tiers | Maps the probability to an action. Mild actions accept more false alarms; destructive ones need more confidence | `src/models/threshold.py` |
| Agent | A Monitor - Analyze - Plan - Execute loop (LangGraph), checkpointed so a crash cannot repeat an action | `src/agent/` |
| Guardrails | Six hard pre-action rules (score cut-off, feature confidence, fresh telemetry, never the last healthy node, keep replication quorum, limit concurrent drains), three soft ones (high I/O, maintenance window, rate limit), and two hard post-action checks (data integrity, service continuity) | `src/guardrails/` |
| Human review | Blocked or uncertain destructive actions wait for an operator to approve or reject | `src/api/` |
| Fleet simulator | Drives, nodes and replication groups with injected faults, so actions and rollbacks can be exercised without real hardware | `src/simulator/` |
| Reliability checker | Scores every decision on correctness, necessity and timeliness; any safety or hard-guardrail violation sets the trust score to zero | `src/reliability/` |
| Dashboard | Fleet health, approval queue, trust trend, audit trail | `src/dashboards/` |

## 3. Why the system does not need a perfect model

Disk failure is rare: in the test period 621 of about 354,000 drives failed,
roughly 1 in 570. At that rate any predictor produces false alarms, so the
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

Setup: Backblaze Q1 + Q2 2026, 30-day horizon, time-ordered split. Training
uses drive-days to 2026-03-16 (every training label is settled by
2026-04-15), validation is 2026-04-16 to 2026-05-10, test is 2026-05-11 to
2026-05-31. Results are per drive, on test, with thresholds chosen on
validation. Test has 621 failing and 353,328 healthy drives.

**Precision depends on how rare failures are in the test set**, so every
operating point below is shown twice: at this fleet's own failure rate, which
is what an operator would experience, and restated for a test set in which
15% of drives fail, which is how most published results are reported and the
figure to compare with the literature (Amram et al., 2021, used about 15%
positives). The model and its alerts are identical either way; only the mix
of drives it is judged on changes.

**The model of record** is a frozen training run, `29e338248d8948ccb41ee58db5321e76`,
kept unchanged so that the sealed quarter and the cross-vendor set can each be
scored against it exactly once:

| Operating point | Precision, real fleet (0.18% fail) | Failing drives caught | Precision, 15% fail |
|---|---|---|---|
| **Primary threshold** | **48.6%**, 95% CI [38.9%, 58.2%] | **8.7%** | 99.0% |
| Broader | 39.7% | 20.3% | - |
| Broadest | 29.5% | 33.3% | - |

At the primary threshold an alerted drive is about 277 times more likely to
fail than one picked at random, and fewer than 1 healthy drive in 6,000 is
alerted. The interval is a 1000-draw bootstrap that resamples whole drives,
so a drive's many near-identical rows move together.

**An earlier freeze is void, not merely superseded.** A prior run,
`0ee06c01ff2a427ca76011fb1afb8ce3`, was trained before two bugs in
`feature_confidence` were fixed (see Section 7 and ADR 0002); its own
feature columns are now far outside the distribution the rebuilt data
produces, so it may not be scored on the sealed quarter or SMART-Z even
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
  drive picked at random (277 times for the current frozen run above).
- **Warning time:** median 14 days before the failure (mean 17.9).
- **False-alarm rate:** 70 of 353,328 healthy drives, about 1 in 5,000.
- **Validation agrees with test once the failure rate is held fixed:** at
  15% failing, precision at the primary threshold is 99.0% on validation and
  98.9% on test.

**Against the project's stated goal.** The original target was 95% precision
at 35-50% recall on the real fleet; it was later set to 90% at 10% recall.
Neither is met: the real-fleet figure is 48.6% at 8.7% recall, and the
interval around it ([38.9%, 58.2%]) is nowhere near 90%.

The earlier run also scored the whole fleet (363,548 drives) and proposed an
action for 3,973 of them (`make score-fleet`) - but, per Section 7, every one
of those proposals was downgraded before a destructive action could fire,
regardless of how confident the model was. The current fleet has not yet
been re-scored with the fix in place.

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
| Requiring several consecutive high-score days | Lower precision |
| 30-day horizon in place of 14 days | The one clear gain: 34% to 46% precision near 10% recall, same test period and same split rules |
| Survival model (time to failure, XGBoost AFT) | Same as the classifier |
| Anomaly detection (isolation forest): alone, as a filter, as a feature | Alone far worse; the others within noise |

The consistent ceiling across model families, feature sets and framings
indicates the limit is the information in daily SMART data, not the model.
Most false alarms are drives with abnormal readings that kept running: of 531
healthy test drives alerted by the single model at the broadest threshold, 500
were still in service when the data ended.

## 6. How the system is verified

- Automated tests: unit, property-based, golden-dataset, integration (agent
  with guardrails and simulator, crash recovery, approval and rejection), and
  chaos tests (missing telemetry, action failures, guardrail latency). Lint
  and type checks run with them (`make ci`).
- The data pipeline and model were run end to end on the real two-quarter
  data set, and the trained model scored the whole fleet (`make score-fleet`).
- Headline test numbers carry a bootstrap interval over whole drives, and the
  second stage was judged by a paired interval on shared drives rather than
  by comparing two point estimates.
- Every computation touching the test split appends a line to
  `data/audit/test_access_log.jsonl`, so the number of comparisons behind a
  reported gain is recorded rather than recalled.
- A final quarter (Q3 2026) is sealed: no training, validation or test step
  reads it. It is scored once, against the frozen run, by
  `make evaluate-frozen`.
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
- **The live agent loop is not yet driven by the trained model.** The model
  scores the real fleet in batch and proposes actions, but the agent and API
  run against a small demonstration fleet. The agent also still uses fixed
  score cut-offs from `configs/agent.yaml` instead of the tier thresholds
  derived in training.
- **Two fleet-safety guardrails need real topology data.** "Never the last
  healthy node" and "keep quorum" are implemented and tested, but in the
  default demonstration wiring nothing supplies them with node and
  replication state, so they cannot fire there.
- **Validation read higher than test on the earlier, now-void run** (60.4%
  against 46.2% precision at the primary threshold). About half of that gap
  was because more drives failed in the validation period (0.27% against
  0.18%); the rest was unexplained. Not yet re-measured on the current
  frozen run.
- **The two-stage gain is not confirmed.** A direct, paired significance test
  disagrees between the void run (gap excludes 0) and the current run (gap
  includes 0); the second stage stays the default on a five-seed repeat that
  showed a smaller gain in every seed, not on either paired test alone.
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
- **One test period.** All results come from a single three-week test window.
  The sealed quarter exists to test stability over time, and has not been
  scored yet.

## 8. Where to look

| Topic | Document |
|---|---|
| The full project report | `docs/project_report.md` |
| All measured results, run history, how to reproduce | `docs/model_status_and_runbook.md` |
| What the model sees and why | `docs/feature_engineering.md` |
| Commands for every pipeline stage | `docs/pipeline_usage.md` |
| Code structure, agent, guardrails, known gaps | `docs/developer_guide.md` |
| Original goals and plan | `docs/design_goal.md`, `docs/project_plan.md` |
| Why `make train` runs as four processes | `docs/adr/0001-training-memory-isolation.md` |
| The confidence-floor bug and the attribute screen | `docs/adr/0002-attribute-selection-and-vendor-aware-confidence.md` |
