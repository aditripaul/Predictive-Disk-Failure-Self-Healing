# AI-Based Autonomous System Validation and Reliability Checker

## Predictive Disk-Failure Self-Healing Agent for Server Fleets

**Project report.** Author: Aditri Paul. Date: 6 October 2026.
Repository: `github.com/aditripaul/predictive_disk_failure`, branch `adi_dev`.

---

## Abstract

Storage drives fail, and a failed drive can take data and service with it.
This project builds a system that predicts drive failure from SMART health
telemetry and acts on the prediction autonomously, and then asks the harder
question: can the autonomous system be trusted to act safely on predictions
that are often wrong?

The system has two layers. The first is a self-healing agent: a data pipeline,
a failure-prediction model, and a guardrailed decision loop that chooses among
warn, cordon, migrate and drain. The second is a reliability checker that
audits every decision and gives it a trust score, with a veto when a safety
rule is broken.

The pipeline was run end to end on real Backblaze data: 62.6 million
drive-days from 363,548 drives over the first half of 2026. Measured per
drive, on a later time period than the model was trained on, the model catches
9.7% of failing drives at 46% precision at its primary threshold, and 62% of
failing drives at 10% precision at its most lenient action level. An alerted
drive is about 260 times more likely to fail than a typical drive, and about 1
healthy drive in 5,000 is alerted.

The project's original target of 95% precision was not met. Eleven modelling
approaches were tested; two raised precision (a 30-day prediction window and
a second-stage model) and nine did not. The evidence points to a limit set by
how rare failures are and by how much daily SMART data reveals, not by the
choice of model. The same alerts would be 99% precise on a test set in which
15% of drives fail, the kind of test set most published results use. The
system is designed around this limit: mild actions tolerate false alarms, and
destructive actions pass through independent safety checks and human approval.

---

## 1. Introduction

### 1.1 The problem

Data-centre operators replace failed drives constantly. Predictive
maintenance tries to move the work earlier: detect a failing drive, move its
data, and take it out of service before it fails. The next step, and the
subject of this project, is to let software take those actions without
waiting for a person.

Autonomy changes the cost of a wrong prediction. A wrong warning costs a
glance at a dashboard. A wrong drain takes healthy capacity out of service,
and a drain issued at the wrong moment, for example on the last healthy node
of a replication group, can itself cause the outage it was meant to prevent.
So the central question is not only whether failures can be predicted but
whether a system can act safely on predictions that are uncertain.

### 1.2 What was built

1. A **data pipeline** that turns raw daily SMART telemetry into model-ready
   features and labels at fleet scale, within a fixed memory budget.
2. A **failure-prediction model** with thresholds tied to actions.
3. An **autonomous agent** that runs a Monitor - Analyze - Plan - Execute loop
   with a guardrail engine, human approval, rollback and crash recovery.
4. A **fleet simulator** in which the agent's actions and failures can be
   exercised safely.
5. A **reliability checker** that scores every decision.
6. A **dashboard and API** for operators.

### 1.3 Contributions

- A complete, tested pipeline from raw telemetry to audited action, run on a
  real two-quarter fleet data set.
- An honest measurement of what SMART-based prediction achieves on a
  time-ordered split at real fleet failure rates, with eleven approaches
  compared on the same drives.
- An evaluation method that reports results per drive and at stated failure
  rates, and three corrections to the evaluation found during the work, each
  of which had made results look better than they were.
- A system design in which safety does not depend on the model being right.

---

## 2. Objectives and outcome

| Objective | Target | Outcome |
|---|---|---|
| Prediction precision, per drive, real fleet | at least 95% (later 90%) | **Not met.** 46.2% at the primary threshold |
| Prediction recall | 35-50% (later at least 10%) | 9.7% at the primary threshold; 33.3% at 29.0% precision; 62.5% at the warn level |
| Pipeline runs on real fleet data within a memory cap | 20 GB | **Met.** Two quarters, 62.6 million drive-days |
| Guardrail evaluation latency | under 500 ms | **Met** in test: every one of 2,000 evaluations under the budget |
| Hard-guardrail compliance in simulation | 100% | **Met** in the integration and chaos tests |
| Data loss and quorum violations in simulation | 0 | **Met** in the chaos tests |
| Duplicate actions after crash recovery | 0 | **Met** in the replay test |
| Human review of blocked or uncertain actions | approve and reject paths work | **Met** in the integration tests |
| Every decision audited with a trust score | all | **Met** |
| Live agent driven by the trained model | yes | **Not done.** The model scores the fleet in batch; the agent runs on a demonstration fleet |
| Second data source (SMART-Z) | evaluated | **Not done.** Supported in code, not evaluated |

---

## 3. System design

### 3.1 How one decision flows

```text
SMART telemetry (daily, per drive)
   -> data pipeline: ingest -> harmonize -> features -> labels
   -> failure model: score that the drive fails within 30 days
   -> action tier: warn / cordon / migrate / drain, each with its own cut-off
   -> guardrail engine: 11 rules; a hard violation always blocks
   -> execute in the fleet simulator, or pause for human approval
   -> validate the outcome; roll back if the action failed
   -> reliability checker: trust score, veto on any safety violation
   -> dashboard and audit trail
```

### 3.2 Components

| Component | Role | Technology |
|---|---|---|
| Data pipeline | Bronze (raw), Silver (harmonized) and Gold (features, labels) tables | Polars, PyArrow, Parquet |
| Model | Failure score per drive-day | LightGBM, two stages; MLflow for tracking |
| Agent | Decision loop with checkpoints | LangGraph |
| Guardrails | Rule engine between plan and execution | Python rule catalogue, optional Redis state |
| Simulator | Drives, nodes, replication groups, injected faults | Python |
| Reliability checker | Trust score per decision, drift measures | Python |
| Operator interface | Approval queue, audit trail, fleet view | FastAPI, Streamlit |

### 3.3 Designing for an imperfect model

Failure is rare: in the test period about 1 drive in 570 failed. At that rate
any predictor raises false alarms, so the design makes a false alarm cheap
and a wrong destructive action unlikely.

- **Graduated actions.** A low score only warns. Draining needs the highest
  score.
- **Two independent checks before a destructive action.** Low feature
  confidence or stale telemetry downgrades an action when it is planned, and
  the guardrail engine blocks it again if the first check is bypassed.
- **Fleet-safety rules that ignore the model.** The agent may not drain the
  last healthy node in a failure domain, break replication quorum, or exceed
  a limit on concurrent drains, however confident the model is.
- **Human approval** for anything blocked or uncertain.
- **Rollback and idempotence.** A failed action is compensated; an action
  ledger prevents a crash from repeating an action.
- **Audit.** The reliability checker scores each decision on correctness,
  necessity and timeliness, and sets the score to zero on any safety or
  hard-guardrail violation.

The guardrail catalogue has six hard pre-action rules (score cut-off, feature
confidence, telemetry freshness, last healthy node, quorum, concurrent
drains), three soft rules (high I/O, maintenance window, rate limit) and two
hard post-action checks (data integrity, service continuity).

---

## 4. Data

### 4.1 Source and volume

The data is the public Backblaze drive statistics for Q1 and Q2 2026: one row
per drive per day with SMART attributes and a failure flag.

| | |
|---|---|
| Period | 2026-01-01 to 2026-06-30 (181 daily files) |
| Drive-days | 62,565,487 |
| Drives | 363,548 |
| Failing drives in the test period | 621 of about 354,000 (0.18%) |

### 4.2 Processing

Ingestion reads every SMART column as a number and replaces rows by drive and
date, after an early run failed on a column that was empty at the start of
one file and was read as text. Harmonization maps vendor attribute numbers to
canonical names and records telemetry gaps. The whole pipeline works in
batches of whole drives and writes results to disk batch by batch, so its
memory use is set by the batch size and not by the size of the data. A
watchdog stops any process whose resident memory exceeds 20 GB.

### 4.3 Labels and splits

A drive-day is positive if the drive fails within the horizon (30 days) after
that day, negative if the drive is observed for the whole horizon without
failing, and unlabelled otherwise.

The split is by time:

| Split | Dates | Rule |
|---|---|---|
| Train | to 2026-03-16 | stops 30 days before the training cut-off, so every training label is settled before validation begins |
| Validation | 2026-04-16 to 2026-05-10 | only dates after the training cut-off |
| Test | 2026-05-11 to 2026-05-31 | ends 30 days before the data does, so every test label is complete |

Ten percent of drives are also held out of training altogether. Random
splitting is not used: it would let the model see each drive's future.

---

## 5. Features

The model sees 213 features per drive-day (`docs/feature_engineering.md`).

- **Core error counters** (SMART 5, 187, 188, 197, 198: reallocated, reported
  uncorrectable, command timeout, pending and offline uncorrectable sectors)
  receive the full family: rolling mean, maximum and standard deviation over
  7, 14 and 30 days, deltas, slopes, counts of days with increases, spike
  counts, and z-scores against other drives of the same model.
- **Context attributes** (seek error rate, power-on hours, temperature, CRC
  error count) receive a light family: the value and its 7- and 30-day change.
- **Cross-attribute features:** total active defects and its velocity and
  acceleration, days since each counter last increased, age, ratios such as
  uncorrectable errors per power-on hour, and a temperature spike measure.
- **Telemetry quality:** coverage and freshness, combined into a feature
  confidence that the agent uses to block destructive actions on thin data.

The tiering keeps the feature table narrow enough for the memory budget.

---

## 6. Model

### 6.1 First stage

A LightGBM binary classifier trained on 5.0 million rows: every failing row
plus a sample of the 22.8 million healthy training rows. Failing rows are
weighted by the square root of the class ratio. The trees are regularized
(31 leaves, at least 200 rows per leaf, L2 penalty 10, capped leaf output)
and training stops when average precision on validation stops improving
(220 trees in the reported run).

These choices came from a failure early in the project. With the standard
class-balancing option the model collapsed on real data, producing extreme
scores for a handful of drives. A later attempt at early stopping watched the
wrong metric and stopped after about 13 trees. Regularization, square-root
weighting and early stopping on average precision alone produced the first
working model.

### 6.2 Second stage

A second LightGBM re-ranks the drive-days the first model finds most
suspicious: those scoring above the validation threshold that still catches
half of the failing drives. It is trained only on such rows (9,454 training
rows, 5,262 of them failing), with the first model's score as an extra
feature, so all of its capacity goes to telling failing drives from the
healthy drives that resemble them. Training rows receive their first-stage
score from a model that never saw their drive, so the second stage learns from
scores like those it meets in use. Rows below the candidate threshold keep
their first-stage score, so the lenient action levels are unaffected.

### 6.3 Thresholds and action tiers

All thresholds are chosen on validation at the drive level and then applied
unchanged to test. The primary threshold is the most precise point that
still catches 10% of failing drives. Each action tier has its own threshold,
set from the precision that action can tolerate.

---

## 7. Evaluation method

- **Per drive, not per drive-day.** A failing drive produces about thirty
  nearly identical positive rows, so row-level figures count it thirty times.
  A failing drive is *caught* if any day in its window scores above the
  threshold; a healthy drive is a *false alarm* if any of its days does.
- **Later data only.** The test period follows the validation period, which
  follows training.
- **Thresholds from validation.** No number in this report was obtained by
  tuning on test.
- **Precision is reported with its failure rate.** Precision depends on how
  common failures are among the drives judged. For catch rate *r*,
  false-alarm rate *f* and failure rate *p*,
  precision = *r·p* / (*r·p* + *f*·(1 − *p*)). The catch rate and false-alarm
  rate do not depend on *p*; precision does. Results are therefore given at
  the fleet's own failure rate and, restated, at 15% and 50%, the rates
  typical of published evaluations. The restated figures are arithmetic on
  the measured rates, not separate experiments.

### 7.1 Corrections made to the evaluation

Three errors were found and fixed during the project. Each had flattered the
results, and each is recorded with its effect in
`docs/model_status_and_runbook.md`.

1. **The test set ran past its intended end.** In the last days of the data a
   healthy drive-day has no label yet, but a failing one does, so those days
   added failing drives with no healthy drives beside them. Correcting this
   lowered 14-day precision near 10% recall from 39.6% to 33.6%.
2. **Training labels reached into the validation period.** A training row
   dated just before the cut-off is labelled by what happens after it.
   Training now stops one horizon earlier.
3. **Validation contained rows from the training period** (the held-out
   drives' history), which made validation look easier than the future.
   Validation now holds only later dates. After this, a threshold chosen to
   catch 20% of failing drives on validation caught 18.7% on test with the
   single model; before, it caught 12.6%.

---

## 8. Results

Backblaze Q1 + Q2 2026, 30-day horizon, two-stage model. Test set: 621
failing and 353,328 healthy drives.

### 8.1 Precision and recall

| Operating point | Failing drives caught | Healthy drives alerted | Precision, real fleet (0.18% fail) | At 15% failing | At 50% failing | Lift |
|---|---|---|---|---|---|---|
| Strictest | 18 (2.9%) | 22 (0.006%) | 45.0% | 98.8% | 99.8% | 256 |
| **Primary threshold** | **60 (9.7%)** | **70 (0.020%)** | **46.2%** | **98.9%** | **99.8%** | **263** |
| Broader | 129 (20.8%) | 164 (0.046%) | 44.0% | 98.7% | 99.8% | 251 |
| Broadest | 207 (33.3%) | 506 (0.143%) | 29.0% | 97.6% | 99.6% | 165 |

Lift is how many times more likely an alerted drive is to fail than a drive
chosen at random.

### 8.2 Action tiers

| Tier | Precision target on validation | Test precision | Test recall |
|---|---|---|---|
| Warn | 15% | 9.9% | 62.5% |
| Cordon | 25% | 17.4% | 51.2% |
| Migrate | 40% | 29.1% | 33.0% |
| Drain | 60% | 46.0% | 10.3% |

Test precision falls short of each validation target. More drives failed in
the validation period than in the test period (0.27% against 0.18%), which
accounts for about half of the difference; the rest is unexplained. At a
fixed failure rate of 15%, validation and test agree (99.0% and 98.9% at the
primary threshold).

### 8.3 Other measures at the primary threshold

- Warning time: median 14 days before failure, mean 17.9.
- False-alarm rate: 70 of 353,328 healthy drives, about 1 in 5,000.
- Fleet scoring: the model scored all 363,548 drives and proposed an action
  for 3,973.

---

## 9. What was tried

Every row was measured on real data against the standard model on the same
drives.

| # | Approach | Result |
|---|---|---|
| 1 | Fixing the training collapse (weighting, regularization, early stopping) | Necessary: produced the working model |
| 2 | Hyperparameter variants | Plateau |
| 3 | XGBoost in place of LightGBM | Same (test ranking 0.208 against 0.213) |
| 4 | One model per drive family | Worse than the pooled model for all three largest families |
| 5 | A second quarter of data | Same precision, tighter estimate |
| 6 | Four more SMART attributes; velocity, recency and age features | No measurable change (all variants within 0.01 of each other) |
| 7 | Requiring several consecutive high-score days | Lower precision |
| 8 | **30-day horizon in place of 14 days** | **Gain: 33.6% to 46.3% precision near 10% recall, same test period** |
| 9 | Survival model (time to failure) | Same as the classifier (0.216 against 0.213) |
| 10 | Anomaly detection (isolation forest), alone, as a filter, as a feature | Alone far worse (24% against 46%); the others within noise |
| 11 | **Second-stage model** | **Gain: 42.4% to 47.7% at the primary threshold, averaged over five seeds and positive in all five** |

The second-stage gain was first seen in single runs, where it rested on about
fifteen drives. It was adopted only after a five-seed repeat showed it in
every seed at the 10% and 20% operating points. The same repeat showed that
the single model alone varies between 40.9% and 43.6% from seed to seed, a
useful measure of how much weight a difference of two points between single
runs deserves.

---

## 10. Validation of the autonomous system

The agent, guardrails, simulator and reliability checker were validated with
automated tests, not on real hardware.

| Test layer | Tests | What it covers |
|---|---|---|
| Unit | 389 | Features, labels, model code, guardrail rules, trust score, API |
| Property-based | 10 | Invariants of hysteresis, trust score and metrics over generated inputs |
| Golden data | 3 | Feature pipeline against a fixed reference data set |
| Integration | 16 | Full decision cycles with the real guardrail engine and simulator |
| Chaos | 5 | Stale telemetry, correlated multi-drive failure, action timeout, human-review timeout, guardrail latency |
| Smoke | 3 | End-to-end start-up |

Behaviours demonstrated by these tests:

- A high-risk drive is drained through the simulator end to end.
- A low-confidence prediction is downgraded to cordon.
- A drain that would break quorum, or that targets the last healthy node, is
  blocked and routed to human review; approval resumes and executes it,
  rejection ends without executing.
- After a simulated crash, replay does not repeat the action.
- An action that times out triggers a compensating action with no data loss.
- An unanswered review falls back to the safe action.
- Hysteresis requires a risk to persist over cycles before escalation, and a
  cooldown prevents an immediate repeat drain.
- Each of 2,000 guardrail evaluations completes within the 500 ms budget.

---

## 11. Discussion

### 11.1 Why precision stops where it does

The false-alarm rate is already very low: 1 healthy drive in 5,000. Precision
is still under 50% because failing drives are rarer still. To reach 90%
precision at the primary threshold the model would have to alert no more than
about 7 healthy drives out of 353,328, ten times fewer than it does.

Three observations suggest the limit is in the data and not the model.
Different model families, feature sets and problem framings all arrive at the
same place. Most false alarms are drives whose SMART readings really are
abnormal: of 531 healthy drives alerted by the single model at its broadest
threshold, 500 were still in service a month later. And many failing drives
give no warning in their SMART readings at all, which caps recall.

### 11.2 What did help, and why

Lengthening the horizon from 14 to 30 days helped because the model was being
marked wrong for warnings that came early: of 330 test drives it flagged
that did not fail within 14 days, 43 failed within the following six weeks. The second stage helped because, once the easy healthy drives are set
aside, a model trained only on the hard cases separates them slightly better.

### 11.3 Comparison with published work

Published studies commonly report precision above 90%. They are usually
measured on one drive model, with random splits, and on test sets with far
more failures than a fleet contains. On a test set with 15% failing drives
this model's alerts are 98.9% precise at the primary threshold. The two
statements are consistent: they describe the same alerts judged against
different populations. This report gives the real-fleet figure first because
it is what an operator would experience.

### 11.4 What the result means for autonomy

A model that is right less than half the time at its strictest setting cannot
be allowed to drain drives unsupervised, and this system does not allow it.
What the model provides is a short, highly enriched list: about 130 drives
flagged out of 354,000, nearly half of which will fail within a month. That
is useful input to a system in which the consequential decisions are checked
by rules that do not depend on the model and, where needed, by a person.

---

## 12. Limitations

- **The live agent is not yet driven by the trained model.** The model scores
  the real fleet in batch and proposes actions. The agent and API run against
  a small demonstration fleet, and the agent uses fixed score cut-offs
  instead of the tier thresholds derived in training.
- **Two fleet-safety guardrails need real topology data.** The last-healthy-
  node and quorum rules are implemented and tested, but the demonstration
  wiring supplies no node or replication state, so they cannot fire there.
- **Simulation only.** No action touches real hardware. The API has no
  authentication, and its audit store does not survive a restart.
- **One data source and one half-year.** All results are Backblaze drives
  from January to June 2026, with a single three-week test period. The
  five-seed test shows the second-stage gain does not depend on training
  randomness; it does not show how results would vary in another period.
- **Validation precision exceeds test precision**, and only about half of the
  difference is explained.
- **Action-tier precision targets are placeholders.** They should be set from
  the real cost of a false alarm for each action.
- **Explanations describe the first-stage model only.**

---

## 13. Future work

1. Drive the live agent from the trained model, with the trained tier
   thresholds and real topology data for the fleet-safety guardrails.
2. Evaluate on a further quarter, both to test stability over time and to
   allow a 60- to 90-day horizon.
3. Add signals beyond daily SMART: I/O latency, error logs and workload.
4. Set tier targets from operational cost, and measure the precision of
   actions taken after human review.
5. Evaluate on the second data set to test generalization across vendors.
6. Schedule the trust-score resolution and drift checks that currently run
   only on demand.

---

## 14. Conclusion

The project delivered a working, tested system that goes from raw fleet
telemetry to an audited action, and ran it on a real fleet of 363,548 drives.

It did not reach its precision target, and the evidence indicates that the
target is not reachable from daily SMART data at real fleet failure rates:
eleven approaches were tried, and the two that helped moved precision at the
primary threshold from about 34% to about 46%. The model is nonetheless
strongly informative, with alerts 260 times more likely than chance to be
right and a false-alarm rate of 1 in 5,000.

The more durable result is the design. Because safety rests on graduated
actions, independent guardrails, human approval, rollback and audit, the
system remains safe to operate with a model of this accuracy, and would
become more autonomous, not differently built, as better signals improve the
model.

---

## References

1. Backblaze. *Hard Drive Data and Stats* (drive statistics data set), Q1 and
   Q2 2026.
2. Amram, M., et al. *Interpretable predictive maintenance for hard drives.*
   2021.
3. Klein, A. *Using Machine Learning to Predict Hard Drive Failures.*
   Backblaze blog, 12 October 2021.
4. Lu, S., et al. *Making Disk Failure Predictions SMARTer!* USENIX FAST 2020.
5. Wei, et al. *SMART-Z data set.* Scientific Data, 2025.
6. Kephart, J. and Chess, D. *The Vision of Autonomic Computing.* 2003.
7. Chen, T. and Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD
   2016.
8. Ke, G., et al. *LightGBM: A Highly Efficient Gradient Boosting Decision
   Tree.* NeurIPS 2017.
9. Liu, F. T., Ting, K. M. and Zhou, Z.-H. *Isolation Forest.* ICDM 2008.
10. LangGraph documentation.

---

## Appendix A. Reproducing the results

```bash
make install
make download-backblaze        # quarters set in configs/data.yaml
make ingest-backblaze
make build-silver
make build-features
make build-labels
make train                     # two-stage model, reports, model card
make score-fleet
make final-report              # runs make plots first
```

Model comparisons, on arrays kept from a training run:

```bash
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw xgboost xgboost_aft --skip-baseline --two-stage --two-stage-seeds 5 --anomaly-stage --deep-dive reg_spw"
```

Outputs: the evaluation report and final report under
`data/audit/data_quality_reports/`, the model card under
`data/audit/model_cards/`, plots under `data/audit/plots/`, fleet scores
under `data/audit/predictions/`.

## Appendix B. Key settings

| Setting | Value |
|---|---|
| Horizon | 30 days |
| Splits | train to 2026-04-15 (rows to 2026-03-16), validation to 2026-05-10, test to 2026-05-31 |
| Training rows | 5,000,295 used of 22,844,327 |
| Features | 213 |
| First stage | LightGBM, 220 trees, 31 leaves, at least 200 rows per leaf, L2 10, learning rate 0.05 |
| Second stage | LightGBM, 38 trees, at least 50 rows per leaf; candidates above first-stage score 0.830 |
| Class weighting | square root of the healthy-to-failing ratio |
| Memory cap | 20 GB resident per process |

## Appendix C. Where to find more

| Topic | Document |
|---|---|
| Short overview | `docs/system_summary.md` |
| Every measured result and the run history | `docs/model_status_and_runbook.md` |
| Features and the reasons for them | `docs/feature_engineering.md` |
| Commands and options | `docs/pipeline_usage.md` |
| Code structure and known gaps | `docs/developer_guide.md` |
| Operating the system | `docs/user_guide.md` |
| Original goals and plan | `docs/design_goal.md`, `docs/project_plan.md`, `docs/dataset_strategy.md` |
