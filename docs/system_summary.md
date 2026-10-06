# System summary: a self-healing disk agent that can be trusted

Status: 2026-10-06. This is the entry point for a reviewer. It describes what
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
| Data pipeline | Turns 62.6 million drive-days from 363,548 drives (Backblaze Q1 + Q2 2026) into about 213 features per drive-day, within a 20 GB memory cap | `pipelines/`, `src/features/` |
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

**Precision depends on how rare failures are in the test set**, so each
operating point is shown at three failure rates. The model and its alerts are
identical in all three columns; only the mix of drives it is judged on
changes. The first column is what an operator of this fleet would see. The
other two are how most published results are reported, and are the columns to
compare with the literature (Amram et al., 2021, used about 15% positives).

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

Other measures at the primary threshold:

- **Lift:** an alerted drive is about 263 times more likely to fail than a
  drive picked at random.
- **Warning time:** median 14 days before the failure (mean 17.9).
- **False-alarm rate:** 70 of 353,328 healthy drives, about 1 in 5,000.
- **Validation agrees with test once the failure rate is held fixed:** at
  15% failing, precision at the primary threshold is 99.0% on validation and
  98.9% on test.

**Against the project's stated goal.** The original target was 95% precision
at 35-50% recall on the real fleet; it was later set to 90% at 10% recall.
Neither is met: the real-fleet figure is 46% at 9.7% recall.

The trained model scored the whole fleet (363,548 drives) and proposed an
action for 3,973 of them (`make score-fleet`).

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
| Second-stage model on the hard cases | No gain at 14 days. At 30 days a modest, repeatable gain: over five seeds, 42.4% to 47.7% precision at the primary threshold at the same recall. Adopted: it is part of the model reported in section 4 (the single model on the same run gave 41.4% at 8.5% recall, where the pair gives 46.2% at 9.7%) |
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
- The agent, guardrails, rollback and trust scoring were exercised in the
  simulator, not on real hardware.

## 7. Limitations

- **The live agent loop is not yet driven by the trained model.** The model
  scores the real fleet in batch and proposes actions, but the agent and API
  run against a small demonstration fleet. The agent also still uses fixed
  score cut-offs from `configs/agent.yaml` instead of the tier thresholds
  derived in training.
- **Two fleet-safety guardrails need real topology data.** "Never the last
  healthy node" and "keep quorum" are implemented and tested, but in the
  default demonstration wiring nothing supplies them with node and
  replication state, so they cannot fire there.
- **Validation still reads higher than test on the real fleet** (60.4%
  against 46.2% precision at the primary threshold). About half of that gap
  is because more drives failed in the validation period (0.27% against
  0.18%); the rest is unexplained. The share of failing drives caught now
  carries over closely (10.3% on validation, 9.7% on test).
- **Simulation only.** No action touches real hardware. The API has no
  authentication, and the audit store does not survive a restart.
- **One data source.** All results are Backblaze drives; the second data set
  (SMART-Z) is supported in code but has not been evaluated.
- **Action-tier precision targets are placeholders** and should be set from
  the real cost of a false alarm for each action.

## 8. Where to look

| Topic | Document |
|---|---|
| All measured results, run history, how to reproduce | `docs/model_status_and_runbook.md` |
| What the model sees and why | `docs/feature_engineering.md` |
| Commands for every pipeline stage | `docs/pipeline_usage.md` |
| Code structure, agent, guardrails, known gaps | `docs/developer_guide.md` |
| Original goals and plan | `docs/design_goal.md`, `docs/project_plan.md` |
