# AI-Based Autonomous System Validation & Reliability Checker

### Predictive Disk-Failure Self-Healing Agent for Server Fleets

A project focused on building and validating an autonomous self-healing agent that predicts disk failure from SMART telemetry, executes guardrailed remediation actions, and proves that each autonomous decision is correct, safe, necessary, timely, and explainable.

---

## 1. Project Overview

Hardware failure in data centers is inevitable. Modern platforms increasingly use predictive maintenance to detect failing components before they cause outages. The next step is autonomous remediation: systems that can isolate, migrate, cordon, or drain failing drives without waiting for human intervention.

However, autonomy introduces a critical question:

> **Can we trust the autonomous system to act safely?**

This project builds a two-layer system:

1. **Self-Healing Agent**
   - Ingests SMART telemetry.
   - Predicts impending disk failure.
   - Plans remediation actions.
   - Executes actions inside a guardrailed LangGraph MAPE-K loop.

2. **Reliability Checker / Meta-Validator**
   - Audits every autonomous decision.
   - Checks correctness, safety, necessity, timeliness, and guardrail compliance.
   - Produces a trust score.
   - Applies veto semantics for safety or hard-guardrail violations.
   - Generates explainability and audit logs.

The project is simulation-first. The agent operates against a simulated fleet before any consideration of real-world deployment.

---

## 2. Core Problem

Disk failures often show early warning signs through SMART telemetry, such as:

- increasing reallocated sector counts;
- growing pending sector counts;
- uncorrectable errors;
- degraded read/seek error rates;
- spin retries;
- command timeouts.

However, raw daily SMART values are noisy. A single abnormal reading may be transient, while a drive may appear stable even while its degradation rate is accelerating.

Therefore, this project does not merely detect abnormal SMART values. It learns **failure trajectories** using time-window features, slopes, acceleration, event counts, telemetry freshness, and lifecycle context.

Because autonomous remediation can be operationally expensive or unsafe, the system follows a **precision-first** policy:

- false positives are preferable to unsafe false negatives;
- destructive actions require high confidence and guardrail approval;
- stale or low-confidence telemetry cannot trigger destructive actions;
- every decision must be auditable.

---

## 3. Key Capabilities

- SMART telemetry ingestion from Backblaze and SMART-Z datasets.
- RAM-efficient data processing using Polars, DuckDB, PyArrow, and Parquet.
- Trajectory-based feature engineering:
  - 7/14/30-day rolling aggregates;
  - deltas, slopes, and acceleration;
  - threshold-crossing and event-count features;
  - telemetry-gap and staleness features;
  - lifecycle and model-relative features.
- High-precision disk-failure prediction using XGBoost or LightGBM.
- Optional LSTM sequence-model comparison.
- LangGraph-based MAPE-K autonomous agent.
- Guardrail engine with hard and soft safety rules.
- Human-in-the-loop approval queue.
- Fleet simulator with chaos injection and compensating actions.
- Reliability Checker with trust scoring and veto semantics.
- Dashboards for fleet health, trust trends, approvals, and audit trails.
- Replayable, checkpointed, crash-safe autonomous execution.

---

## 4. High-Level Architecture

```text
                 ┌────────────────────────────────────────────────────┐
                 │              SERVER FLEET / SIMULATOR              │
                 │                                                    │
                 │  SMART telemetry · drive states · node states      │
                 │  replication groups · action outcomes · I/O load   │
                 └──────────────────────────┬─────────────────────────┘
                                            │
                                            │ telemetry / events
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────  ────┐
│                     LAYER 1 · SELF-HEALING AGENT                                   │
│                     LangGraph MAPE-K Runtime                                       │
│                                                                                    │
│  ┌─────────┐    ┌─────────┐    ┌──────────────────┐    ┌───────────┐    ┌────────┐ │
│  │ MONITOR │──▶│ ANALYZE │──▶│ PLAN +           │──▶│ EXECUTE   │──▶│VALIDATE│ │
│  │         │    │         │    │ GUARDRAIL ENGINE │    │           │    │        │ │
│  │ ingest  │    │ predict │    │ action proposal  │    │ cordon /  │    │ verify │ │
│  │ fleet   │    │ p_fail  │    │ threshold check  │    │ migrate / │    │ outcome│ │
│  │ state   │    │ rank    │    │ hard/soft rules  │    │ drain     │    │        │ │
│  └─────────┘    └─────────┘    └──────┬───────────┘    └───────────┘    └┬───────┘ │
│                                       │                                  │         │
│                                       │ blocked / uncertain              │         │
│                                       ▼                                  │         │
│                              ┌────────────────┐                          │         │
│                              │ HUMAN REVIEW   │                          │         │
│                              │ FastAPI queue  │                          │         │
│                              └────────────────┘                          │         │
│                                                                          │         │
└──────────────────────────────────────────────────────────────────────────┼─────────┘
                                                                           │
                                                                           │ decision +
                                                                           │ outcome record
                                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 2 · RELIABILITY CHECKER                                │
│                     Meta-Validator                                               │
│                                                                                  │
│  correctness · necessity · safety · timeliness · guardrail compliance            │
│                                                                                  │
│  → Trust Score                                                                   │
│  → Veto if safety violation or hard-guardrail violation                          │
│  → Explainability log                                                            │
│  → Drift / retrain signal                                                        │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────────┐
                         │   DASHBOARDS & AUDIT TRAIL   │
                         │                              │
                         │ fleet health · trust trend   │
                         │ approval queue · compliance  │
                         │ decision replay · alerts     │
                         └──────────────────────────────┘
```

---

## 5. Design Principles

| Principle | Description |
|---|---|
| Safety over availability | The system prefers false positives over unsafe destructive actions |
| Precision-first autonomy | High-confidence thresholds are required before cordon/migrate/drain |
| Explainability by default | Every prediction and action must be explainable |
| Idempotent actions | Retries must not duplicate migrations, drains, or compensations |
| Crash-safe operation | LangGraph checkpoints allow safe resume after failure |
| Lean agent state | LangGraph state stores IDs and metadata, not large telemetry payloads |
| RAM-aware data engineering | Polars, DuckDB, and Parquet are used instead of pandas-heavy workflows |
| Simulation-first validation | Autonomous behavior is validated in a simulated fleet |

---

## 6. Project Documentation

| Document | Purpose |
|---|---|
| `docs/design_goal.md` | Project goal, architecture, safety invariants, guardrails, reliability framework, and success criteria |
| `docs/dataset_strategy.md` | Datasets, preprocessing, labeling, feature engineering, leakage prevention, and feature governance |
| `docs/project_plan.md` | Phase-wise execution plan, milestones, deliverables, tools, and exit criteria |
| `docs/developer_guide.md` | How the implemented codebase is organized, how the pieces fit together, and how to extend it |
| `docs/user_guide.md` | How to run, operate, and approve/reject decisions from the system — no code-reading required |

---

## 7. Repository Structure

```text
project-root/
├── README.md
├── Makefile
├── pyproject.toml
│
├── docs/
│   ├── design_goal.md
│   ├── dataset_strategy.md
│   ├── project_plan.md
│   ├── developer_guide.md
│   └── user_guide.md
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   ├── agent.yaml
│   └── guardrails.yaml
│
├── data_contracts/
│   └── schemas.py
│
├── src/
│   ├── ingest/
│   ├── preprocess/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── agent/
│   ├── guardrails/
│   ├── reliability/
│   ├── simulator/
│   ├── api/
│   └── dashboards/
│
├── pipelines/
│   ├── ingest_backblaze.py
│   ├── build_silver.py
│   ├── build_gold_features.py
│   ├── build_labels.py
│   └── train_model.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── chaos/
│   └── smoke/
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── synthetic/
│   └── audit/
│
├── mlflow/
├── notebooks/
└── scripts/
```

---

## 8. Technology Stack

| Area | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Checkpointing | SQLite / LangGraph saver |
| Fleet-state cache | Redis (declared dependency; not yet wired — currently an in-memory stand-in, `src/guardrails/operational_state.py`) |
| DataFrame processing | Polars |
| SQL engine | DuckDB |
| Columnar I/O | PyArrow |
| Storage format | Parquet + ZSTD |
| ML models | XGBoost / LightGBM |
| Optional sequence model | PyTorch LSTM |
| Experiment tracking | MLflow |
| API | FastAPI |
| Dashboard | Streamlit |
| Testing | pytest, Hypothesis, optional Locust |
| Logging | structlog / JSON logs (declared dependency; not yet wired — pipelines currently use plain `print()`) |
| Optional tracing | OpenTelemetry |

---

## 9. Datasets

### Backblaze Hard Drive Stats

Primary training dataset.

- Millions of drive-days.
- Daily SMART attributes.
- Drive model, serial number, capacity.
- Failure indicators.

Used for:

- model training;
- feature engineering;
- threshold tuning;
- failure-trajectory learning.

### SMART-Z

Cross-vendor validation dataset.

- 147,496 disks.
- 65 standardized SMART attributes.
- Multiple vendors.

Used for:

- generalization testing;
- cross-vendor validation;
- schema harmonization evaluation.

### Synthetic / Chaos Dataset

Generated dataset for stress-testing.

Used for:

- guardrail testing;
- chaos injection;
- missing telemetry scenarios;
- multi-drive failure cascades;
- compensating-action validation.

---

## 10. Data Pipeline

The data pipeline follows a layered architecture.

```text
raw/
  Immutable source files

bronze/
  Parsed source data with minimal transformation

silver/
  Canonical schema, harmonized SMART attributes,
  telemetry-gap flags, survivorship filtering

gold/
  ML-ready features, labels, splits,
  optional LSTM sequence tensors

audit/
  Dataset versions, feature registry,
  data quality reports, decision records
```

All large datasets are stored as:

```text
Parquet + ZSTD compression
```

---

## 11. Feature Engineering Strategy

The project uses trajectory-based features instead of raw daily SMART values.

### Feature Families

| Feature Family | Examples |
|---|---|
| Time-window aggregates | 7/14/30-day mean, median, min, max, std |
| Derivatives | 7-day delta, 30-day delta, slope, acceleration |
| Event counts | Positive-day counts, spike counts, zero-to-non-zero flags |
| Missingness / staleness | Days since last telemetry, gap counts, stale flags |
| Cross-vendor harmonization | Normalized SMART values, model-family z-scores |
| Lifecycle context | Drive age, age², capacity, power-on hours |
| Sequence tensors | Last 30 days × top SMART attributes for LSTM |

### Example Features

```text
reallocated_sector_count_30d_slope
current_pending_sector_count_7d_delta
offline_uncorrectable_zero_to_nonzero_7d
command_timeout_event_count_14d
drive_age_days
telemetry_coverage_30d
```

---

## 12. Getting Started

### Prerequisites

- Python 3.11+
- Make
- Redis, optional for live agent state
- Git LFS or external storage if datasets are large

### Clone the Repository

```bash
git clone https://github.com/your-username/disk-failure-self-healing-agent.git
cd disk-failure-self-healing-agent
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

Or using Poetry:

```bash
poetry install
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure Before Running

Everything the code reads lives in `configs/*.yaml` — there are no
hardcoded fallbacks for the settings below, so review these before your
first real run. Full reference: `docs/developer_guide.md` §4. Nothing
here needs to change to run `make smoke`, `make agent-demo`, or the
synthetic-data walkthrough (§14) — only real-data / production settings.

| File | Setting | When you must set it |
|---|---|---|
| `configs/data.yaml` | `download.backblaze.quarters` (or `start_quarter`/`end_quarter`) | Before `make download-backblaze` — empty by default, nothing downloads until you configure a time period |
| `configs/data.yaml` | `download.smartz.url` | Before `make download-smartz` — SMART-Z has no public bulk-download API; request access first |
| `configs/model.yaml` | `splits.train_end` / `validation_end` / `test_end` | Before `make train` against real data whose date range doesn't overlap 2021–2023 (the defaults) — an empty split raises a clear error naming which one |
| `configs/model.yaml` | `model.type` (`lightgbm` \| `xgboost`) | Only if you want XGBoost instead of the LightGBM default |
| `configs/model.yaml` | `mlflow.tracking_uri` / `experiment_name` | Only if you want runs logged somewhere other than the local `sqlite:///mlflow/mlflow.db` default |
| `configs/model.yaml` | `hyperparameter_search.enabled` / `smote_comparison.enabled` | Optional — both default to `false` so `make train` stays fast and deterministic |
| `configs/agent.yaml` | `action_thresholds`, `human_review.*_sla_hours` | Only to change the default risk-tier cutoffs / SLA clock before a real deployment |
| `configs/guardrails.yaml` | `operational_state.backend` (`in_memory` \| `redis`) + `redis_url` | Only for a multi-instance deployment where drain/rate-limit counters must be shared — a single process is correct with the `in_memory` default |
| `configs/features.yaml` | `sequences.*` | Only if using the optional LSTM branch (`make build-sequences` / `train-lstm`) |

Every `pipelines/*.py` script and CLI flag that reads one of these prints
a clear, actionable error if something required is missing — none of
them silently fall back to guessed values.

### Run Tests

```bash
make lint
make test
```

Or manually:

```bash
ruff check .
pytest
```

---

## 13. Common Make Targets

The project uses a `Makefile` for reproducible pipeline execution.

| Command | Purpose |
|---|---|
| `make install` | Install runtime + dev dependencies (`uv sync --extra dev`) |
| `make lint` | Run ruff + mypy |
| `make format` | Auto-format and fix lint issues |
| `make test` | Run the full test suite (unit + integration + chaos + golden + property + smoke) |
| `make test-unit` / `test-integration` / `test-chaos` / `test-golden` / `test-property` / `smoke` | Run one test layer only |
| `make coverage` | Run the suite under pytest-cov; writes `htmlcov/index.html` |
| `make ci` | `lint` + `test` — what CI runs |
| `make clean` | Remove caches (never data or runtime state) |
| `make clean-data` | Remove regenerated pipeline outputs, preserving `.gitkeep` placeholders (never raw source data) |
| `make download-backblaze` / `download-smartz` | Configurable raw-data download — see §14 |
| `make ingest-backblaze` | Ingest Backblaze raw data into Bronze |
| `make ingest-smartz` | Ingest SMART-Z raw data into Bronze |
| `make ingest-synthetic-stub` | Land the synthetic placeholder dataset into Bronze |
| `make build-silver` | Build canonical Silver telemetry |
| `make build-features` | Build Gold trajectory features |
| `make build-labels` | Build failure labels and splits |
| `make train` | Train model, tune threshold, compute SHAP importance, log to MLflow |
| `make build-sequences` / `train-lstm` | Optional LSTM comparison branch (`train-lstm` needs `uv sync --extra torch`) |
| `make score-fleet` | Batch-score the current fleet with the latest trained model |
| `make plots` | Render performance-metric plots (calibration, SHAP, class imbalance) to `data/audit/plots/` |
| `make final-report` | Aggregate chaos/latency/model reports into a final evaluation report |
| `make agent-demo` | Run a minimal LangGraph agent demo |
| `make dashboard` | Launch Streamlit dashboard |
| `make api` | Launch FastAPI approval service |

Full details on every target: `docs/developer_guide.md` §3.1. For a
step-by-step walkthrough of running all of this end-to-end — with
synthetic data (fast, offline) or real Backblaze/SMART-Z data — see
**§14** below, or `docs/developer_guide.md` §12.1.

Example:

```bash
make ingest-backblaze
make build-silver
make build-features
make build-labels
make train
```

---

## 14. End-to-End Testing: Synthetic Data vs. Real Data

Full details and verified command output: `docs/developer_guide.md` §12.1.
Two ways to exercise the whole system, from raw data to a decision.

### A. Synthetic data (fast, offline, no download)

Agent/API loop only, no data pipeline needed:

```bash
make smoke        # one real MAPE-K cycle + every documented API endpoint, in-memory
make agent-demo   # one real MAPE-K cycle against a hardcoded 2-drive fleet, printed live
```

Full data pipeline, bronze through a real trained model — this genuinely
completes end-to-end:

```bash
make ingest-synthetic-stub   # 15 drives, 2021-01-01..2023-06-01 -> data/bronze/synthetic/
make build-silver
make build-features
make build-labels
make train           # trains, tunes threshold, logs to MLflow, writes a model card
make score-fleet     # batch-scores the current fleet -> data/audit/predictions/
make plots           # renders calibration/SHAP/imbalance plots -> data/audit/plots/
```

The synthetic stub generates real degrading failure trajectories spread
across the default chronological split boundaries, so every split has
both real failures and healthy drives to learn from. Expect a
suspiciously perfect AUPRC (~1.0) — the trajectories are noise-free by
design, so this confirms the pipeline plumbing works, not that the model
is good. Clean up with `make clean-data` between runs.

### B. Real data (Backblaze, optionally SMART-Z)

```bash
# configs/data.yaml -> download.backblaze.quarters: ["Q1_2025"]  (start with ONE)
make download-backblaze
make ingest-backblaze
make build-silver
make build-features
make build-labels
make train          # trains, tunes threshold, logs to MLflow, writes a model card
make score-fleet    # batch-scores the current fleet -> data/audit/predictions/
make plots          # renders calibration/SHAP/imbalance plots -> data/audit/plots/
```

**Disk space warning:** one Backblaze quarterly archive is ~1-1.5GB
compressed and expands to several GB of CSV. Check `df -h` before
configuring more than one or two quarters.

`make score-fleet`'s output is a standalone batch report — it is *not*
wired into `make agent-demo`/`make api`, which always run against the
hardcoded demo fleet (see `docs/developer_guide.md` §5.8 for why).

---

## 15. ML Workflow

The ML workflow is:

```text
Backblaze data
    ↓
Bronze ingestion
    ↓
Silver harmonization
    ↓
Gold feature engineering
    ↓
Labeling and temporal splitting
    ↓
Model training with XGBoost / LightGBM
    ↓
Precision-first threshold tuning
    ↓
MLflow model registry
    ↓
Agent Analyze node
```

### Primary Evaluation Metric

```text
AUPRC — Area Under the Precision-Recall Curve
```

### Target Operating Point

```text
Precision ≥ 95%
Recall ≥ 35–50%
```

---

## 16. Autonomous Agent Workflow

The agent uses a MAPE-K loop:

```text
Monitor
    ↓
Analyze
    ↓
Plan
    ↓
Guardrail Engine
    ↓
Execute or Human Review
    ↓
Validate
    ↓
Checkpoint + Audit
    ↓
Return to Monitor
```

### Action Tiers

| Tier | Trigger | Operational Effect |
|---|---|---|
| Monitor | Low or uncertain risk | No action |
| Warn | Medium risk | Dashboard alert |
| Cordon | High risk | Stop new workload placement |
| Migrate | High risk + guardrails pass | Proactively move data/workload |
| Drain | Very high risk + guardrails pass | Remove drive/node from service |
| Human Review | Blocked or uncertain critical action | Requires approval/rejection |

---

## 17. Guardrails

Guardrails are mandatory policy checks between prediction and execution.

### Example Hard Guardrails

- Never drain the last healthy node in a failure domain.
- Maintain replication quorum.
- Enforce maximum concurrent drains.
- Block destructive actions when telemetry is stale.
- Block destructive actions when feature confidence is low.

### Example Soft Guardrails

- Avoid migration during high I/O periods.
- Respect maintenance windows.
- Enforce action rate limits.

### Conflict Resolution

```text
Safety > Operational > Efficiency
```

If rules conflict, the most restrictive rule wins.

---

## 18. Reliability Checker and Trust Score

The Reliability Checker audits every autonomous decision.

### Reliability Checks

| Check | Meaning |
|---|---|
| Correctness | Was the prediction ultimately true? |
| Necessity | Was the action actually needed? |
| Safety | Did the action avoid data loss and quorum violations? |
| Timeliness | Was the action completed early enough? |
| Guardrail compliance | Were hard and soft rules respected? |

### Trust Score Formula

```text
BaseScore =
    0.40 × Correctness
  + 0.30 × Timeliness
  + 0.30 × Necessity

TrustScore =
    BaseScore
  × SafetyMultiplier
  × GuardrailMultiplier
```

Veto semantics:

```text
Safety violation            → TrustScore = 0
Hard guardrail violation    → TrustScore = 0
Soft guardrail violation    → TrustScore × 0.5
```

---

## 19. Dashboards

The project includes Streamlit dashboards for:

- fleet health;
- at-risk drives;
- predicted failures by horizon;
- trust-score trends;
- guardrail compliance;
- human approval queue;
- audit trail;
- compensating actions;
- model drift indicators.

Launch dashboard:

```bash
make dashboard
```

Or manually:

```bash
streamlit run src/dashboards/app.py
```

---

## 20. FastAPI Approval Service

The FastAPI service exposes endpoints for human oversight and audit.

Example endpoints:

```text
POST /api/v1/agent/run-cycle
GET  /api/v1/fleet/state
GET  /api/v1/predictions/latest
GET  /api/v1/actions/pending
POST /api/v1/actions/{action_id}/approve
POST /api/v1/actions/{action_id}/reject
GET  /api/v1/audit/decisions
GET  /api/v1/reliability/trust-trend
GET  /api/v1/guardrails/violations
```

`POST /api/v1/agent/run-cycle` runs one MAPE-K decision cycle; approve/reject
genuinely resume the paused LangGraph thread, not just update a record. No
authentication exists on any endpoint yet — see `docs/developer_guide.md`
§13 "Known gaps" before exposing this beyond local/trusted use.

Launch API:

```bash
make api
```

Or manually:

```bash
uvicorn src.api.main:app --reload
```

---

## 21. Testing Strategy

The project uses a layered testing strategy.

| Test Type | Purpose |
|---|---|
| Unit tests | Feature logic, labels, guardrails, trust score |
| Integration tests | LangGraph nodes, simulator API, FastAPI queue |
| Golden-dataset tests | Regression testing for feature engineering |
| Replay tests | Checkpoint recovery and duplicate-action prevention |
| Chaos tests | Missing telemetry, multi-drive failures, action crashes |
| Cross-dataset tests | SMART-Z generalization |
| Human-review tests | Approval, rejection, timeout paths |

Run tests:

```bash
make test
```

---

## 22. Milestones

| Milestone | Deliverable |
|---|---|
| M0 Bootstrap | Repository, configs, data contracts, golden dataset |
| M1 Data & Features | Ingestion, harmonization, feature pipeline |
| M2 Models + MLOps | Trained models, MLflow, threshold policy |
| M3 LangGraph Core | Cyclic MAPE-K graph, checkpointing, recovery |
| M4 Guardrails | Rule catalog, guardrail engine, test harness |
| M5 Simulation | Fleet simulator, compensating actions, chaos |
| M6 Validation | Reliability checker, trust score, human-in-loop |
| M7 Dashboard | Streamlit dashboards, approval queue |
| M8 Evaluation | Chaos tests, cross-dataset evaluation, final report |

---

## 23. Target Metrics

| Metric | Target |
|---|---:|
| Prediction precision | ≥ 95% |
| Prediction recall | 35–50% |
| Loop cycle time | < 5 minutes |
| Guardrail evaluation latency | < 500 ms |
| Hard-guardrail compliance | 100% |
| Soft-guardrail compliance | ≥ 95% |
| False-positive action rate | < 5% |
| False-negative action rate | < 2% |
| Data-loss events in simulation | 0 |
| Quorum violations in simulation | 0 |
| Duplicate actions after crash recovery | 0 |

---

## 24. Safety and Limitations

This project is a MVP and is simulation-first.

It does **not** currently include:

- direct control of physical production hardware;
- full distributed storage engine implementation;
- multi-tenant production deployment;
- real-time kernel-level telemetry collection;
- fully autonomous production operation without supervision.

The system is evaluated in a simulated operational environment with guardrails, human oversight, and auditability.

---

## 25. References

1. Google Cloud Blog: Seagate & Google predict HDD failures with ML
2. Wei et al., SMART-Z dataset, Scientific Data 2025
3. Backblaze Hard Drive Test Data
4. Lu et al., Making Disk Failure Predictions SMARTer!, USENIX FAST 2020
5. LangGraph Documentation
6. Kephart & Chess, The Vision of Autonomic Computing, 2003
7. Chen & Guestrin, XGBoost, KDD 2016
8. Chawla et al., SMOTE, 2002

---

## 26. Final Statement

This project is not only a disk-failure prediction system. It is a **validated autonomy framework**.

The prediction model identifies risk.
The agent plans and executes remediation.
The guardrails prevent unsafe behavior.
The human-review layer handles uncertainty.
The Reliability Checker audits every decision.
The dashboards and audit logs make autonomy inspectable.

The core question this project answers is:

> **Can an autonomous self-healing system be trusted to act, and can we prove that its decisions are correct, safe, necessary, timely, and explainable?**
