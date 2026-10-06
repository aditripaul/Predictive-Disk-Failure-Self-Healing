# Phase-Wise Project Plan

## AI-Based Autonomous System Validation & Reliability Checker
### Predictive Disk-Failure Self-Healing Agent for Server Fleets

> **Status note (2026-10-06).** This document records the original design
> and targets and is kept as written. What was built and measured is in
> `docs/system_summary.md` and `docs/model_status_and_runbook.md`. In
> particular: the prediction target stated here (95% precision at 35-50%
> recall) is not met on real data (41% precision at 8.5% recall on the real
> fleet, 30-day horizon), the primary horizon is now 30 days, and only
> Backblaze data has been evaluated.

**Document role:**
This document defines the complete execution roadmap for the Project. It translates the design goals from `design_goal.md` and the dataset/feature strategy from `dataset_strategy.md` into a phased, actionable engineering plan.

Each phase includes:

- objective;
- key tasks;
- recommended tools;
- RAM-aware implementation guidance;
- deliverables;
- exit criteria.

---

# 1. Purpose and Scope

This project builds and validates a two-layer autonomous system:

| Layer | Role |
|---|---|
| **Layer 1 — Self-Healing Agent** | Predicts impending disk failure from SMART telemetry and autonomously executes remediation actions such as monitor, warn, cordon, migrate, or drain inside a guardrailed LangGraph MAPE-K loop |
| **Layer 2 — Reliability Checker** | Audits every autonomous decision for correctness, safety, necessity, timeliness, and guardrail compliance; issues a trust score with veto semantics and produces explainability logs |

The project is simulation-first. The agent operates against a simulated fleet, and all autonomous behavior is validated before any consideration of real-world deployment.

---

# 2. Project Objectives

| # | Objective | Success Criterion |
|---:|---|---|
| O1 | Ingest and harmonize SMART telemetry from Backblaze and SMART-Z | Canonical schema; labeled healthy/failing trajectories |
| O2 | Predict disk failure within N days | Precision ≥ 95%; Recall ≥ 35–50% |
| O3 | Implement guardrailed agentic decision loop | Zero hard-guardrail violations in simulation |
| O4 | Execute autonomous remediation in fleet simulator | Actions complete without data loss; compensating actions on failure |
| O5 | Implement reliability checker and trust scoring | Trust score discriminates trustworthy vs. untrustworthy decisions |
| O6 | Produce audit trail and explainability | Per-decision rationale, guardrail status, checkpoint history |

---

# 3. Planning Principles

The execution plan follows these principles:

1. **Safety over availability**
   - False positives are preferable to unsafe false negatives.

2. **Precision-first autonomy**
   - Autonomous actions must not be triggered by noisy or low-confidence predictions.

3. **Explainability by default**
   - Every prediction, action, block, override, and trust score must be explainable.

4. **Idempotent and retryable actions**
   - Every action must have a unique action ID.
   - Retries must not duplicate migrations, drains, or compensations.

5. **Crash-safe operation**
   - LangGraph checkpointing must allow safe resume after failure.

6. **RAM-aware data engineering**
   - Use Polars, DuckDB, PyArrow, and Parquet.
   - Avoid pandas as the primary processing engine.
   - Avoid loading full Backblaze datasets into memory.

7. **Simulation-first validation**
   - All autonomous behavior is tested against a simulated fleet.

8. **Test-driven and replayable**
   - Golden datasets, unit tests, integration tests, chaos tests, and replay tests are required.

9. **Incremental delivery**
   - Start with a minimal vertical slice before building the full pipeline.

---

# 4. RAM-Constrained Engineering Rules

All data-intensive work must follow these rules.

| Principle | Implementation |
|---|---|
| Never load full CSVs into memory | Use `polars.scan_csv()` with projection and predicate pushdown, or DuckDB `read_csv_auto()` |
| Out-of-core SQL | DuckDB operates directly on Parquet files on disk |
| Columnar storage | Store intermediate and final data as Parquet + ZSTD |
| Partitioned processing | Partition by `source_dataset/year/month` or `drive_id_hash` |
| Reduced precision | Use `float32` instead of `float64`; use categorical encoding for strings |
| Incremental feature computation | Maintain rolling-state tables; avoid recomputing full history |
| Lazy execution | Use `polars.LazyFrame` and `.sink_parquet()` for streaming writes |
| Sequence data on disk | Store LSTM tensors as memory-mapped `.zarr` or `.npy`, not in-memory arrays |
| Subsampled tuning | Hyperparameter search on stratified samples; full evaluation on validation set |
| Lean LangGraph state | Agent state carries IDs and metadata only; bulk data lives in DuckDB/Redis |

---

# 5. Technology Stack

| Layer | Tool | Purpose | Notes |
|---|---|---|---|
| DataFrame engine | Polars | ETL, feature engineering, profiling | Lazy execution; RAM-efficient |
| SQL engine | DuckDB | Window functions, joins, ad-hoc queries on Parquet | Out-of-core |
| Columnar I/O | PyArrow | Parquet read/write; Arrow interchange | Efficient columnar memory layout |
| Storage format | Parquet + ZSTD | Bronze/silver/gold data zones | Compressed; column-pruned |
| Data validation | Pandera with Polars backend, or native Polars schema checks + pytest | Schema enforcement, null checks, range checks | Lightweight |
| Configuration | Pydantic Settings or Hydra | Typed configs for every pipeline stage | Reproducible runs |
| Batch pipeline runner | Makefile / Just; optional Dagster for scheduled runs | ETL orchestration | LangGraph is not used for batch ETL |
| ML training | LightGBM / XGBoost | Primary failure prediction models | Histogram-based; memory-efficient |
| Experiment tracking | MLflow | Model registry, metrics, artifacts | Local file store for MVP |
| Hyperparameter tuning | Optuna | Threshold and model tuning | Subsampled search |
| Deep learning branch | PyTorch + zarr / numpy memmap | Optional LSTM sequence model | Memory-mapped tensors |
| Agent orchestration | LangGraph | Sole runtime orchestrator for MAPE-K loop | Lean state; SqliteSaver checkpoints |
| Fleet-state cache | Redis | Live fleet state; guardrail counters; cooldowns; active-action tracking | Low-latency operational checks |
| API layer | FastAPI | Human approval queue; audit APIs; agent endpoints | Async; lightweight |
| Dashboard | Streamlit | Fleet health, trust trend, approval UI, audit trail | Fast prototyping |
| Testing | pytest, Hypothesis, optional Locust | Unit, property-based, load, chaos tests | Golden-dataset regression tests |
| Logging / audit | structlog / JSON logs | Per-decision rationale, guardrail status, explainability | Query-friendly; append-only |
| Optional observability | OpenTelemetry | Distributed tracing of LangGraph nodes | Add only if needed |

---

# 6. Data Architecture and Storage Zones

```text
data/
├── raw/                        # Immutable source archives
│   ├── backblaze/
│   ├── smartz/
│   └── synthetic/
│
├── bronze/                     # Parsed source data; source schema preserved
│   ├── backblaze/year=2024/month=01/
│   └── smartz/
│
├── silver/                     # Canonical schema; harmonized; gap-aware
│   ├── canonical_telemetry/
│   ├── drive_metadata/
│   └── telemetry_gaps/
│
├── gold/                       # ML-ready features, labels, splits, sequences
│   ├── features/date=2024-01-01/
│   ├── labels/
│   ├── splits/
│   └── sequences/
│
├── synthetic/                  # Generated chaos / guardrail test data
│
└── audit/                      # Decision records; trust scores; explainability logs
    ├── dataset_versions/
    ├── feature_registry/
    └── data_quality_reports/
```

---

# 7. Phase Summary

| Phase | Name | Milestone | Duration | Primary Outcome |
|---:|---|---|---:|---|
| 0 | Project Bootstrap & Architecture | M0 | 1 week | Repo, contracts, environment, storage layout, golden dataset |
| 1 | Data Ingestion & Raw Landing | M1 | 1–2 weeks | Bronze-layer Parquet from all three sources |
| 2 | Harmonization & Canonical Schema | M1 | 1–2 weeks | Unified silver-layer telemetry |
| 3 | Feature Engineering Pipeline | M1 → M2 | 2 weeks | Gold trajectory-based features |
| 4 | Labeling, Splitting & Imbalance | M2 | 1 week | Leakage-free train/validation/test datasets |
| 5 | Model Training & Threshold Tuning | M2 | 2 weeks | High-precision failure model + MLflow registry |
| 6 | LangGraph MAPE-K Agent Core | M3 | 2 weeks | Checkpointed cyclic loop; crash recovery |
| 7 | Guardrail Engine | M4 | 1–2 weeks | Hard/soft policy enforcement; < 500 ms |
| 8 | Fleet Simulator & Chaos Injection | M5 | 2 weeks | Simulated fleet; compensating actions |
| 9 | Reliability Checker & Trust Scoring | M6 | 2 weeks | Meta-validator; trust score; audit records |
| 10 | Dashboards & Human-in-the-Loop UI | M7 | 1–2 weeks | Streamlit + FastAPI approval system |
| 11 | End-to-End Evaluation & Final Report | M8 | 2 weeks | Chaos tests; cross-dataset eval; final report |

**Total estimated duration:** 13–15 weeks, adjustable based on team size.

---

# 8. Phase Dependency and Delivery Flow

```text
Phase 0
Bootstrap
   │
   ▼
Phase 1
Ingestion
   │
   ▼
Phase 2
Harmonization
   │
   ▼
Phase 3
Feature Engineering
   │
   ▼
Phase 4
Labels & Splits
   │
   ▼
Phase 5
Model Training
   │
   ├──────────────────────────────┐
   ▼                              ▼
Phase 6                        Phase 7
LangGraph Agent Core           Guardrail Engine
   │                              │
   └──────────────┬───────────────┘
                  │
                  ▼
              Phase 8
        Fleet Simulator & Chaos
                  │
                  ▼
              Phase 9
      Reliability Checker & Trust
                  │
                  ▼
              Phase 10
       Dashboards & HITL UI
                  │
                  ▼
              Phase 11
     End-to-End Evaluation & Report
```

Some phases can partially overlap:

- Guardrail Engine can begin while LangGraph core is being stabilized.
- Fleet Simulator can begin after guardrail rules are defined.
- Dashboards can begin once audit records and approval queue contracts exist.
- Evaluation should be continuous, but formal end-to-end evaluation occurs in Phase 11.

---

# 9. Detailed Phase Plans

---

## Phase 0 — Project Bootstrap & Architecture

**Milestone:** M0
**Duration:** Week 1

### Objective

Establish a reproducible, typed, testable project foundation with clear data zones, configuration management, and CI.

### Key Tasks

1. **Create repository scaffold**

```text
project-root/
├── configs/
├── data_contracts/
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
├── pipelines/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── chaos/
├── notebooks/
├── mlflow/
├── scripts/
├── Makefile
└── pyproject.toml
```

2. **Environment and dependency pinning**
   - Use `uv`, `poetry`, or `pip-tools`.
   - Pin versions for:
     - `polars`
     - `duckdb`
     - `pyarrow`
     - `langgraph`
     - `lightgbm`
     - `xgboost`
     - `mlflow`
     - `fastapi`
     - `redis`
     - `streamlit`
     - `pandera`
     - `optuna`

3. **Define data contracts**
   - `CanonicalTelemetryRecord`
   - `DriveMetadata`
   - `FeatureRecord`
   - `FeatureConfidence`
   - `PredictionOutput`
   - `ActionProposal`
   - `GuardrailResult`
   - `DecisionAuditRecord`
   - `TrustScoreRecord`

4. **Set up CI**
   - Linting with `ruff`.
   - Type checking with `mypy` or `pyright`.
   - Unit tests with `pytest`.
   - Golden-dataset smoke test.

5. **Create golden dataset**
   - Extract 100 healthy and 100 failing drive trajectories from Backblaze.
   - Include missing-telemetry and stale-telemetry examples.
   - Save as a small Parquet file under `tests/golden/`.

6. **Define Makefile targets**

```text
make install
make lint
make test
make data-check
make features
make train
make agent-demo
```

### RAM Practices

- No large-scale data processing in this phase.
- Validate that Polars and DuckDB can read a sample CSV without memory issues.

### Deliverables

- Repository scaffold.
- Passing CI pipeline.
- Typed data contracts.
- Golden dataset Parquet file.
- Makefile with basic targets.

### Exit Criteria

- [ ] `make install && make test` passes.
- [ ] Golden dataset loads in Polars without error.
- [ ] Data contracts reviewed.
- [ ] CI is green.
- [ ] Storage zones and configuration strategy documented.

---

## Phase 1 — Data Ingestion & Raw Landing

**Milestone:** M1 start
**Duration:** Weeks 1–2

### Objective

Ingest Backblaze, SMART-Z, and synthetic source data into a Bronze Parquet layer without loading full files into memory.

### Key Tasks

1. **Backblaze ingestion**
   - Download quarterly archives to `data/raw/backblaze/`.
   - Process one file at a time using Polars or DuckDB.
   - Add metadata columns:
     - `source_dataset`
     - `source_file`
     - `ingested_at`
     - `schema_version`
   - Sink to partitioned Parquet:

```text
data/bronze/backblaze/year=YYYY/month=MM/
```

2. **SMART-Z ingestion**
   - Access SMART-Z dataset.
   - Store raw files in `data/raw/smartz/`.
   - Build initial source-to-canonical column mapping table.
   - Sink to:

```text
data/bronze/smartz/
```

3. **Synthetic data stub**
   - Define synthetic generator schema.
   - Create a small placeholder dataset for pipeline testing.
   - Full synthetic chaos generation occurs in Phase 8.

4. **Data profiling report**
   - Row counts per file.
   - Column presence/absence.
   - Missingness percentage per SMART attribute.
   - Drive model distribution.
   - Failure label distribution, expected to be below 1%.

### Example RAM-Safe Ingestion

```python
import polars as pl

REQUIRED_COLS = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
    "failure",
]

lf = pl.scan_csv(
    "data/raw/backblaze/2024_Q1.csv",
    infer_schema_length=10_000,
)

(
    lf.select(REQUIRED_COLS)
      .filter(pl.col("date").is_not_null())
      .sink_parquet(
          "data/bronze/backblaze/year=2024/month=03/part.parquet",
          compression="zstd",
      )
)
```

### RAM Practices

- Never call `.collect()` on the full scan.
- Process one quarterly file, then release.
- Use DuckDB for compressed CSVs if extraction is costly.
- Store outputs as Parquet with ZSTD compression.

### Deliverables

- Bronze Parquet datasets for Backblaze and SMART-Z.
- Source-to-canonical column mapping draft.
- Ingestion profiling report.
- Synthetic placeholder dataset.

### Exit Criteria

- [ ] All required Backblaze quarters parsed into Parquet.
- [ ] SMART-Z raw data accessible and profiled.
- [ ] No full-dataset in-memory load required.
- [ ] Profiling report shows expected row counts and failure rate below 1%.
- [ ] Raw data remains immutable in `data/raw/`.

---

## Phase 2 — Harmonization & Canonical Schema

**Milestone:** M1 complete
**Duration:** Weeks 2–3

### Objective

Create a unified Silver-layer schema across Backblaze and SMART-Z, handling vendor differences, missing data, telemetry gaps, and survivorship bias.

### Key Tasks

1. **Normalize identifiers**
   - Standardize `drive_id`.
   - Standardize `date`.
   - Parse capacity strings into numeric GB.
   - Derive `model_family`.
   - Infer `manufacturer` where possible.

2. **Harmonize SMART attributes**
   - Map vendor-specific SMART names to canonical names.
   - Preserve:
     - `smart_raw_value`
     - `smart_normalized_value`
   - Create direction-normalized:

```text
smart_badness_value
```

where higher always means worse health.

3. **Handle missing telemetry**
   - Forward-fill short gaps where appropriate.
   - Compute:

```text
days_since_last_telemetry
hours_since_last_telemetry
telemetry_gap_7d_count
telemetry_gap_30d_count
stale_telemetry_flag
telemetry_coverage_30d
```

4. **Apply survivorship and feature maturity rules**
   - Mark drives as:
     - `WARMUP`
     - `MATURE`
     - `STALE`
     - `DECOMMISSIONED`
   - Require at least 30 days of observed history before full 30-day feature generation.

5. **Apply transformations**
   - Use `log1p` for skewed SMART attributes.
   - Apply robust clipping or winsorization where necessary.

6. **Run data quality checks**
   - Duplicate `drive_id + date` detection.
   - Null date checks.
   - Invalid SMART value range checks.
   - Failure date consistency checks.
   - Capacity positivity checks.

### Example DuckDB Gap Detection

```sql
COPY (
  SELECT *,
    date - LAG(date) OVER (
      PARTITION BY drive_id
      ORDER BY date
    ) AS days_since_last_telemetry
  FROM read_parquet('data/bronze/backblaze/**/*.parquet')
)
TO 'data/silver/canonical_telemetry/backblaze/year=2024/month=01/part.parquet'
(FORMAT PARQUET, CODEC 'ZSTD');
```

### RAM Practices

- Partition silver data by `source_dataset/year/month`.
- Process one partition at a time.
- Use DuckDB window functions instead of loading entire drive histories.

### Deliverables

- Canonical silver Parquet dataset.
- SMART attribute mapping table.
- SMART badness orientation configuration.
- Telemetry-gap and feature-maturity logic.
- Data quality report.

### Exit Criteria

- [ ] Backblaze and SMART-Z queryable through one canonical schema.
- [ ] `days_since_last_telemetry` correctly flags gaps.
- [ ] Duplicate, null, and range checks pass.
- [ ] Forward-fill and stale-telemetry logic unit-tested.
- [ ] Feature maturity states are generated correctly.

---

## Phase 3 — Feature Engineering Pipeline

**Milestone:** M1 → M2
**Duration:** Weeks 3–5

### Objective

Transform daily SMART snapshots into trajectory-based features that capture degradation trends, volatility, rate of change, intermittent faults, telemetry confidence, and lifecycle context.

### Feature Groups

| Group | Features | Rationale |
|---|---|---|
| A. Time-window aggregates | 7/14/30-day rolling mean, median, min, max, std | Smooths daily noise; establishes baseline |
| B. Derivatives and rate of change | 7-day delta, 30-day delta, slope, acceleration | Captures how fast a drive is degrading |
| C. Threshold crossings and event counts | Positive-day counts, spike counts, zero-to-non-zero flags | Detects intermittent faults hidden by averages |
| D. Missingness and telemetry confidence | Gap counts, staleness, coverage, feature confidence | Treats silence as potentially informative |
| E. Cross-vendor harmonization | Normalized values, badness scores, ratios, model-family z-scores | Supports SMART-Z generalization |
| F. Metadata and lifecycle | Drive age, age², capacity, model family, power-on hours | Captures bathtub-curve behavior |
| G. Sequence vectors | Last 30 days × top attributes | Optional LSTM branch |

### Priority SMART Attributes

Start with the top 10–15 attributes:

```text
reallocated_sector_count
current_pending_sector_count
offline_uncorrectable
reported_uncorrectable_errors
raw_read_error_rate
seek_error_rate
spin_retry_count
command_timeout
power_on_hours
smart_overall_health_normalized
```

### Key Tasks

1. Implement all feature groups in Polars/DuckDB.
2. Build a YAML-driven feature registry.
3. Compute features in drive-level partitions.
4. Cast numeric features to `float32`.
5. Store gold features partitioned by date.
6. Generate sequence tensors as `.zarr` or memory-mapped `.npy` for the LSTM branch.
7. Compute feature confidence:

```text
feature_confidence =
    telemetry_coverage_30d
  × recency_factor
  × attribute_coverage_factor
```

8. Run golden-dataset regression tests.

### Feature Registry Example

```yaml
features:
  - name: current_pending_sector_count_30d_slope
    source_attribute: current_pending_sector_count
    transform: log1p
    window_days: 30
    operation: linear_slope
    partition_by: drive_id
    order_by: date
    dtype: float32
    version: 1
```

### RAM Practices

- Use `polars.LazyFrame` rolling expressions with `.sink_parquet()`.
- For slopes, use DuckDB window functions or small drive-level partitions.
- Avoid creating one massive wide DataFrame.
- Store incremental rolling state tables to avoid recomputing full history.
- Write features incrementally.

### Deliverables

- Gold feature Parquet tables.
- Feature registry YAML.
- Feature confidence logic.
- Sequence dataset for optional LSTM.
- Golden-dataset regression test suite.
- Feature engineering documentation.

### Exit Criteria

- [ ] Features are deterministic and reproducible.
- [ ] No future data leakage in rolling windows.
- [ ] Golden-dataset tests pass.
- [ ] Gold features load within RAM budget.
- [ ] Top features show visually distinct failure trajectories.
- [ ] Feature confidence and telemetry staleness are available.

---

## Phase 4 — Labeling, Splitting & Imbalance Strategy

**Milestone:** M2
**Duration:** Weeks 5–6

### Objective

Create reliable failure labels, leakage-free temporal splits, and a class-imbalance mitigation strategy.

### Key Tasks

1. **Label construction**
   - Create labels for horizons:
     - 7 days
     - 14 days
     - 30 days
   - Use:

```text
label = 1 if confirmed failure within horizon
label = 0 if confirmed healthy for full horizon
label = null if censored
```

2. **Event classification**
   - Distinguish:
     - confirmed failure;
     - failure followed by replacement;
     - preventive replacement;
     - removed without failure;
     - right-censored observations.

3. **Censoring rules**
   - Do not treat removed-without-failure drives as automatic negatives.
   - Censor rows where the future horizon is not observable.

4. **Temporal splitting**
   - Train: earliest period.
   - Validation: middle period.
   - Test: latest unseen period.
   - Optional drive-level holdout.
   - External SMART-Z validation set.

5. **Class imbalance handling**
   - Use class weighting:
     - XGBoost `scale_pos_weight`
     - LightGBM `is_unbalance`
   - Use SMOTE only on training data if it improves AUPRC.
   - Evaluate using AUPRC, not accuracy.
   - Tune threshold for precision-first operation.

### RAM Practices

- Store labels as a narrow table:

```text
drive_id
date
horizon_days
label
event_type
observable_until_horizon
days_to_event
censoring_flag
```

- Join labels to features lazily.
- Do not materialize full joined tables until training.

### Deliverables

- Label generation logic.
- Train/validation/test split definitions.
- Censoring report.
- Class imbalance report.
- Label leakage tests.

### Exit Criteria

- [ ] Labels are deterministic.
- [ ] No future information appears in features.
- [ ] Censored rows are correctly excluded or marked.
- [ ] Class distribution documented, with failures below 1%.
- [ ] Temporal split is reproducible from config.

---

## Phase 5 — Model Training & Threshold Tuning

**Milestone:** M2 complete
**Duration:** Weeks 6–8

### Objective

Train high-precision disk-failure prediction models and select a safe operating threshold and action policy.

### Model Candidates

| Model | Role |
|---|---|
| LightGBM | Primary baseline; fast and histogram-based |
| XGBoost | Strong alternative; external-memory support |
| Logistic Regression | Interpretable sanity baseline |
| LSTM | Optional sequence-based comparison |

### Key Tasks

1. **Train baseline model**
   - Use gold features.
   - Use chronological validation.
   - Track metrics with MLflow.

2. **Metrics**
   - AUPRC, primary.
   - Precision.
   - Recall.
   - False-positive rate.
   - False-negative rate.
   - Calibration.
   - Warning lead time.
   - Precision at top-K risky drives.

3. **Threshold tuning**
   - Use Optuna.
   - Subsample for search.
   - Retrain best configuration on full training partition.
   - Target:

```text
Precision ≥ 95%
Recall ≥ 35–50%
```

4. **Define action tiers**

| Tier | Trigger | Operational Effect |
|---|---|---|
| Monitor | Low or uncertain risk | No action; continue observation |
| Warn | Medium risk | Dashboard alert only |
| Cordon | High risk | Stop new workload placement; non-destructive |
| Migrate | High risk + guardrails pass + sufficient feature confidence | Proactive migration |
| Drain | Very high risk + guardrails pass + high feature confidence | Remove drive/node from service |
| Human Review | Blocked, uncertain, or critical | Requires explicit approval/rejection |

5. **Add feature-confidence gating**
   - Destructive actions require:

```text
p_fail >= high_threshold
AND feature_confidence >= confidence_threshold
AND feature_maturity == MATURE
AND telemetry is not stale
AND guardrails pass
```

6. **Generate explainability artifacts**
   - Global feature importance.
   - Local explanations for high-risk predictions.
   - Store explanations for Reliability Checker consumption.

7. **Register model**
   - MLflow model registry.
   - Store:
     - threshold policy;
     - feature schema;
     - dataset version;
     - model card;
     - explainability artifacts.

### RAM Practices

- Use `float32` feature matrices.
- Use subsampled Optuna search.
- Prefer class weighting over repeated full-data SMOTE.
- Load Parquet columns selectively during training.

### Deliverables

- Trained model artifacts.
- MLflow experiment records.
- Threshold policy.
- Action-tier policy.
- Model evaluation report.
- Explainability artifacts.
- Model card.

### Exit Criteria

- [ ] Validation precision meets or exceeds 95% at chosen threshold.
- [ ] Recall is within acceptable 35–50% range.
- [ ] Model is reproducible from config.
- [ ] AUPRC is significantly better than random baseline.
- [ ] Explainability artifacts are generated.
- [ ] Action tiers and feature-confidence gating are documented.

---

## Phase 6 — LangGraph MAPE-K Agent Core

**Milestone:** M3
**Duration:** Weeks 8–10

### Objective

Build the autonomous self-healing agent as a checkpointed, cyclic LangGraph workflow.

### MAPE-K Nodes

| Node | Responsibility |
|---|---|
| Monitor | Pull latest fleet state from simulator/Redis |
| Analyze | Run registered ML model and rank risky drives |
| Plan | Propose remediation action and invoke guardrails |
| Execute | Execute approved action in fleet simulator |
| Validate | Verify outcome and emit decision/outcome record |

### Critical Architecture Rule: Lean State

Never put raw telemetry or large DataFrames in the LangGraph state.

Recommended state schema:

```python
class AgentState(TypedDict):
    run_id: str
    timestamp: str
    fleet_snapshot_id: str
    flagged_drive_ids: list[str]
    prediction_summary: dict
    proposed_action: dict
    guardrail_result: dict
    human_review_required: bool
    execution_result: dict
    validation_result: dict
    decision_record_id: str
```

Bulk telemetry and features remain in DuckDB, Parquet, or Redis.

### Key Tasks

1. Implement all five MAPE-K nodes.
2. Use `SqliteSaver` or equivalent checkpointing.
3. Verify crash recovery without duplicate actions.
4. Implement idempotent action IDs.
5. Implement cooldown and hysteresis:
   - Prevent repeated actions within cooldown window.
   - Require sustained high risk before escalation.
6. Implement human-in-the-loop interrupts:
   - Blocked critical actions go to FastAPI approval queue.
   - Approval/rejection resumes the LangGraph workflow.
7. Implement timeout degradation:
   - If no human response within critical SLA, default to safest non-destructive action.
   - Example: cordon or monitor, but do not drain.
8. Implement compensating actions:
   - If execution fails, trigger rollback or recovery operation.

### Deliverables

- LangGraph MAPE-K workflow.
- Lean AgentState schema.
- Checkpoint and crash-recovery tests.
- Human interrupt integration.
- Timeout degradation logic.
- Idempotency and cooldown logic.

### Exit Criteria

- [ ] Full Monitor → Analyze → Plan → Execute → Validate cycle completes.
- [ ] Crash recovery works without duplicate actions.
- [ ] Interrupts route blocked actions to FastAPI approval queue.
- [ ] Timeout degradation is tested.
- [ ] Loop cycle time is measurable and targets < 5 minutes.
- [ ] LangGraph checkpoint database remains small and lean.

---

## Phase 7 — Guardrail Engine

**Milestone:** M4
**Duration:** Weeks 9–11

### Objective

Implement policy checks that prevent unsafe autonomous actions, with sub-500 ms evaluation latency.

### Guardrail Catalog

| ID | Type | Rule | Enforcement |
|---|---|---|---|
| `PRED_THRESHOLD` | Prediction | Only act above high-confidence cutoff | Block low-confidence actions |
| `FEATURE_CONFIDENCE` | Prediction/data quality | Require sufficient feature confidence for destructive actions | Block or downgrade action |
| `TELEMETRY_FRESHNESS` | Hard pre-action | Block destructive action if telemetry is stale | Hard block/downgrade |
| `HARD_NO_LAST_NODE` | Hard pre-action | Never drain last healthy node in failure domain | Hard block |
| `HARD_QUORUM` | Hard pre-action | Maintain replication quorum after proposed drain | Hard block |
| `HARD_MAX_DRAINS` | Hard pre-action | Limit concurrent drain operations | Hard block |
| `SOFT_HIGH_IO` | Soft pre-action | Avoid migration during high I/O window | Warn or delay |
| `OPS_MAINTENANCE` | Operational | Respect maintenance windows | Delay or queue |
| `OPS_RATE_LIMIT` | Operational | Enforce global action rate limits | Throttle |
| `POST_DATA_INTEGRITY` | Post-action | Verify data integrity after action | Compensating action if failed |
| `POST_SERVICE_CONTINUITY` | Post-action | Verify service continuity | Compensating action if failed |

### Key Tasks

1. Define rule catalog in YAML and Pydantic models.
2. Implement guardrail evaluation engine.
3. Integrate Redis-backed operational state:
   - active drain counter;
   - cooldown tracker;
   - recent action log;
   - pending review state.
4. Implement conflict resolution:

```text
Safety > Operational > Efficiency
```

If rules conflict, the most restrictive rule wins.

5. Benchmark evaluation latency.
   - Target: < 500 ms.

6. Test every rule:
   - individually;
   - in combination;
   - under chaos conditions.

### RAM Practices

- Guardrail rules should evaluate compact operational state.
- Avoid querying full telemetry during guardrail checks.
- Use Redis for hot counters and active state.
- Use DuckDB only for audit or non-latency-critical queries.

### Deliverables

- Guardrail rule catalog.
- Guardrail engine module.
- Redis state integration.
- Latency benchmark report.
- Guardrail test suite.

### Exit Criteria

- [ ] All hard guardrails block unsafe actions in tests.
- [ ] Soft guardrails produce warnings or delays as expected.
- [ ] Stale telemetry blocks destructive actions.
- [ ] Evaluation latency is below 500 ms.
- [ ] Every guardrail decision is logged.
- [ ] Conflict resolution is tested.

---

## Phase 8 — Fleet Simulator & Chaos Injection

**Milestone:** M5
**Duration:** Weeks 10–12

### Objective

Create a simulated fleet environment for safe agent testing, including realistic failure cascades and chaos scenarios.

### Key Tasks

1. **Fleet state simulation**
   - Nodes.
   - Drives.
   - Replication groups.
   - Quorum state.
   - Drive health states:

```text
HEALTHY
DEGRADED
PREDICTED_FAILURE
FAILING
FAILED
CORDONED
MIGRATED
DRAINED
REPLACED
RECOVERED
```

2. **Telemetry generation**
   - Generate SMART trajectories.
   - Use trajectory morphing:
     - extract real Backblaze failure trajectories;
     - add noise;
     - compress or stretch timelines;
     - replicate across multiple drives;
     - inject correlated multi-drive failures.

3. **Action simulation**
   - Cordon.
   - Migrate.
   - Drain.
   - Replace.
   - Restore.
   - Compensate.

4. **Chaos scenarios**

| Scenario | Purpose |
|---|---|
| Missing/stale telemetry | Test Monitor resilience and telemetry-freshness guardrails |
| Sudden multi-drive failure | Test quorum guardrails |
| False-positive SMART spikes | Test precision threshold and action tiers |
| Clock skew/noise | Test feature robustness |
| Node crash during action | Test checkpoint recovery and compensating actions |
| Action timeout | Test idempotent retry |
| Partial migration failure | Test rollback and recovery |
| Human-review timeout | Test safe fallback behavior |

5. **Outcome validation**
   - No data loss.
   - Quorum maintained.
   - Compensating action triggered when required.
   - Action completion time recorded.

### Deliverables

- Fleet simulator.
- Simulator API for LangGraph Execute node.
- Chaos scenario library.
- Trajectory-morphed synthetic telemetry generator.
- Scenario test reports.

### Exit Criteria

- [ ] Agent can execute actions against simulator.
- [ ] Simulator detects data loss and quorum violations.
- [ ] All chaos scenarios are reproducible.
- [ ] Compensating actions are verified.
- [ ] Human-review timeout fallback is tested.
- [ ] Synthetic telemetry is based on realistic failure trajectories.

---

## Phase 9 — Reliability Checker & Trust Scoring

**Milestone:** M6
**Duration:** Weeks 11–13

### Objective

Build the meta-validator that audits every autonomous decision and produces a trust score with veto semantics.

### Reliability Checks

| Check | Pass Condition |
|---|---|
| Correctness | True positive or true negative |
| Necessity | No action on drives that would not have failed |
| Guardrail compliance | Zero hard violations; soft violations documented |
| Safety | No data loss; no quorum breach |
| Timeliness | Action completed sufficiently before actual failure |

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

Where:

```text
SafetyMultiplier =
    0 if safety violation
    1 otherwise

GuardrailMultiplier =
    0   if hard-guardrail violation
    0.5 if soft-guardrail violation
    1.0 if no violation
```

### Provisional and Final Trust Scores

| State | When Produced | Description |
|---|---|---|
| Provisional Trust Score | Immediately after action | Based on safety, guardrails, execution result, and necessity proxy |
| Final Trust Score | After label horizon completes | Includes correctness and timeliness |

### Key Tasks

1. Define `DecisionAuditRecord` schema.
2. Implement synchronous checks in Validate node:
   - guardrail compliance;
   - safety checks;
   - execution status.
3. Implement asynchronous evaluation pipeline:
   - join decision records with actual failure outcomes after N days;
   - update final trust score;
   - emit drift/retrain signal.
4. Generate explainability logs:
   - top contributing features;
   - guardrails evaluated;
   - action rationale;
   - human override reason, if applicable.
5. Implement drift detection:
   - feature drift;
   - prediction drift;
   - telemetry-gap drift;
   - false-positive/false-negative trend.
6. Store decision records in `data/audit/`.

### RAM Practices

- Compute SHAP explanations only for:
  - actions taken;
  - blocked actions;
  - sampled high-risk predictions;
  - sampled false positives/negatives.
- Do not compute SHAP for every drive-day.

### Deliverables

- Reliability checker module.
- Trust score implementation.
- Provisional/final trust scoring pipeline.
- Decision audit records.
- Explainability log generator.
- Drift/retrain signal logic.

### Exit Criteria

- [ ] Every autonomous decision has an audit record.
- [ ] Trust score correctly vetoes unsafe decisions.
- [ ] Hard-guardrail violations produce zero trust score.
- [ ] Async evaluation updates final trust scores after N days.
- [ ] Explainability logs are human-readable.
- [ ] Drift signal can trigger retraining workflow.

---

## Phase 10 — Dashboards & Human-in-the-Loop UI

**Milestone:** M7
**Duration:** Weeks 12–14

### Objective

Provide operational visibility, approval workflows, and full auditability.

### Dashboards

| Dashboard | Content |
|---|---|
| Fleet Health | Total/healthy/at-risk/failed/drained drives; predicted failures by horizon |
| Trust & Reliability | Average trust score; trend; vetoed decisions; guardrail compliance |
| Model Performance | Precision, recall, AUPRC, threshold, feature importance, drift indicators |
| Audit Trail | Decision history, guardrail outcomes, human overrides, compensating actions |
| Approval Queue | Pending actions, prediction confidence, blocked rule, approve/reject with reason code |

### FastAPI Endpoints

```text
GET  /api/v1/fleet/state
GET  /api/v1/predictions/latest
GET  /api/v1/actions/pending
POST /api/v1/actions/{action_id}/approve
POST /api/v1/actions/{action_id}/reject
GET  /api/v1/audit/decisions
GET  /api/v1/reliability/trust-trend
GET  /api/v1/guardrails/violations
```

### Key Tasks

1. Build FastAPI service.
2. Build Streamlit dashboards.
3. Read analytics from DuckDB.
4. Read live operational state from Redis.
5. Implement approval queue with mandatory override reason codes.
6. Implement audit export to CSV/Parquet.
7. Show provisional and final trust scores separately.

### Deliverables

- FastAPI service.
- Streamlit dashboards.
- Approval queue with HITL flow.
- Audit export functionality.
- Trust trend visualization.

### Exit Criteria

- [ ] Blocked actions appear in approval queue.
- [ ] Approve/reject resumes LangGraph workflow.
- [ ] Dashboards show trust score and guardrail compliance.
- [ ] Audit trail traces a decision end-to-end.
- [ ] Provisional and final trust scores are visible.
- [ ] Human override reason codes are mandatory.

---

## Phase 11 — End-to-End Evaluation & Final Report

**Milestone:** M8
**Duration:** Weeks 13–15

### Objective

Validate the full system against all success criteria and produce the final report.

### Evaluation Matrix

| Dimension | Metric | Target |
|---|---|---:|
| Prediction precision | Precision | ≥ 95% |
| Prediction recall | Recall | 35–50% |
| Loop performance | End-to-end cycle time | < 5 minutes |
| Guardrail latency | Evaluation time | < 500 ms |
| Safety | Hard-guardrail violations | 0 |
| Safety | Data loss events | 0 |
| Safety | Quorum violations | 0 |
| Reliability | False-positive action rate | < 5% |
| Reliability | False-negative action rate | < 2% |
| Trust scoring | Separation between trustworthy and untrustworthy decisions | Clear separation |
| Generalization | SMART-Z performance vs. Backblaze | Acceptable degradation |
| Guardrail compliance | Hard/soft | 100% hard / ≥ 95% soft |
| Crash recovery | Duplicate actions after recovery | 0 |

### Key Tasks

1. **Cross-dataset evaluation**
   - Train on Backblaze.
   - Validate on SMART-Z.
   - Report performance drop.
   - Analyze vendor-specific failure patterns.

2. **Chaos testing**
   - Run agent against all Phase 8 scenarios.
   - Verify zero hard-guardrail violations.
   - Verify compensating actions.
   - Verify crash recovery.

3. **Replay testing**
   - Use LangGraph checkpoint replay.
   - Ensure deterministic re-execution.
   - Validate no duplicate actions after crash.

4. **Guardrail effectiveness analysis**
   - Count blocked unsafe actions.
   - Count soft violations.
   - Measure delay introduced by operational guardrails.
   - Validate conflict-resolution logic.

5. **Reliability checker evaluation**
   - Compare trust score distribution for:
     - correct actions;
     - unnecessary actions;
     - blocked actions;
     - safety-violating actions.
   - Verify veto behavior.

6. **Feature-confidence evaluation**
   - Verify stale telemetry reduces confidence.
   - Verify destructive actions are blocked when confidence is low.
   - Verify immature drives do not trigger unsafe actions.

7. **Load testing**
   - Verify loop cycle < 5 minutes under realistic simulated fleet size.

8. **Final report**
   - Dataset summary.
   - Feature engineering impact.
   - Model performance.
   - Cross-dataset generalization.
   - Guardrail effectiveness.
   - Trust score analysis.
   - Limitations.
   - Future work.

### Deliverables

- Final evaluation report.
- Chaos test results.
- Cross-dataset analysis.
- Guardrail compliance report.
- Trust score analysis.
- Demo scripts.
- Runbook/deployment guide.

### Exit Criteria

- [ ] All primary success metrics are measured and reported.
- [ ] Zero hard safety violations in simulation.
- [ ] Trust score discriminates decision quality.
- [ ] System recovers from simulated crash.
- [ ] Feature-confidence gating works as designed.
- [ ] Final report is complete.

---

# 10. Milestone Mapping

| Milestone | Phases | Key Deliverable |
|---|---:|---|
| M0 Bootstrap | Phase 0 | Repo, contracts, CI, golden dataset |
| M1 Data & Features | Phases 1–3 | Ingestion, harmonization, feature pipeline |
| M2 Models + MLOps | Phases 4–5 | Trained models, MLflow, threshold policy |
| M3 LangGraph Core | Phase 6 | Cyclic MAPE-K graph, checkpointing, recovery |
| M4 Guardrails | Phase 7 | Rule catalog, engine, test harness |
| M5 Simulation | Phase 8 | Fleet simulator, compensating actions, chaos |
| M6 Validation | Phase 9 | Reliability checker, trust score, HITL |
| M7 Dashboard | Phase 10 | Streamlit + FastAPI dashboards, approval queue |
| M8 Evaluation | Phase 11 | Chaos tests, cross-dataset eval, final report |

---

# 11. Suggested 15-Week Timeline

| Week | Phase(s) | Focus |
|---:|---|---|
| 1 | 0 + 1 | Bootstrap; start Backblaze ingestion |
| 2 | 1 + 2 | Complete ingestion; begin harmonization |
| 3 | 2 | Canonical schema; SMART-Z mapping; telemetry gaps |
| 4 | 3 | Core feature engineering: aggregates, deltas, slopes |
| 5 | 3 → 4 | Sequence features; labels; splits |
| 6 | 4 → 5 | Imbalance strategy; baseline models |
| 7 | 5 | Threshold tuning; explainability; MLflow |
| 8 | 5 → 6 | Model finalization; LangGraph spike |
| 9 | 6 | MAPE-K core; checkpointing; HITL interrupts |
| 10 | 6 → 7 | Agent completion; guardrail engine |
| 11 | 7 → 8 | Guardrail tests; fleet simulator |
| 12 | 8 | Chaos scenarios; compensating actions |
| 13 | 9 | Reliability checker; trust score; async eval |
| 14 | 10 | Dashboards; approval queue |
| 15 | 11 | End-to-end evaluation; final report |

---

# 12. Cross-Cutting Workstreams

These workstreams run across multiple phases.

## 12.1 Data Quality and Golden Tests

- Maintain golden dataset.
- Run regression tests after every feature change.
- Validate schema, missingness, labels, and leakage rules.

## 12.2 MLOps and Experiment Tracking

- Track all model runs in MLflow.
- Store dataset version, feature registry version, threshold policy, and model card.
- Link model artifacts to evaluation reports.

## 12.3 Audit and Explainability

- Every decision must produce an audit record.
- Every blocked action must store guardrail context.
- Every human override must store reason code.
- Every trust score must link to decision evidence.

## 12.4 Testing Strategy

```text
Unit Tests
  feature logic, labels, guardrails, trust score

Integration Tests
  LangGraph nodes, simulator API, FastAPI queue

Replay Tests
  checkpoint recovery, time-travel, no duplicate actions

Chaos Tests
  missing telemetry, multi-drive failures, action crashes

Evaluation Tests
  precision, recall, trust-score separation, cross-dataset validation
```

---

# 13. Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---:|---|
| Backblaze data exceeds RAM | Pipeline crash | High | Polars lazy + DuckDB + partitioned Parquet |
| SMART-Z schema mismatch | Poor generalization | Medium | Early harmonization; canonical mapping; badness orientation |
| Severe class imbalance | Low precision | High | Class weighting; AUPRC; precision-first thresholding |
| Label leakage via forward-fill | Overoptimistic metrics | Medium | `days_since_last_telemetry`; strict temporal splits |
| Missing telemetry hides failure | Missed failures | Medium | Staleness flags; feature confidence; destructive-action gating |
| LangGraph state bloat | Slow checkpointing and recovery | High | Lean state; bulk data in DuckDB/Redis |
| False-positive autonomous actions | OpEx waste | Medium | Precision threshold; action tiers; guardrails |
| Unsafe drain action | Data loss/quorum breach | Low | Hard guardrails; simulator validation; compensating actions |
| Mid-cycle crash | Duplicate actions | Medium | Checkpointing; idempotent action IDs |
| Guardrail latency > 500 ms | Loop slows down | Low | Redis cache; optimized rule engine; benchmarking |
| Human approval timeout | Agent stalls | Medium | Timeout degradation to safest non-destructive action |
| Synthetic data unrealistic | Guardrails not truly tested | Medium | Trajectory morphing from real failures |
| Trust score cannot evaluate correctness immediately | Misleading metrics | High | Provisional/final trust scoring; async evaluation |
| Model drift | Degraded reliability | Medium | Drift detection; retrain signal |
| SMART-Z access delay | Cross-dataset validation blocked | Medium | Early request; Backblaze-only fallback |

---

# 14. Recommended First Implementation: Vertical Slice

Do not build the entire data pipeline before writing agent code.

Execute a minimal end-to-end slice early:

1. Download one month of Backblaze data for one drive model.
2. Use Polars to compute three features:
   - 7-day rolling mean;
   - 7-day delta;
   - drive age.
3. Train a dummy XGBoost model.
   - Accuracy is not important.
   - Structure and reproducibility are important.
4. Build a 3-node LangGraph:

```text
Monitor → Analyze → Plan
```

5. Add one hardcoded guardrail.
6. Print a provisional trust score.

This proves:

- Polars integration;
- LangGraph state management;
- checkpointing;
- guardrail hook;
- audit logging;

before committing to the full pipeline.

---

# 15. Global Definition of Done

| Deliverable | Done When… |
|---|---|
| Data pipeline | Backblaze + SMART-Z ingested; canonical schema validated; gold features reproducible; runs within RAM budget |
| ML model | Precision ≥ 95% at chosen threshold; registered in MLflow; explainability artifacts generated |
| LangGraph agent | Full MAPE-K cycle completes; crash recovery verified; no duplicate actions; HITL interrupts work |
| Guardrails | Hard rules cannot be bypassed; latency < 500 ms; conflict resolution tested |
| Fleet simulator | All chaos scenarios reproducible; compensating actions verified; zero data loss |
| Reliability Checker | Every decision audited; trust score vetoes unsafe decisions; async evaluation updates final scores |
| Dashboards | Fleet health, trust trend, approval queue, and audit trail visible |
| Final report | All success metrics measured; cross-dataset analysis complete; limitations documented |

---

# 16. Final Acceptance Criteria

The project is complete when the following are true:

1. The model predicts disk failure with high precision.
2. Predictions generalize reasonably from Backblaze to SMART-Z.
3. The LangGraph MAPE-K loop runs continuously in simulation.
4. The system recovers from crashes without duplicate actions.
5. Guardrails prevent all hard safety violations.
6. Stale or low-confidence telemetry cannot trigger destructive actions.
7. Human-in-the-loop escalation works for blocked or uncertain actions.
8. The fleet simulator validates remediation and compensating actions.
9. The Reliability Checker produces meaningful provisional and final trust scores.
10. Trust scores correctly veto unsafe or hard-guardrail-violating decisions.
11. Every decision is explainable and auditable.
12. Dashboards provide visibility into fleet health, trust, approvals, and compliance.
13. Chaos and replay tests demonstrate robustness under abnormal conditions.

---

# 17. Final Statement

This plan delivers more than a disk-failure prediction model. It delivers a validated autonomous system.

The data pipeline creates trustworthy features.
The model identifies risk with high precision.
The agent plans and executes remediation.
The guardrails prevent unsafe behavior.
The human-review layer handles uncertainty.
The Reliability Checker audits every decision.
The dashboards and audit logs make autonomy inspectable.

Together, these phases answer the core question:

> **Can an autonomous self-healing system be trusted to act, and can we prove that its decisions are correct, safe, necessary, timely, and explainable?**
