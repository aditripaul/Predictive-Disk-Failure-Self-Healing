# Goal and Design Strategy

## AI-Based Autonomous System Validation & Reliability Checker
### Predictive Disk-Failure Self-Healing Agent for Server Fleets

> **The plan as written, before implementation, and kept that way.** Of the
> objectives below, O2's prediction target is the one not met, and O1 is
> Backblaze-only: SMART-Z is harmonized in code but has never been evaluated.
> O3 to O6 were met in simulation. Measured outcomes:
> `docs/system_summary.md` and `docs/model_status_and_runbook.md`.

---

# 1. Goal

Hardware failure in data centers is inevitable. Modern storage and compute platforms increasingly use predictive maintenance systems to detect failing components before they cause outages. In many cases, these systems are moving from human-assisted remediation toward autonomous remediation.

However, an autonomous agent that acts without human approval must be trusted. A wrong, premature, or unsafe action can cause:

- unnecessary operational cost;
- service degradation;
- replication-quorum violations;
- data loss;
- outage amplification.

The central gap this project addresses is therefore not only:

> “Can the system predict disk failure and act autonomously?”

but more importantly:

> “Can we validate that the autonomous system is correct, safe, necessary, timely, and explainable before, during, and after it acts?”

This project therefore builds a two-layer system:

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
   - Produces explainability and audit logs.

The project is simulation-first. The agent operates against a simulated fleet before any consideration of real-world deployment.

---

# 2. Problem Statement

Disk failures often show early warning signs through SMART telemetry, such as:

- increasing reallocated sector counts;
- growing pending sector counts;
- uncorrectable errors;
- read/seek error rate degradation;
- spin retries;
- command timeouts.

However, raw daily SMART values are noisy. A single abnormal reading may be transient. Conversely, a drive may appear stable while its degradation rate is accelerating.

Therefore, the system must learn **failure trajectories**, not merely react to single-day anomalies.

At the same time, autonomous remediation must be conservative. False positives waste operational resources and may unnecessarily migrate healthy drives. False negatives may cause data loss. Because safety is paramount, the system adopts a **precision-first** policy.

---

# 3. Scope

## 3.1 In Scope

The project includes:

1. SMART telemetry ingestion and harmonization.
2. Time-window and trajectory-based feature engineering.
3. Disk-failure prediction using tree-based models.
4. Optional sequence-based model comparison using LSTM.
5. LangGraph-based MAPE-K autonomous agent.
6. Guardrail engine with hard and soft rules.
7. Human-in-the-loop approval queue.
8. Fleet simulator with compensating actions.
9. Reliability Checker and trust scoring.
10. Dashboards for fleet health, trust, audit, and approvals.
11. Testing using unit tests, chaos injection, and replay.
12. Final evaluation across datasets and failure scenarios.

## 3.2 Out of Scope for MVP

The project does not require:

1. Direct control of physical production hardware.
2. Full distributed storage engine implementation.
3. Multi-tenant production deployment.
4. Real-time kernel-level telemetry collection.
5. Fully autonomous production operation without supervision.

The system is evaluated in a simulated operational environment.

---

# 4. Objectives and Success Criteria

| # | Objective | Success Criterion |
|---:|---|---|
| O1 | Ingest and harmonize SMART telemetry from Backblaze and SMART-Z | Canonical schema; labeled healthy/failing trajectories |
| O2 | Predict disk failure within N days | Precision ≥ 95%; Recall ≥ 35–50% |
| O3 | Implement guardrailed agentic decision loop | Zero hard-guardrail violations in simulation |
| O4 | Execute autonomous remediation in fleet simulator | Actions complete without data loss; compensating actions on failure |
| O5 | Implement reliability checker and trust scoring | Trust score discriminates trustworthy vs. untrustworthy decisions |
| O6 | Produce audit trail and explainability | Per-decision rationale, guardrail status, checkpoint history |

---

# 5. Design Principles

The system is governed by the following design principles.

## 5.1 Safety over Availability

A false positive is preferable to an unsafe false negative. The system should rather over-monitor or request human review than perform an unsafe destructive action.

## 5.2 Precision-First Autonomy

Autonomous actions are operationally expensive. The system must avoid unnecessary migrations or drains. Therefore, high-confidence thresholds are required before destructive or disruptive actions.

## 5.3 Explainability by Default

Every decision must be explainable in human-readable terms. The Reliability Checker must be able to answer:

- Why did the system predict failure?
- Why was this action chosen?
- Which guardrails were evaluated?
- Why was the action blocked or approved?
- Was the decision later confirmed as correct?

## 5.4 Idempotent and Retryable Actions

Every action must have a unique action ID. Retries must not cause duplicate migrations, duplicate drains, or repeated compensations.

## 5.5 Crash-Safe Operation

The LangGraph workflow must checkpoint every node transition. If the system crashes, it must resume without duplicating actions or losing decision context.

## 5.6 Lean Agent State

The LangGraph state must not contain large telemetry payloads. It should store identifiers, metadata, action proposals, guardrail results, and checkpoint references. Bulk telemetry and feature data should reside in external storage such as DuckDB, Parquet, or Redis.

## 5.7 RAM-Aware Data Engineering

Because Backblaze data can contain millions of drive-days, the data pipeline should avoid loading entire datasets into memory.

The recommended implementation stack uses:

- Polars for lazy DataFrame processing;
- DuckDB for out-of-core SQL over Parquet;
- PyArrow for columnar I/O;
- Parquet with ZSTD compression for storage.

Pandas may be used only for small compatibility tasks if required by a specific library.

## 5.8 Telemetry Freshness Matters

Missing or stale telemetry is not harmless silence. A drive that stops reporting may itself be failing. Therefore, feature confidence and telemetry freshness must influence action gating.

## 5.9 Simulation-First Validation

All autonomous behaviors are first validated against a simulated fleet. The simulator must model drive health, telemetry, action outcomes, failures, and compensating actions.

---

# 6. System Safety Invariants

The following invariants must never be violated by the autonomous agent.

| Invariant | Description |
|---|---|
| No last-node drain | Never drain the last healthy node in a failure domain |
| Quorum preservation | Maintain replication quorum before and after action |
| Concurrency limit | Enforce maximum concurrent disruptive actions |
| Idempotency | Retries must not duplicate actions |
| Crash safety | Recovery must not repeat completed actions |
| Telemetry freshness | Stale or missing telemetry blocks destructive actions |
| Human veto | Human rejection overrides autonomous action |
| Safe timeout | If human review times out, choose safest non-destructive action |
| Audit completeness | Every decision must have an audit record |

---

# 7. High-Level Architecture

The system consists of a simulated fleet, the self-healing agent, the reliability checker, and the audit/dashboard layer.

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
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 1 · SELF-HEALING AGENT                                │
│                     LangGraph MAPE-K Runtime                                    │
│                                                                                 │
│  ┌─────────┐   ┌─────────┐   ┌──────────────────┐   ┌───────────┐   ┌───────┐   │
│  │ MONITOR │─▶│ ANALYZE │─▶│ PLAN +           │─▶│ EXECUTE   │─▶│VALIDATE│  │
│  │         │   │         │   │ GUARDRAIL ENGINE │   │           │   │        │  │
│  │ ingest  │   │ predict │   │ action proposal  │   │ cordon /  │   │ verify │  │
│  │ fleet   │   │ p_fail  │   │ threshold check  │   │ migrate / │   │ outcome│  │
│  │ state   │   │ rank    │   │ hard/soft rules  │   │ drain     │   │        │  │
│  └─────────┘   └─────────┘   └────────┬─────────┘   └───────────┘   └───┬────┘  │
│                                       │                                  │      │
│                                       │ blocked / uncertain              │      │
│                                       ▼                                  │      │
│                              ┌────────────────┐                          │      │
│                              │ HUMAN REVIEW   │                          │      │
│                              │ FastAPI queue  │                          │      │
│                              └────────────────┘                          │      │
│                                                                          │      │
└──────────────────────────────────────────────────────────────────────────┼──────┘
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

# 8. Component Responsibilities

| Component | Responsibility |
|---|---|
| Fleet Simulator | Simulates drives, nodes, telemetry, failure progression, action outcomes |
| Monitor | Reads current fleet state and SMART telemetry batches |
| Analyze | Runs failure prediction model and ranks risky drives |
| Plan | Converts predictions into action proposals |
| Guardrail Engine | Evaluates safety, operational, and policy constraints |
| Execute | Performs approved actions in the simulator |
| Validate | Verifies outcome and emits decision/outcome record |
| Human Review Queue | Receives blocked or uncertain critical actions |
| Reliability Checker | Audits decisions and produces trust scores |
| Dashboards | Displays health, trust, approvals, audit trail |
| MLOps Pipeline | Retrains model on drift signal or degraded performance |

---

# 9. MAPE-K Agent Workflow

The self-healing agent follows a cyclic MAPE-K pattern:

- Monitor
- Analyze
- Plan
- Execute
- Validate
- Knowledge / checkpoint state

```text
                         ┌───────────────────────────────┐
                         │       SCHEDULE / EVENT        │
                         │   timer, telemetry batch,     │
                         │   alert, or manual trigger    │
                         └──────────────┬────────────────┘
                                        │
                                        ▼
                               ┌────────────────┐
                               │    MONITOR     │
                               │                │
                               │ read fleet     │
                               │ snapshot,      │
                               │ SMART batch,   │
                               │ open actions   │
                               └───────┬────────┘
                                       │
                                       ▼
                               ┌────────────────┐
                               │    ANALYZE     │
                               │                │
                               │ load features  │
                               │ predict p_fail │
                               │ rank drives    │
                               └───────┬────────┘
                                       │
                                       ▼
                               ┌────────────────┐
                               │      PLAN      │
                               │                │
                               │ select action  │
                               │ tier based on  │
                               │ confidence and │
                               │ policy         │
                               └───────┬────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   GUARDRAIL ENGINE  │
                            │                     │
                            │ hard rules          │
                            │ soft rules          │
                            │ operational limits  │
                            └──────────┬──────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
                  ▼                                         ▼
          ┌──────────────┐                         ┌────────────────┐
          │   EXECUTE    │                         │ HUMAN REVIEW   │
          │              │                         │                │
          │ cordon       │                         │ FastAPI queue  │
          │ migrate      │                         │ approval /     │
          │ drain        │                         │ rejection      │
          └──────┬───────┘                         └───────┬────────┘
                 │                                         │
                 ▼                                         │
          ┌──────────────┐                                 │
          │   VALIDATE   │◀────────────────────────────────┘
          │              │
          │ verify       │
          │ integrity,   │
          │ service,     │
          │ quorum       │
          └──────┬───────┘
                 │
                 ▼
        ┌────────────────────┐
        │ CHECKPOINT + AUDIT │
        │                    │
        │ decision record    │
        │ action_id          │
        │ guardrail result   │
        │ outcome            │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ RETURN TO MONITOR  │
        │ next cycle         │
        └────────────────────┘
```

## Loop Mechanics

- Validate routes back to Monitor for continuous operation.
- Every node transition is checkpointed.
- Crash recovery must not duplicate actions.
- Cooldowns and hysteresis prevent oscillation.
- Drift signals from Validate can trigger model retraining.

---

# 10. Checkpointing and Crash Recovery

Every important transition in the LangGraph workflow is checkpointed.

```text
┌────────────┐
│ Node step  │
└─────┬──────┘
      │
      ▼
┌────────────────────────┐
│ Save checkpoint        │
│                        │
│ run_id                 │
│ step_id                │
│ action_id              │
│ fleet_snapshot_id      │
│ guardrail_result       │
│ human_review_status    │
└─────┬──────────────────┘
      │
      ▼
┌────────────────────────┐
│ Continue to next node  │
└────────────────────────┘


If crash occurs:

┌────────────┐
│ Crash      │
└─────┬──────┘
      │
      ▼
┌────────────────────────┐
│ Restore last checkpoint│
└─────┬──────────────────┘
      │
      ▼
┌────────────────────────┐
│ Check action_id state  │
│                        │
│ already executed?      │
│ pending?               │
│ failed?                │
└─────┬──────────────────┘
      │
      ▼
┌────────────────────────┐
│ Resume idempotently    │
└────────────────────────┘
```

The checkpoint store must remain lightweight. Therefore, the LangGraph state should not embed full telemetry tables.

Recommended state contents:

```text
run_id
timestamp
fleet_snapshot_id
flagged_drive_ids
prediction_summary
proposed_action
guardrail_result
human_review_required
execution_result
validation_result
decision_record_id
```

Bulk telemetry and features remain in external storage.

---

# 11. Data Strategy and Feature Pipeline

The data pipeline transforms raw SMART telemetry into trajectory-based features.

The recommended data engineering stack is:

- Polars for lazy transformations;
- DuckDB for out-of-core SQL;
- PyArrow for columnar I/O;
- Parquet + ZSTD for storage.

This replaces a pandas-first approach for large-scale processing.

```text
┌──────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                         │
│                                                              │
│   Backblaze SMART stats      SMART-Z       Synthetic/Chaos   │
│   primary training           validation    guardrail tests   │
└───────────────┬──────────────────┬─────────────────┬─────────┘
                │                  │                 │
                ▼                  ▼                 ▼
┌──────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                       │
│                                                              │
│   Polars scan_csv / DuckDB read_csv                          │
│   projection pushdown                                        │
│   predicate pushdown                                         │
│   one file/partition at a time                               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         BRONZE ZONE                          │
│                                                              │
│   parsed source data                                         │
│   source-specific schema preserved                           │
│   Parquet + ZSTD                                             │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         SILVER ZONE                          │
│                                                              │
│   canonical schema                                           │
│   drive_id normalization                                     │
│   model_family normalization                                 │
│   SMART attribute harmonization                              │
│   normalized + raw SMART values                              │
│   telemetry gap flags                                        │
│   survivorship filtering                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                          GOLD ZONE                           │
│                                                              │
│   time-window aggregates                                     │
│   deltas and slopes                                          │
│   threshold-crossing counts                                  │
│   lifecycle metadata                                         │
│   model-relative z-scores                                    │
│   labels for N-day failure horizon                           │
│   optional LSTM sequence tensors                             │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       MODEL TRAIN / EVAL                     │
│                                                              │
│   XGBoost / LightGBM                                         │
│   MLflow tracking                                            │
│   precision-recall thresholding                              │
│   explainability artifacts                                   │
└──────────────────────────────────────────────────────────────┘
```

Detailed dataset, labeling, and feature-engineering rules are defined in `dataset_strategy.md`.

---

# 12. Feature Engineering Philosophy

Raw daily SMART values are too noisy for autonomous action. The system therefore engineers features that capture degradation trajectories.

## Feature Families

| Family | Examples | Purpose |
|---|---|---|
| Time-window aggregates | 7/14/30-day mean, median, min, max, std | Reduce noise and establish baseline |
| Derivatives | 7-day delta, 30-day delta, slope, acceleration | Capture degradation speed |
| Event counts | Days with pending sectors > 0, spike counts | Detect intermittent danger events |
| Threshold crossings | Zero-to-non-zero transitions | Detect sudden fault appearance |
| Missingness / staleness | Telemetry gap counts, hours since last telemetry | Treat silence as potentially informative |
| Lifecycle context | Drive age, age², capacity, power-on hours | Model bathtub-curve failure behavior |
| Cross-vendor normalization | Normalized SMART values, model z-scores | Improve SMART-Z generalization |
| Sequence tensors | Last 30 days × top attributes | Optional LSTM branch |

## Example Trajectory Logic

```text
Stable but aged drive:

Reallocated_Sector_Count: 100, 100, 100, 100, 100
30-day slope ≈ 0
Action: monitor


Rapidly degrading drive:

Current_Pending_Sector_Count: 2, 6, 14, 28, 55
7-day slope high
acceleration positive
Action: high-confidence remediation candidate
```

## Feature Confidence

Destructive actions must require more than a high failure probability. They must also require sufficient feature confidence.

```text
destructive_action_allowed =
    p_fail >= high_threshold
    AND feature_confidence >= confidence_threshold
    AND drive_feature_maturity == MATURE
    AND telemetry is not stale
    AND guardrails pass
```

If confidence is low, the system should downgrade to monitor, warn, or cordon, but not drain.

---

# 13. Prediction and Action Policy

The system does not map every prediction directly to a destructive action. Instead, it uses action tiers.

```text
                      p_fail prediction score
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ Low risk    │        │ Medium risk │        │ High risk   │
│             │        │             │        │             │
│ Monitor     │        │ Warn        │        │ Cordon /    │
│             │        │             │        │ Migrate /   │
│ No action   │        │ Alert only  │        │ Drain       │
└─────────────┘        └─────────────┘        └─────────────┘
                                                     │
                                                     ▼
                                            Guardrail checks
                                            and confidence
                                            thresholds
```

## Recommended Action Tiers

| Tier | Trigger | Operational Effect |
|---|---|---|
| Monitor | Low or uncertain risk | No action; continue observation |
| Warn | Medium risk | Dashboard alert; no automatic fleet change |
| Cordon | High risk | Stop new workload placement; non-destructive |
| Migrate | High risk + guardrails pass | Move data/workload proactively |
| Drain | Very high risk + guardrails pass | Remove drive/node from service |
| Human Review | Blocked, uncertain, or critical | Requires explicit approval/rejection |

## Hysteresis and Cooldown

To prevent oscillation:

- a drive must remain high-risk for multiple cycles before escalation;
- after an action, the drive enters a cooldown period;
- repeated actions require new evidence;
- downgrades from high-risk to healthy require sustained stability.

---

# 14. Guardrail Engine

The guardrail engine is a mandatory policy layer between prediction and execution.

## Guardrail Categories

| Type | Examples | Enforcement |
|---|---|---|
| Prediction threshold | High-confidence cutoff | Blocks low-confidence actions |
| Hard pre-action | Never drain last healthy node | Hard block |
| Hard pre-action | Maintain replication quorum | Hard block |
| Hard pre-action | Max concurrent drains | Hard block |
| Soft pre-action | High I/O period | Delay or warn |
| Operational | Maintenance window | Delay or queue |
| Operational | Rate limits | Throttle |
| Post-action | Data integrity check | Compensating action if failed |
| Post-action | Service continuity check | Compensating action if failed |

## Guardrail Decision Flow

```text
                       ┌─────────────────────┐
                       │  Action Proposal    │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Prediction          │
                       │ Threshold Check     │
                       └──────────┬──────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌────────────────┐               ┌────────────────┐
        │ Below threshold│               │ Above threshold│
        │                │               │                │
        │ Monitor/Warn   │               │ Continue       │
        └────────────────┘               └────────┬───────┘
                                                  │
                                                  ▼
                                  ┌─────────────────────────────┐
                                  │ Hard Pre-Action Guardrails  │
                                  │                             │
                                  │ last healthy node?          │
                                  │ quorum maintained?          │
                                  │ max concurrent drains?      │
                                  └──────────────┬──────────────┘
                                                 │
                        ┌────────────────────────┴───────────────────────┐
                        │                                                │
                        ▼                                                ▼
               ┌────────────────┐                              ┌────────────────┐
               │ Hard fail      │                              │ Pass           │
               │                │                              │                │
               │ Block action   │                              │ Continue       │
               │ Human review   │                              └───────┬────────┘
               └────────────────┘                                      │
                                                                       ▼
                                                    ┌──────────────────────────────┐
                                                    │ Operational / Soft Guardrails│
                                                    │                              │
                                                    │ maintenance window?          │
                                                    │ high I/O?                    │
                                                    │ rate limit?                  │
                                                    └──────────────┬───────────────┘
                                                                   │
                                  ┌────────────────────────────────┴──────────────┐
                                  │                                               │
                                  ▼                                               ▼
                         ┌────────────────┐                             ┌────────────────┐
                         │ Soft violation │                             │ Pass           │
                         │                │                             │                │
                         │ Delay / warn / │                             │ Execute action │
                         │ review         │                             └───────┬────────┘
                         └────────────────┘                                     │
                                                                                ▼
                                                              ┌──────────────────────────────┐
                                                              │ Post-Action Validation       │
                                                              │                              │
                                                              │ data integrity?              │
                                                              │ service continuity?          │
                                                              │ quorum still healthy?        │
                                                              └──────────────┬───────────────┘
                                                                             │
                                            ┌────────────────────────────────┴──────────────────┐
                                            │                                                   │
                                            ▼                                                   ▼
                                   ┌────────────────┐                                ┌────────────────┐
                                   │ Fail           │                                │ Pass           │
                                   │                │                                │                │
                                   │ Compensating   │                                │ Record success │
                                   │ action         │                                └────────────────┘
                                   └────────────────┘
```

## Conflict Resolution

When guardrails conflict, resolution follows:

```text
Safety > Operational > Efficiency
```

If multiple rules apply and there is ambiguity, the most restrictive rule wins.

---

# 15. Human Oversight and Escalation

Human oversight is required for blocked critical actions, uncertain high-impact actions, and guardrail exceptions.

## Human-in-the-Loop Flow

```text
┌────────────────────────────────────────────┐
│ Guardrail blocks critical action OR        │
│ confidence is high but policy requires     │
│ human approval                             │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│ Create Human Review Ticket                 │
│                                            │
│ drive_id                                   │
│ prediction confidence                      │
│ proposed action                            │
│ blocked guardrail rule                     │
│ fleet snapshot reference                   │
│ feature explanations                       │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│ FastAPI Approval Queue / Dashboard         │
└──────────────────┬─────────────────────────┘
                   │
       ┌───────────┼────────────┐
       │           │            │
       ▼           ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Approve  │ │ Reject   │ │ Timeout      │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │               │
     │            │               ▼
     │            │      ┌────────────────────┐
     │            │      │ Safe fallback      │
     │            │      │                    │
     │            │      │ cordon/monitor,    │
     │            │      │ do not drain       │
     │            │      └────────────────────┘
     │            │
     ▼            ▼
┌────────────────────────────────────────┐
│ Resume LangGraph workflow              │
│                                        │
│ approve → execute with action_id       │
│ reject  → safe no-op / monitor         │
└────────────────────────────────────────┘
```

## Human Review Requirements

Every override must include:

- operator identifier;
- timestamp;
- reason code;
- optional comment;
- action ID;
- fleet snapshot ID.

## SLA Targets

| Review Type | SLA |
|---|---:|
| Critical blocked action | 1 hour |
| Non-critical review | 24 hours |

If a critical review times out, the system defaults to the safest non-destructive action.

---

# 16. Reliability Checker and Trust Framework

The Reliability Checker is the meta-validator. It does not merely log decisions; it evaluates decision quality.

## Reliability Checks

| Check | Pass Condition |
|---|---|
| Correctness | True positive or true negative |
| Necessity | No action on drives that would not have failed |
| Guardrail compliance | Zero hard violations; soft violations documented |
| Safety | No data loss; no quorum breach |
| Timeliness | Action completed sufficiently before actual failure |

## Trust Score Formula

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

## Veto Semantics

Any safety violation or hard-guardrail violation forces:

```text
TrustScore = 0
```

This represents a vetoed or untrustworthy decision, regardless of other factors.

---

# 17. Reliability Checker Workflow

```text
                       ┌────────────────────────┐
                       │ Decision + Outcome     │
                       │ Record                 │
                       └───────────┬────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ Correctness  │          │ Necessity    │          │ Timeliness   │
│              │          │              │          │              │
│ Was failure  │          │ Was action   │          │ Was action   │
│ predicted    │          │ actually     │          │ completed    │
│ correctly?   │          │ needed?      │          │ early enough?│
└──────┬───────┘          └──────┬───────┘          └──────┬───────┘
       │                         │                         │
       └────────────┬────────────┴────────────┬────────────┘
                    │                         │
                    ▼                         ▼
             ┌──────────────────────────────────────┐
             │ BaseScore                            │
             │                                      │
             │ 0.4 Correctness                      │
             │ 0.3 Timeliness                       │
             │ 0.3 Necessity                        │
             └──────────────────┬───────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────┐
             │ Safety Check                         │
             │                                      │
             │ data loss?                           │
             │ quorum breach?                       │
             └──────────────────┬───────────────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
            ┌──────────────┐      ┌──────────────┐
            │ Violation    │      │ No violation │
            │              │      │              │
            │ Safety       │      │ Continue     │
            │ multiplier=0 │      └──────┬───────┘
            └──────┬───────┘             │
                   │                     ▼
                   │          ┌────────────────────────┐
                   │          │ Guardrail Check        │
                   │          │                        │
                   │          │ hard / soft / none     │
                   │          └───────────┬────────────┘
                   │                      │
                   ▼                      ▼
             ┌──────────────────────────────────────┐
             │ Final TrustScore                     │
             │                                      │
             │ If safety or hard violation: 0       │
             └──────────────────┬───────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────┐
             │ Explainability + Audit Log           │
             │                                      │
             │ rationale, guardrails, features,     │
             │ human override, checkpoint IDs       │
             └──────────────────────────────────────┘
```

---

# 18. Provisional and Final Trust Scores

Correctness often cannot be evaluated immediately because failure labels depend on a future horizon.

Example:

```text
Prediction: drive will fail within 14 days
Action taken today
Correctness known only after 14 days
```

Therefore, trust scores should have two states.

| State | When Produced | Description |
|---|---|---|
| Provisional Trust Score | Immediately after action | Based on safety, guardrails, execution result, necessity proxy |
| Final Trust Score | After label horizon completes | Includes correctness and timeliness |

```text
Action executed
      │
      ▼
Provisional TrustScore
      │
      │ wait N days / observe outcome
      ▼
Final TrustScore
```

---

# 19. Audit Trail and Explainability

Every decision must produce a structured audit record.

## Minimum Audit Fields

```text
decision_id
run_id
timestamp
drive_id
fleet_snapshot_id
prediction_score
predicted_failure_horizon
proposed_action
final_action
guardrail_result
blocked_rules
human_review_required
human_decision
override_reason_code
execution_status
validation_status
correctness_status
trust_score_provisional
trust_score_final
checkpoint_id
feature_explanations
```

## Explainability Requirements

The system should be able to generate explanations such as:

```text
Drive D-1042 was flagged because:

- 30-day slope of current_pending_sector_count increased sharply.
- Pending sector count rose from 4 to 31 in 7 days.
- Offline uncorrectable errors became non-zero in the last 3 days.
- Prediction confidence exceeded the migrate threshold.
- Guardrails passed.
- Migration completed 9 days before predicted failure horizon.
```

---

# 20. Deployment Maturity Modes

The system should progress through operational maturity levels.

```text
┌────────────┐    ┌────────────┐    ┌────────────────┐    ┌────────────────┐
│ Offline    │ -> │ Shadow     │ -> │ Human-Approved │ -> │ Supervised     │
│ Training   │    │ Monitoring │    │ Actions        │    │ Autonomy       │
└────────────┘    └────────────┘    └────────────────┘    └────────────────┘
```

| Mode | Description |
|---|---|
| Offline Training | Model training and evaluation only |
| Shadow Monitoring | Agent predicts but does not act |
| Human-Approved Actions | Agent proposes; human approves |
| Supervised Autonomy | Agent acts within strict guardrails; humans monitor |
| Full Autonomy | Future production goal, not required for MVP |

---

# 21. Fleet Simulator and Failure State Machine

The fleet simulator models drives, nodes, replication groups, and action outcomes.

## Drive State Machine

```text
                 ┌────────────┐
                 │  HEALTHY   │
                 └─────┬──────┘
                       │
                       ▼
                 ┌────────────┐
                 │ DEGRADED   │
                 └─────┬──────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PREDICTED       │
              │ FAILURE         │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ CORDONED │ │ MIGRATED │ │ DRAINED  │
   └────┬─────┘ └────┬─────┘ └────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
              ┌────────────┐
              │ REPLACED / │
              │ RECOVERED  │
              └────────────┘


Failure path:

HEALTHY / DEGRADED / PREDICTED_FAILURE
                 │
                 ▼
              FAILED
```

The simulator must also model:

- telemetry gaps;
- false-positive spikes;
- simultaneous multi-drive failures;
- action timeouts;
- partial migration failures;
- compensating actions.

---

# 22. Testing, Chaos Injection, and Replay

The project requires a layered testing strategy.

```text
┌─────────────────────────────────────────────────────────────┐
│                       EVALUATION PYRAMID                    │
└─────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │ Chaos Tests  │
                         │              │
                         │ multi-drive  │
                         │ failures,    │
                         │ stale data,  │
                         │ action crash │
                         └──────┬───────┘
                                │
                       ┌────────┴────────┐
                       │ Replay Tests    │
                       │                 │
                       │ checkpoint      │
                       │ time-travel,    │
                       │ no duplicate    │
                       │ actions         │
                       └────────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    │ Integration Tests     │
                    │                       │
                    │ LangGraph + simulator │
                    │ guardrails + API      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │ Unit Tests                        │
              │                                   │
              │ features, labels, guardrails,     │
              │ trust score, idempotency          │
              └───────────────────────────────────┘
```

## Required Test Categories

| Test Type | Purpose |
|---|---|
| Unit tests | Validate feature logic, guardrails, trust score |
| Integration tests | Validate node transitions and simulator interactions |
| Chaos tests | Validate behavior under abnormal conditions |
| Replay tests | Validate crash recovery and checkpoint consistency |
| Cross-dataset tests | Validate SMART-Z generalization |
| Guardrail tests | Validate zero hard-guardrail violations |
| Human-review tests | Validate approval, rejection, timeout paths |

---

# 23. Technology Stack

The updated stack preserves the original architectural intent but improves data-processing scalability.

| Area | Recommended Choice | Notes |
|---|---|---|
| Agent orchestration | LangGraph | Sole runtime orchestrator for MAPE-K loop |
| State checkpointing | SQLite / LangGraph saver | Lean state only |
| Fleet-state cache | Redis | Low-latency operational state |
| Data processing | Polars | Lazy, RAM-efficient DataFrame processing |
| SQL engine | DuckDB | Out-of-core queries over Parquet |
| Columnar I/O | PyArrow | Efficient Parquet read/write |
| Storage | Parquet + ZSTD | Bronze/silver/gold layers |
| ML models | XGBoost / LightGBM | Primary tree-based predictors |
| Optional sequence model | PyTorch LSTM | Comparison branch |
| Experiment tracking | MLflow | Metrics, artifacts, model registry |
| API | FastAPI | Human approval queue and audit APIs |
| Dashboard | Streamlit | Fleet health, trust, approvals |
| Testing | pytest, Hypothesis, Locust optional | Unit, property, load tests |
| Logging | structlog / JSON logs | Audit-friendly structured logs |
| Optional tracing | OpenTelemetry | Use if distributed tracing needed |

---

# 24. Deliverables

The project deliverables are:

1. Data pipeline with time-window feature engineering.
2. XGBoost/LightGBM failure-prediction models with MLflow tracking.
3. LangGraph MAPE-K workflow with checkpoints and interrupts.
4. Guardrail Engine with hard/soft rule catalog.
5. Fleet simulator with compensating actions.
6. Reliability Checker with trust score and veto semantics.
7. Explainability and audit logging.
8. PyTest, chaos injection, and replay test suite.
9. FastAPI approval queue.
10. Streamlit dashboards for fleet health, trust, compliance, and audit.
11. Final report with cross-dataset and guardrail-effectiveness analysis.

---

# 25. Milestones

| Milestone | Deliverable |
|---|---|
| M0 Bootstrap | Repository, configs, data contracts, golden dataset |
| M1 Data & Features | Dataset ingestion, harmonization, aggregate pipeline |
| M2 Models + MLOps | Trained/benchmarked models; LangGraph spike |
| M3 LangGraph Core | Cyclic graph, checkpointing, recovery tests |
| M4 Guardrails | Rule catalog, guardrail engine, test harness |
| M5 Simulation | Fleet simulator + compensating actions |
| M6 Validation | Reliability checker, trust score, human-in-loop |
| M7 Dashboard | MVP dashboards + approval queue |
| M8 Evaluation | Chaos testing, cross-dataset evaluation, final report |

---

# 26. Target Metrics

| Metric | Target |
|---|---:|
| Loop cycle time | < 5 minutes |
| Guardrail evaluation latency | < 500 ms |
| Hard-guardrail compliance | 100% |
| Soft-guardrail compliance | ≥ 95% |
| Prediction precision | ≥ 95% |
| Prediction recall | 35–50% |
| False-positive action rate | < 5% |
| False-negative action rate | < 2% |
| Data-loss events in simulation | 0 |
| Quorum violations in simulation | 0 |
| Duplicate actions after crash recovery | 0 |

---

# 27. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| False alarms cause OpEx waste | Unnecessary migrations/drains | Precision-first thresholding and action tiers |
| Unsafe autonomous action | Data loss or outage | Hard guardrails, simulator validation, human review |
| LangGraph state bloat | Slow checkpointing, crash recovery issues | Lean state; store bulk data externally |
| Mid-cycle crash | Duplicate or lost actions | Checkpointing, idempotent action IDs |
| Class imbalance | Poor precision/recall | Class weighting, threshold tuning, AUPRC |
| Missing telemetry | Hidden failure signals | Gap flags, stale telemetry policy, feature confidence |
| SMART-Z schema mismatch | Poor generalization | Canonical schema, normalized values |
| Synthetic data unrealistic | Weak chaos testing | Trajectory morphing from real failures |
| Human review delay | Agent stalls | Timeout degradation to safest action |
| Model drift | Degraded reliability | Drift detection and retrain signal |
| Single orchestration dependency | Project risk | Pinned versions, early LangGraph spike |

---

# 28. Definition of Success

The project is successful if the final system demonstrates all of the following:

1. The model predicts disk failure with high precision.
2. Predictions generalize reasonably from Backblaze to SMART-Z.
3. The LangGraph MAPE-K loop runs continuously in simulation.
4. The system recovers from crashes without duplicate actions.
5. Guardrails prevent all hard safety violations.
6. Human-in-the-loop escalation works for blocked or uncertain actions.
7. The fleet simulator validates remediation and compensating actions.
8. The Reliability Checker produces meaningful trust scores.
9. Trust scores correctly veto unsafe or hard-guardrail-violating decisions.
10. Every decision is explainable and auditable.
11. Dashboards provide visibility into fleet health, trust, approvals, and compliance.
12. Chaos and replay tests demonstrate robustness under abnormal conditions.

---

# 29. Final Design Statement

This project does not merely build a disk-failure prediction model. It builds a **validated autonomy framework**.

The prediction model identifies risk.
The agent plans and executes remediation.
The guardrails prevent unsafe behavior.
The human-review layer handles uncertainty.
The Reliability Checker audits every decision.
The dashboards and audit logs make autonomy inspectable.

Together, these layers answer the core question:

> **Can an autonomous self-healing system be trusted to act, and can we prove that its decisions are correct, safe, necessary, timely, and explainable?**

---

# 30. References

1. Google Cloud Blog: Seagate & Google predict HDD failures with ML
2. Wei et al., SMART-Z dataset, Scientific Data 2025
3. Backblaze Hard Drive Test Data
4. Lu et al., Making Disk Failure Predictions SMARTer!, USENIX FAST 2020
5. LangGraph Documentation
6. Kephart & Chess, The Vision of Autonomic Computing, 2003
7. Chen & Guestrin, XGBoost, KDD 2016
8. Chawla et al., SMOTE, 2002
