# Dataset Strategy & Feature Engineering

## AI-Based Autonomous System Validation & Reliability Checker
### Predictive Disk-Failure Self-Healing Agent for Server Fleets

> **The plan as written, before implementation, and kept that way.** The
> prediction target stated here (precision >= 95%, recall 35-50%) was not met;
> the primary horizon is now 30 days, and per-action thresholds are set from
> lift rather than precision. Measured outcomes: `docs/system_summary.md`
> and `docs/model_status_and_runbook.md`.

**Document role:**
This document is the single source of truth for dataset selection, data preprocessing, labeling, feature engineering, data quality, leakage prevention, and ML-readiness for the project.

---

# 1. Summary

This project predicts impending disk failure using SMART telemetry from real-world drive fleets. The core challenge is not simply detecting abnormal SMART values on a single day. The real challenge is detecting **degradation trajectories** early enough to allow safe autonomous remediation, while maintaining extremely high precision.

Raw daily SMART values are noisy, vendor-dependent, and often incomplete. A single abnormal reading may be transient, while a drive may appear stable even while its degradation rate is accelerating. Therefore, the dataset and feature-engineering strategy transforms raw daily SMART telemetry into trajectory-based features that capture:

- recent level changes;
- multi-day volatility;
- rate of degradation;
- acceleration of failure indicators;
- intermittent fault events;
- telemetry freshness and missingness;
- lifecycle and hardware-context signals;
- vendor-normalized and model-relative baselines.

The primary training dataset is **Backblaze Hard Drive Stats**. Cross-vendor generalization is evaluated using **SMART-Z**. Synthetic and chaos datasets are used to stress-test the autonomous agent, guardrails, and reliability checker.

The pipeline is designed to be:

- **RAM-efficient**, using Polars, DuckDB, PyArrow, and Parquet;
- **leakage-aware**, using strict temporal feature windows and censoring;
- **explainable**, using interpretable time-window and slope-based features;
- **operationally safe**, supporting precision-first thresholding for autonomous actions;
- **auditable**, using versioned datasets, feature registry, and reproducible transformations.

---

# 2. Strategic Objectives

The dataset and feature-engineering strategy must directly support the project’s core objectives.

| Objective | Dataset / Feature Requirement |
|---|---|
| Predict failure within N days | Time-aware labels for 7/14/30-day horizons |
| Precision ≥ 95% | Trajectory features, threshold tuning, AUPRC-based evaluation |
| Recall 35–50% | Early-warning trajectory features and calibrated operating point |
| Cross-vendor robustness | Canonical SMART schema and normalized/raw harmonization |
| Explainability | Interpretable features such as slopes, counts, and rolling aggregates |
| Guardrail testing | Synthetic chaos data with missing telemetry and cascading failures |
| RAM-constrained execution | Out-of-core Polars/DuckDB pipeline with Parquet storage |
| Auditability | Versioned datasets, feature registry, and reproducible transformations |
| Safe autonomy | Feature confidence and telemetry-freshness gating for destructive actions |

---

# 3. Datasets

## 3.1 Backblaze Hard Drive Stats - Primary Training Dataset

**Purpose:**
Primary dataset for model training, baseline evaluation, threshold tuning, and feature engineering.

**Scale:**
Millions of drive-days of production telemetry.

**Content:**

- daily SMART attributes;
- drive model;
- serial number;
- capacity;
- explicit failure indicators;
- replacement/removal signals in some releases.

**Role in project:**

- train the main failure-prediction model;
- learn failure trajectories from real production drives;
- tune precision-first decision thresholds;
- generate realistic synthetic chaos trajectories through trajectory morphing.

**Limitations:**

- severe class imbalance, with failures below 1% of drive-days;
- vendor-specific SMART reporting inconsistencies;
- missing telemetry days;
- ambiguous removal/replacement cases;
- survivorship bias if failed or removed drives are handled incorrectly.

---

## 3.2 SMART-Z - Cross-Vendor Validation Dataset

**Purpose:**
Evaluate model generalization across vendors, firmware behaviors, and SMART attribute implementations.

**Scale:**
147,496 disks with 65 standardized SMART attributes.

**Role in project:**

- validate whether features learned from Backblaze generalize to other vendors;
- test robustness of the canonical SMART schema;
- evaluate vendor-normalized versus raw SMART representations;
- identify model overfitting to Backblaze-specific patterns.

**Key requirement:**
SMART-Z must not be used as the primary training set unless explicitly performing a controlled cross-dataset training experiment. Its primary role is validation and generalization testing.

---

## 3.3 Synthetic & Chaos Dataset

**Purpose:**
Stress-test the agent loop, guardrails, reliability checker, and compensating-action logic.

**Content:**

- stale or missing telemetry;
- sudden multi-drive failures;
- false-positive SMART spikes;
- transient errors that recover;
- clock skew and noisy timestamps;
- partial action failures;
- quorum-threatening failure cascades.

**Generation strategy:**
The synthetic dataset should not rely only on random noise. Instead, it should use **trajectory morphing**:

1. Extract real failure trajectories from Backblaze.
2. Add noise and measurement jitter.
3. Compress or stretch the failure timeline.
4. Replicate trajectories across multiple simulated drives.
5. Inject correlated failures into the same replication group or rack.
6. Add telemetry gaps and false-positive spikes.

This creates realistic chaos scenarios that better reflect operational risk.

**Usage restriction:**
Synthetic data is primarily for guardrail, simulator, and chaos testing. It should not be used as the primary training dataset unless explicitly approved for a specific experiment.

---

# 4. Data Architecture

The pipeline uses a layered data architecture to separate immutable raw data from curated ML-ready features.

```text
data/
├── raw/
│   ├── backblaze/
│   ├── smartz/
│   └── synthetic/
│
├── bronze/
│   ├── backblaze/
│   ├── smartz/
│   └── synthetic/
│
├── silver/
│   ├── canonical_telemetry/
│   ├── drive_metadata/
│   └── telemetry_gaps/
│
├── gold/
│   ├── features/
│   ├── labels/
│   ├── splits/
│   └── sequences/
│
└── audit/
    ├── dataset_versions/
    ├── feature_registry/
    └── data_quality_reports/
```

---

## 4.1 Zone Definitions

| Zone | Purpose |
|---|---|
| `raw/` | Immutable downloaded source files |
| `bronze/` | Parsed source data with minimal transformation |
| `silver/` | Harmonized canonical schema, cleaned and gap-aware |
| `gold/` | ML-ready features, labels, splits, and sequence tensors |
| `audit/` | Dataset versions, feature definitions, quality reports |

---

## 4.2 Storage Format

All large datasets should be stored as:

```text
Parquet + ZSTD compression
```

Recommended partitioning:

```text
dataset/year/month
```

Example:

```text
data/silver/canonical_telemetry/source=backblaze/year=2024/month=03/
```

For some drive-level operations, secondary partitioning by drive hash may be useful:

```text
drive_hash=00/
drive_hash=01/
...
```

---

# 5. RAM-Constrained Processing Strategy

Because Backblaze data can contain millions of rows, the pipeline must avoid loading full datasets into memory.

## Recommended Processing Stack

| Task | Tool |
|---|---|
| Lazy DataFrame processing | Polars |
| Out-of-core SQL | DuckDB |
| Columnar I/O | PyArrow |
| Storage | Parquet + ZSTD |
| Sequence storage | zarr or memory-mapped NumPy |
| Metadata/config | YAML + Pydantic |
| Testing | pytest + golden datasets |

---

## Core Rules

1. **Do not use pandas as the primary processing engine.**
   - Polars should be the default engine.
   - pandas may be used only for small compatibility tasks if required by a library.

2. **Never load all Backblaze data into memory at once.**
   - Process quarterly files or monthly partitions one at a time.

3. **Use lazy scans.**
   - Use `polars.scan_csv()` instead of eager CSV loading.
   - Apply projection and predicate pushdown.

4. **Materialize intermediate outputs to Parquet.**
   - Avoid repeatedly recomputing expensive transformations.

5. **Use DuckDB for large window functions.**
   - DuckDB can perform windowed aggregations directly on Parquet files.

6. **Downcast numeric types.**
   - Use `float32` instead of `float64` where precision permits.
   - Use `int16` or `int32` for counters and ages.

7. **Avoid wide feature explosions.**
   - Start with the top 10–15 SMART attributes.
   - Add more attributes only if validation metrics improve.

---

# 6. Canonical Schema

The raw datasets must be mapped into a unified canonical schema.

## 6.1 Telemetry Schema

| Field | Type | Description |
|---|---|---|
| `drive_id` | string | Unique drive identifier |
| `date` | date | Telemetry date |
| `source_dataset` | categorical | `backblaze`, `smartz`, `synthetic` |
| `drive_model` | string | Raw model name |
| `model_family` | string | Normalized model family |
| `manufacturer` | string | Inferred or normalized vendor |
| `capacity_gb` | float32 | Drive capacity in GB |
| `drive_type` | categorical | HDD/SSD if known |
| `smart_attribute_id` | string/int | SMART attribute ID |
| `smart_attribute_name` | string | Canonical SMART attribute name |
| `smart_raw_value` | float32 | Raw SMART value |
| `smart_normalized_value` | float32 | Vendor-normalized value |
| `smart_badness_value` | float32 | Direction-normalized value where higher means worse |
| `failure_date` | date/null | Confirmed failure date if known |
| `removal_date` | date/null | Removal or replacement date |
| `label` | int8 | Failure label for selected horizon |
| `days_since_last_telemetry` | int16 | Days since previous telemetry record |
| `telemetry_gap_flag` | int8 | 1 if gap exceeds threshold |
| `stale_telemetry_flag` | int8 | 1 if telemetry is too stale for high-confidence action |
| `censoring_flag` | categorical | Failure, replacement, removed, censored |

---

## 6.2 Drive Metadata Schema

| Field | Type | Description |
|---|---|---|
| `drive_id` | string | Unique drive identifier |
| `first_seen_date` | date | First observation date |
| `last_seen_date` | date | Last observation date |
| `failure_date` | date/null | Failure date |
| `replacement_date` | date/null | Replacement date if available |
| `model_family` | string | Normalized model family |
| `capacity_gb` | float32 | Capacity |
| `drive_type` | categorical | HDD/SSD |
| `total_observed_days` | int32 | Observed telemetry days |
| `active_history_days` | int32 | Days available for feature windows |
| `survivorship_valid` | bool | True if ≥ 30 days history |
| `feature_maturity` | categorical | `WARMUP`, `MATURE`, `STALE`, `DECOMMISSIONED` |

---

# 7. Data Preprocessing

## 7.1 Identifier Normalization

Raw drive identifiers must be normalized.

Required operations:

- trim whitespace;
- normalize case where appropriate;
- standardize date formats;
- parse capacity strings into numeric GB;
- normalize model names into model families;
- infer manufacturer where possible.

Example:

```text
ST4000DM000       -> Seagate 4TB HDD
WUS721414AL5204   -> Western Digital 14TB enterprise drive
```

---

## 7.2 SMART Attribute Harmonization

SMART attributes are not perfectly standardized across vendors.

The pipeline must maintain multiple representations:

| Representation | Use |
|---|---|
| `smart_raw_value` | Vendor-specific models and detailed diagnostics |
| `smart_normalized_value` | Cross-vendor models and SMART-Z validation |
| `smart_badness_value` | Direction-normalized feature where higher always means worse |

The canonical schema should map common SMART attributes to stable names.

Examples:

| SMART ID | Common Name | Canonical Name |
|---:|---|---|
| 5 | Reallocated Sector Count | `reallocated_sector_count` |
| 187 | Reported Uncorrectable Errors | `reported_uncorrectable_errors` |
| 188 | Command Timeout | `command_timeout` |
| 197 | Current Pending Sector Count | `current_pending_sector_count` |
| 198 | Offline Uncorrectable | `offline_uncorrectable` |
| 1 | Raw Read Error Rate | `raw_read_error_rate` |
| 7 | Seek Error Rate | `seek_error_rate` |
| 9 | Power-On Hours | `power_on_hours` |
| 10 | Spin Retry Count | `spin_retry_count` |
| 199 | UDMA CRC Error Count | `udma_crc_error_count` |

---

## 7.3 SMART Badness Orientation

Some SMART attributes increase as health worsens. Others, especially vendor-normalized values, may decrease as health worsens.

To avoid confusing the model, the pipeline should define a direction policy per attribute.

Example:

```text
higher raw value means worse:
    smart_badness_value = smart_raw_value

lower normalized value means worse:
    smart_badness_value = normalized_scale_max - smart_normalized_value
```

After orientation normalization:

```text
higher smart_badness_value always means worse drive health
```

This is especially important for cross-vendor generalization.

---

## 7.4 Missing Data Policy

Missing SMART telemetry should not be treated as simply zero.

The recommended policy is:

### Short gaps, for example 1–3 days

- forward-fill the last known value;
- set `days_since_last_telemetry`;
- retain the gap as a feature.

### Longer gaps, for example more than 3 days

- forward-fill only for continuity;
- set `stale_telemetry_flag = 1`;
- treat missingness as potentially informative;
- avoid using these rows for high-confidence autonomous action unless supported by other signals.

### Missingness-derived features

| Feature | Meaning |
|---|---|
| `days_since_last_telemetry` | How stale the latest report is |
| `hours_since_last_telemetry` | Higher-resolution staleness for operational serving |
| `telemetry_gap_7d_count` | Number of missing days in last 7 days |
| `telemetry_gap_30d_count` | Number of missing days in last 30 days |
| `stale_telemetry_flag` | Long gap detected |
| `telemetry_coverage_30d` | Fraction of expected days observed |

This prevents the pipeline from assuming that silence always means health.

---

## 7.5 Survivorship Bias and Feature Maturity

Features based on 7-day, 14-day, or 30-day windows require enough history.

Rules:

- require at least 30 days of observed history before creating full gold features;
- drives with less than 30 days history are marked as:

```text
feature_maturity = WARMUP
```

Drives with sufficient history are marked as:

```text
feature_maturity = MATURE
```

Drives with stale or insufficient recent telemetry may be marked:

```text
feature_maturity = STALE
```

Removed drives are marked:

```text
feature_maturity = DECOMMISSIONED
```

Immature or stale drives may still be monitored, but they should not normally trigger high-confidence destructive autonomous actions unless additional safeguards are satisfied.

---

## 7.6 Log Transformations

Several SMART attributes are highly skewed.

Apply:

```text
log1p(value) = log(1 + value)
```

to attributes such as:

- `reallocated_sector_count`
- `current_pending_sector_count`
- `offline_uncorrectable`
- `reported_uncorrectable_errors`
- `command_timeout`
- `spin_retry_count`

This prevents extreme outliers from dominating model splits and improves feature stability.

---

## 7.7 Outlier Handling

In addition to log transformation, the pipeline should apply robust clipping or winsorization.

Recommended approach:

```text
x_clipped = clip(x, lower=p01, upper=p99)
```

where percentiles are computed per attribute and, where useful, per model family.

This prevents isolated corrupted telemetry values from distorting rolling aggregates.

---

# 8. Labeling Strategy

## 8.1 Prediction Horizon

The system should support multiple horizons:

| Horizon | Use |
|---:|---|
| 7 days | Short-term emergency response |
| 14 days | Default operational horizon |
| 30 days | Early-warning and planning |

The primary operating horizon should be selected during model evaluation. A recommended default is:

```text
N = 14 days
```

---

## 8.2 Label Definition

For each drive-day:

```text
label = 1 if the drive has a confirmed failure within the next N days
label = 0 if the drive is confirmed healthy for the next N days
label = null/censored if the outcome cannot be safely determined
```

The label must only use information that would have been available at prediction time.

---

## 8.3 Event Types

Not all drive removals are the same. The labeling pipeline should distinguish between:

| Event Type | Meaning | Treatment |
|---|---|---|
| Confirmed failure | Drive explicitly failed | Positive event |
| Failure followed by replacement | Replacement caused by failure | Positive event |
| Preventive replacement | Replaced without confirmed failure | Censored or weak positive depending on policy |
| Removed without failure | Drive removed for unrelated reason | Censored, not automatic negative |
| Missing future data | Outcome unknown | Right-censored |

This avoids label corruption from ambiguous operational events.

---

## 8.4 Censoring Rules

A drive-day should be censored if the future outcome cannot be reliably observed.

Examples:

- drive disappears from telemetry before the prediction horizon completes;
- drive is removed without failure;
- replacement reason is unknown;
- telemetry ends due to fleet maintenance rather than failure.

Censored rows should not be treated as guaranteed negatives unless the full horizon is observed.

---

## 8.5 Label Table Schema

| Field | Type | Description |
|---|---|---|
| `drive_id` | string | Drive identifier |
| `date` | date | Prediction date |
| `horizon_days` | int | 7/14/30 |
| `label` | int8 | 1/0/null |
| `event_date` | date/null | Failure or replacement date |
| `event_type` | categorical | failure, replacement, censored, removed |
| `observable_until_horizon` | bool | Whether full horizon is known |
| `days_to_event` | int/null | Days until failure/replacement |

---

# 9. Temporal Splitting Strategy

The project must avoid random train/validation splitting because that can leak future information.

## 9.1 Primary Split: Chronological Split

Recommended:

| Split | Purpose |
|---|---|
| Training | Earliest time period |
| Validation | Middle time period |
| Test | Latest unseen time period |

Example:

```text
Train:      2018–2021
Validation: 2022
Test:       2023
```

Exact dates should depend on dataset availability.

---

## 9.2 Secondary Split: Drive-Level Holdout

To test whether the model is merely memorizing specific drives, create a drive-level holdout:

- some drives appear only in validation/test;
- other drives appear in training.

This evaluates generalization to unseen physical units.

---

## 9.3 External Split: SMART-Z

SMART-Z is used as an external validation set.

Purpose:

- test cross-vendor generalization;
- evaluate robustness to different SMART implementations;
- identify Backblaze-specific overfitting.

---

## 9.4 Split Metadata

Every dataset row should carry split metadata:

| Field | Example |
|---|---|
| `split` | `train`, `validation`, `test`, `external_smartz` |
| `split_version` | `v1.0` |
| `split_strategy` | `chronological`, `drive_holdout`, `vendor_holdout` |

---

# 10. Feature Engineering Strategy

The core principle is:

> Predict failure from degradation trajectories, not from isolated daily SMART values.

The feature pipeline is organized into seven feature families.

---

## 10.1 Feature Family A - Time-Window Aggregates

These features reduce daily noise and establish the recent operating baseline.

For a SMART attribute `x` at time `t`, and window length `W ∈ {7, 14, 30}` days:

```text
rolling_mean_W   = mean(x[t-W+1 : t])
rolling_median_W = median(x[t-W+1 : t])
rolling_min_W    = min(x[t-W+1 : t])
rolling_max_W    = max(x[t-W+1 : t])
rolling_std_W    = std(x[t-W+1 : t])
rolling_range_W  = rolling_max_W - rolling_min_W
```

Example names:

```text
reallocated_sector_count_7d_mean
reallocated_sector_count_30d_max
current_pending_sector_count_14d_std
raw_read_error_rate_7d_range
offline_uncorrectable_30d_median
```

Why this matters:

A high standard deviation in an error-related attribute may indicate unstable hardware even if the average value appears acceptable.

---

## 10.2 Feature Family B - Derivatives and Rate of Change

Failure is a process. These features capture how quickly a drive is degrading.

### Short-term and long-term deltas

```text
delta_7  = x[t] - x[t-7]
delta_14 = x[t] - x[t-14]
delta_30 = x[t] - x[t-30]
```

### Linear slope

For window `W`:

```text
slope_W = least-squares slope of x over the last W days
```

### Acceleration

```text
acceleration = slope_7d - slope_30d
```

or:

```text
acceleration = delta_7d - delta_30d
```

Example names:

```text
current_pending_sector_count_7d_delta
reallocated_sector_count_30d_delta
offline_uncorrectable_14d_slope
pending_sector_acceleration_7d_vs_30d
```

Why this matters:

A drive with 100 reallocated sectors that has been stable for a month is very different from a drive that moved from 10 to 50 sectors in three days.

---

## 10.3 Feature Family C - Threshold Crossings and Event Counts

Averages can hide intermittent but dangerous events. These features detect spikes and fault events.

### Positive-day counts

```text
positive_days_30d = count(days in last 30 days where x > 0)
```

### Spike counts

```text
spike_count_30d = count(days in last 30 days where daily_delta > threshold)
```

Example:

```text
pending_sector_spike_count_30d
```

where:

```text
daily_delta >= 10
```

Thresholds should be attribute-specific and configurable.

### Zero-to-non-zero transition

```text
zero_to_nonzero_flag = 1 if x[t-1] == 0 and x[t] > 0 else 0
```

Example names:

```text
current_pending_sector_positive_30d_count
reallocated_sector_spike_count_30d
offline_uncorrectable_zero_to_nonzero_7d
command_timeout_event_count_14d
```

Why this matters:

A sudden appearance of pending sectors or uncorrectable errors is often more dangerous than a stable historical nonzero value.

---

## 10.4 Feature Family D - Missingness, Staleness, and Feature Confidence

Missing telemetry is not harmless silence. A failing drive may stop reporting SMART data because its reporting agent crashes or the drive becomes unresponsive.

### Telemetry gap features

```text
days_since_last_telemetry
hours_since_last_telemetry
telemetry_gap_7d_count
telemetry_gap_30d_count
stale_telemetry_flag
telemetry_coverage_30d
```

### Feature confidence score

A simple feature-confidence score can be computed as:

```text
feature_confidence =
    telemetry_coverage_30d
  × recency_factor
  × attribute_coverage_factor
```

Where:

```text
recency_factor = exp(-hours_since_last_telemetry / tau)
```

`tau` is a configurable decay constant.

### Action gating

Feature confidence should influence action tiering:

| Condition | Allowed Action |
|---|---|
| High confidence, high failure probability | Cordon / migrate / drain |
| Medium confidence | Warn or monitor |
| Low confidence or stale telemetry | Monitor only; avoid destructive action |

This prevents the agent from performing high-risk actions based on stale or incomplete evidence.

---

## 10.5 Feature Family E - Cross-Vendor Harmonization and Model-Relative Features

SMART attributes differ across vendors. These features improve portability to SMART-Z.

### Normalized SMART values

Use vendor-normalized values, commonly on scales such as:

```text
0–100
0–253
```

where lower normalized values often indicate worse health, depending on vendor semantics.

### Attribute ratios

Useful ratios include:

```text
pending_to_reallocated_ratio =
    current_pending_sector_count / (reallocated_sector_count + 1)

reallocated_per_capacity =
    reallocated_sector_count / capacity_gb

uncorrectable_per_power_on_hour =
    offline_uncorrectable / (power_on_hours + 1)
```

The `+1` avoids division by zero.

### Model-family z-scores

Instead of comparing a drive to the entire fleet, compare it to similar drives:

```text
model_zscore =
    (drive_value - model_family_mean) / (model_family_std + epsilon)
```

Example:

```text
reallocated_sector_count_30d_model_zscore
```

Why this matters:

Some drive models naturally report higher error counts. Model-relative features reduce false alarms caused by model-specific baselines.

---

## 10.6 Feature Family F - Drive Metadata and Lifecycle Context

Hardware failure follows a bathtub curve: higher risk when drives are very new or very old.

### Lifecycle features

| Feature | Description |
|---|---|
| `drive_age_days` | Days since first observation |
| `drive_age_days_squared` | Captures nonlinear aging |
| `capacity_gb` | Drive capacity |
| `drive_type` | HDD/SSD if available |
| `model_family` | Encoded model family |
| `power_on_hours` | Cumulative power-on time |
| `telemetry_coverage_30d` | Fraction of expected days observed |

Example names:

```text
drive_age_days
drive_age_days_squared
capacity_gb
power_on_hours
model_family_encoded
telemetry_coverage_30d
```

Why this matters:

A small increase in reallocated sectors may be more concerning on a relatively new drive than on an old drive with a known high-error profile.

---

## 10.7 Feature Family G - Sequence Features for LSTM Branch

For the optional deep-learning branch, features must be structured as sequences.

### Sequence shape

```text
[samples, time_steps, features]
```

Recommended initial shape:

```text
[samples, 30, 10]
```

where:

- `time_steps = 30` days;
- `features = top 10 SMART attributes`.

### Sequence attributes

Suggested initial attributes:

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

### Time-decay weighting

Older days should receive less weight than recent days:

```text
weight_t = exp(-lambda * age_in_days)
```

This encourages the model to focus on recent degradation.

### Storage

Sequence tensors should not be stored as large in-memory arrays.

Recommended formats:

```text
.zarr
.npy memory-mapped files
Parquet list columns for small prototypes
```

---

# 11. Priority SMART Attributes

The initial feature pipeline should focus on the highest-signal attributes.

| Attribute | Reason |
|---|---|
| `reallocated_sector_count` | Strong indicator of physical media degradation |
| `current_pending_sector_count` | Unstable sectors awaiting reallocation |
| `offline_uncorrectable` | Uncorrectable errors detected offline |
| `reported_uncorrectable_errors` | Runtime uncorrectable errors |
| `raw_read_error_rate` | Read stability |
| `seek_error_rate` | Mechanical actuation health |
| `spin_retry_count` | Motor/startup health |
| `command_timeout` | Controller/interface instability |
| `power_on_hours` | Lifecycle exposure |
| `smart_overall_health_normalized` | Vendor health summary where available |

Additional attributes may be added only after ablation testing demonstrates validation improvement.

---

# 12. Feature Naming Convention

A consistent feature naming convention improves explainability and auditability.

Recommended pattern:

```text
{attribute}_{window}_{operation}
```

Examples:

```text
current_pending_sector_count_7d_delta
current_pending_sector_count_30d_slope
reallocated_sector_count_14d_std
offline_uncorrectable_7d_zero_to_nonzero
command_timeout_30d_positive_count
drive_age_days
telemetry_coverage_30d
```

---

# 13. Feature Registry

All features must be defined declaratively. This ensures reproducibility and auditability.

Example feature registry entry:

```yaml
feature_name: reallocated_sector_count_30d_slope
source_attribute: reallocated_sector_count
value_type: raw_log1p
window_days: 30
operation: linear_slope
partition_by: drive_id
order_by: date
version: 1
dtype: float32
description: >
  Linear slope of log-transformed reallocated sector count over the previous
  30 days. Captures the rate at which reallocated sectors are increasing.
```

Another example:

```yaml
feature_name: current_pending_sector_positive_30d_count
source_attribute: current_pending_sector_count
value_type: raw
window_days: 30
operation: positive_day_count
threshold: 0
partition_by: drive_id
order_by: date
version: 1
dtype: int16
description: >
  Number of days in the last 30 days where current pending sector count was
  greater than zero.
```

The feature registry should be versioned in Git and copied into `audit/feature_registry/` for every model training run.

---

# 14. Feature Confidence and Action Gating

Feature confidence is an operational safety mechanism. It tells the agent whether the feature evidence is fresh and complete enough to support autonomous action.

## 14.1 Confidence Inputs

| Input | Meaning |
|---|---|
| `telemetry_coverage_30d` | How many expected recent telemetry days were observed |
| `hours_since_last_telemetry` | How fresh the latest telemetry is |
| `attribute_coverage_factor` | How many core SMART attributes are available |
| `feature_maturity` | Whether the drive has enough history for stable features |

## 14.2 Drive Maturity States

| State | Meaning |
|---|---|
| `WARMUP` | Less than 30 days of observed history |
| `MATURE` | Sufficient history for stable rolling features |
| `STALE` | Recent telemetry is missing or delayed |
| `DECOMMISSIONED` | Drive removed from active fleet |

## 14.3 Action Gating Rule

Destructive or disruptive actions such as drain should require:

```text
p_fail >= high_threshold
AND feature_confidence >= confidence_threshold
AND feature_maturity == MATURE
AND telemetry is not stale
AND guardrails pass
```

If confidence is low, the agent should downgrade to:

```text
monitor
warn
cordon
```

but not drain.

---

# 15. Class Imbalance Strategy

Disk failure is rare. Failures typically represent less than 1% of drive-days.

Accuracy is therefore a misleading metric.

## Primary evaluation metric

```text
AUPRC - Area Under the Precision-Recall Curve
```

## Secondary metrics

| Metric | Purpose |
|---|---|
| Precision at operating threshold | Operational safety and OpEx control |
| Recall at operating threshold | Failure coverage |
| False-positive action rate | Avoid unnecessary migrations |
| False-negative action rate | Avoid missed failures |
| Calibration | Trustworthiness of predicted probabilities |
| Warning lead time | How early the model warns before failure |

---

## Imbalance Mitigations

### 1. Class weighting

For XGBoost:

```text
scale_pos_weight = negative_count / positive_count
```

For LightGBM:

```text
is_unbalance=true
```

or explicit class weights.

### 2. Precision-first threshold tuning

Do not use the default 0.5 threshold.

Select a threshold on the validation set such that:

```text
precision >= 95%
```

while maximizing recall subject to that constraint.

### 3. Subsampled SMOTE

SMOTE may be used, but only on training data and only if it improves AUPRC.

Recommended policy:

- apply SMOTE only to a stratified training sample;
- never apply SMOTE to validation or test data;
- compare against class weighting;
- prefer class weighting if SMOTE does not clearly improve validation performance.

### 4. Negative downsampling for tuning

For hyperparameter search, negatives may be downsampled to reduce compute cost.

However, final evaluation must use the full validation distribution.

---

# 16. Model Input Strategy

## 16.1 Tree-Based Models

Primary models:

- XGBoost
- LightGBM

Input format:

- tabular Parquet/Arrow table;
- `float32` features;
- explicit train/validation/test split columns.

Recommended training approach:

- histogram-based trees;
- early stopping;
- precision-focused threshold search;
- MLflow experiment tracking.

---

## 16.2 LSTM / Sequence Branch

Input format:

```text
[samples, 30, top_k_features]
```

Recommended implementation:

- PyTorch;
- zarr or memory-mapped NumPy arrays;
- small batch sizes;
- time-decay weighting;
- dropout and early stopping.

The LSTM branch should be compared against the tree-based baseline, not assumed to be superior.

---

# 17. Leakage Prevention Rules

These rules are mandatory.

## Rule 1 - No future data in features

All rolling windows must use only current and past observations.

For date `t`:

```text
features[t] may use data from [t - window + 1, t]
features[t] must not use data from t + 1 or later
```

---

## Rule 2 - Partition by drive

All window operations must be partitioned by drive:

```text
PARTITION BY drive_id
ORDER BY date
```

This prevents one drive’s history from leaking into another drive’s windows.

---

## Rule 3 - Labels must respect time

A label for date `t` may only depend on events occurring after `t` and within the horizon.

```text
label[t] depends on events in (t, t + N]
```

---

## Rule 4 - Censored rows must not become automatic negatives

If the future cannot be observed, the row should be censored or excluded, not labeled healthy by default.

---

## Rule 5 - SMART-Z must remain external unless explicitly stated

SMART-Z should not contaminate Backblaze training unless running a controlled cross-dataset training experiment.

---

# 18. Data Quality Checks

The pipeline must include automated data quality tests.

## Required Checks

| Check | Description |
|---|---|
| Null date check | No telemetry rows without date |
| Duplicate check | No duplicate `drive_id + date` rows |
| Capacity check | Capacity must be positive |
| SMART range check | Normalized values within expected vendor range |
| Label consistency | Failure date must be after or equal to prediction date |
| Horizon observability | Labels require full horizon visibility or censoring |
| Feature maturity | 30-day features require at least 30 days history |
| Telemetry freshness | Stale telemetry is flagged |
| Drift check | Feature distributions monitored across time |

---

## Golden Dataset Regression

Maintain a small golden dataset:

```text
100 healthy drive trajectories
100 failing drive trajectories
```

Every change to feature logic must produce the same expected feature values on this dataset unless the change is intentionally versioned.

The golden dataset should also include:

- missing telemetry cases;
- stale telemetry cases;
- false-positive spike cases;
- drives with short history.

---

# 19. Feature Drift and Monitoring

Once models are deployed or simulated, feature drift must be monitored.

Recommended drift checks:

| Method | Use |
|---|---|
| PSI | Population Stability Index for feature distribution shifts |
| KS test | Distribution shift detection |
| Missingness rate | Detect telemetry gaps |
| Attribute coverage | Detect SMART attribute availability changes |
| Prediction score drift | Detect unusual model output shifts |
| Label feedback | Compare predictions against later observed failures |

The Reliability Checker should consume these drift signals and recommend retraining when necessary.

---

# 20. Operational Feature Serving Strategy

The project begins with batch feature generation, but the agent runtime requires fast feature access during the Analyze stage.

A practical architecture is a **hybrid batch + incremental serving pipeline**.

```text
Offline / Training Pipeline

Backblaze / SMART-Z
       |
       v
Polars / DuckDB batch processing
       |
       v
Silver canonical telemetry
       |
       v
Gold trajectory features
       |
       v
Model training / evaluation / feature backfill


Online / Serving Pipeline

New telemetry event / batch update
       |
       v
Incremental feature updater
       |
       v
Redis online feature store
       |
       v
LangGraph Analyze node queries features by drive_id
```

---

## 20.1 Incremental State Management

Instead of storing 30 days of raw telemetry in memory for every drive, the streaming processor maintains compact incremental state.

For each drive and each monitored attribute:

```text
last_value
last_update_timestamp
circular_buffer_last_N_values
running_mean
running_variance
event_counters
spike_counters
zero_to_nonzero_state
days_since_last_telemetry
feature_confidence
```

Recommended algorithms:

| Feature Type | Incremental Method |
|---|---|
| Rolling mean | Circular buffer or running sum |
| Rolling variance | Welford’s algorithm |
| Rolling min/max | Monotonic deque or periodic recomputation |
| Event counts | Sliding-window counters |
| Slopes | Periodic refit over buffered window |
| Acceleration | Difference between short and long slope estimates |

State should have a strict TTL:

```text
TTL = 35 days
```

This prevents unbounded memory growth for removed or inactive drives.

---

## 20.2 Hot / Cold Feature Storage

To keep Redis memory bounded:

| Drive Category | Storage |
|---|---|
| Healthy low-risk drives | Batch features or lightweight state |
| Elevated-risk drives | Redis hot state |
| Drives under action | Redis hot state + audit context |
| Removed drives | Evict after TTL |

---

## 20.3 Point-in-Time Feature Queries

When the LangGraph agent evaluates a drive, it should request:

```text
drive_id
as_of_timestamp
feature_version
```

The feature store returns only information available at that timestamp.

This ensures:

- no future-data leakage during inference;
- reproducible decision replay;
- auditable explanations;
- consistency between offline evaluation and online behavior.

---

# 21. Integration with the Reliability Checker

The dataset and feature strategy must support explainability and trust scoring.

For every high-risk prediction, the system should be able to answer:

1. Which features contributed most to the prediction?
2. Was the drive showing a rising slope or only a transient spike?
3. Were guardrails satisfied?
4. Was the prediction confirmed by later failure?
5. Was the action timely?
6. Was the action necessary?
7. Was the telemetry fresh enough to trust the decision?

Trajectory-based features make this easier because they are human-interpretable.

Examples of explainable statements:

```text
The 30-day slope of current pending sectors increased sharply.
The drive had 9 positive pending-sector days in the last 30 days.
Offline uncorrectable errors accelerated over the last 7 days.
Telemetry coverage was 30/30 days, so feature confidence was high.
```

These are more meaningful than opaque raw-value alerts.

---

# 22. Acceptance Criteria

The dataset and feature-engineering pipeline is considered complete when the following conditions are met.

## Data Availability

- [ ] Backblaze raw data ingested.
- [ ] SMART-Z raw data ingested or access requested.
- [ ] Synthetic chaos dataset generated.

## Data Quality

- [ ] Canonical schema validated.
- [ ] Missingness and telemetry gaps are tracked.
- [ ] Survivorship bias is handled.
- [ ] Duplicate and null checks pass.
- [ ] SMART badness orientation is standardized.

## Features

- [ ] 7/14/30-day aggregates implemented.
- [ ] Deltas, slopes, and acceleration implemented.
- [ ] Threshold-crossing and event-count features implemented.
- [ ] Missingness and staleness features implemented.
- [ ] Lifecycle metadata features implemented.
- [ ] Feature registry exists and is versioned.
- [ ] Golden-dataset regression tests pass.

## Labels

- [ ] Labels exist for 7/14/30-day horizons.
- [ ] Censoring logic implemented.
- [ ] Temporal splits created.
- [ ] Label leakage tests pass.

## Efficiency

- [ ] Pipeline runs without exceeding RAM budget.
- [ ] Data stored as Parquet.
- [ ] No full-dataset in-memory load required.

## Modeling Readiness

- [ ] Tree-model input table produced.
- [ ] Optional LSTM sequence tensors produced.
- [ ] Class imbalance strategy documented.
- [ ] AUPRC evaluation pipeline available.

## Operational Readiness

- [ ] Feature confidence is computed.
- [ ] Stale telemetry reduces feature confidence.
- [ ] Immature drives are blocked from destructive autonomous actions.
- [ ] Feature lookup latency supports the agent loop.
- [ ] Feature drift monitoring emits retrain signals.

---

# 23. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Memory exhaustion | Pipeline failure | Polars lazy scans, DuckDB, Parquet partitioning |
| Label leakage | Overoptimistic metrics | Strict temporal splits and censoring |
| Forward-fill hides failure | Missed failures | Add telemetry-gap and staleness features |
| SMART-Z schema mismatch | Poor generalization | Canonical mapping and normalized values |
| Class imbalance | Poor precision | AUPRC, class weighting, threshold tuning |
| Ambiguous replacements | Noisy labels | Separate failure, replacement, and censoring flags |
| Synthetic data unrealistic | Weak guardrail testing | Trajectory morphing from real failures |
| Feature drift | Degraded reliability | PSI/KS monitoring and retrain signals |
| Too many features | Noise and RAM pressure | Start with top 10–15 attributes |
| Vendor-specific SMART semantics | Misleading cross-vendor features | Use normalized values, badness orientation, and model-family z-scores |
| Stale telemetry triggers unsafe action | Unsafe autonomy | Feature confidence gating and maturity checks |
| Online/offline feature mismatch | Unreliable inference | Shared feature registry and point-in-time serving |

---

# 24. Summary

The dataset and feature-engineering strategy is central to the project’s success. The system cannot safely act autonomously unless its predictions are precise, explainable, and robust across vendors.

This strategy focuses on:

- transforming raw SMART telemetry into trajectory-based features;
- using strict temporal labeling and censoring;
- avoiding leakage through partitioned, time-aware feature computation;
- treating missing telemetry as informative rather than automatically harmless;
- using Polars, DuckDB, and Parquet to handle large datasets within RAM limits;
- maintaining a versioned feature registry and golden-dataset tests;
- supporting explainability for the Reliability Checker;
- enabling cross-vendor validation through SMART-Z;
- generating realistic synthetic chaos data for guardrail and agent testing;
- adding feature confidence and maturity gating to support safe autonomous actions.

This creates a dataset foundation capable of supporting the project’s core requirement:

> **High-precision, explainable disk-failure prediction that can safely drive autonomous self-healing actions.**
