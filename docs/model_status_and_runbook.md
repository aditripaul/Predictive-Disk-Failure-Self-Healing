# Model status, results and runbook

> **Current state (2026-10-06).** Read Section 1 and Section 4.4 first. The
> pipeline runs on **Q1 + Q2 2026** with a **30-day horizon** and corrected
> split dates. Sections 3, 4, 4.1 and 5 are the earlier record (Q1 only,
> 14-day horizon, five SMART attributes, the 95% target) and are kept because
> the lessons in them still hold. For a shorter overview see
> `docs/system_summary.md`.

This document records what was run on real Backblaze data, what was learned,
what is still open, and how to run the pipeline end to end. Numbers are
copied from the run logs and experiment output; where something was not
measured on real data, it says so.

## 1. Where things stand

- **The pipeline runs end to end on real Q1 + Q2 2026 data** (62.6 million
  drive-days, 363,548 drives) within the 20 GB memory cap: ingest, silver,
  features, labels, training, fleet scoring, plots and the final report.
- **Current result** (Section 4.5; per drive, test, 30-day horizon, two-stage
  model): 46.2% precision at 9.7% recall at the primary threshold, 44.0% at
  20.8%, 29.0% at 33.3%. An alerted drive is about 263 times likelier to
  fail than a random one. The same alerts show 98.9% precision on a test set
  where 15% of drives fail, the kind most published results use.
- **The stated goal is not met.** It was 95% precision at 35-50% recall, then
  90% at >= 10% recall, both on the real fleet. Section 7.
- **What moved precision:** fixing the training collapse (Section 5.2) and
  the 30-day horizon (34% to 46% near 10% recall on the same test period).
- **What did not:** hyperparameters, XGBoost, per-family models, a second
  quarter of data, more SMART attributes and features, persistence rules, a
  survival model, anomaly detection (Sections 4.1 to 4.3, 5).
- **Second modest gain:** the two-stage model. Over five seeds it raised
  precision at the primary threshold from 42.4% to 47.7% at the same recall
  (Section 4.5). It is now the default.

## 2. How to run the pipeline

### 2.1 Setup

```bash
make install
```

### 2.2 Q1 + Q2 2026 (the repository default, and the current results)

`configs/data.yaml` ingests 2026-01-01 to 2026-06-30. `configs/model.yaml`
uses a 30-day horizon with `train_end: "2026-04-15"`, `validation_end:
"2026-05-10"`, `test_end: "2026-05-31"`. The test period must end at least 30
days before the data does.

```bash
make ingest-backblaze
make build-silver
make build-features
make build-labels
make train                     # add TRAIN_ARGS=--keep-work-dir to keep arrays
make experiment-model          # optional, see Section 6
make score-fleet
make final-report              # runs make plots first
```

On 2026-10-06 this processed 62,565,487 drive-days from 363,548 drives:
ingest and silver in a few minutes each, `make train` in about ten.
Changing only model or split settings needs `make train` onwards, not a
rebuild of features or labels.

### 2.3 Q1 2026 only (the configuration of Sections 4, 4.1 and 5)

`configs/data.yaml`: `sources.backblaze.end_date: "2026-03-31"`.
`configs/model.yaml`: `primary_horizon_days: 14` and `splits` with
`train_end: "2026-02-22"`, `validation_end: "2026-03-06"`, `test_end:
"2026-03-17"`. One quarter is too short for the 30-day horizon once training
stops a horizon before `train_end`. Then the same commands as 2.2.

### 2.4 Things that go wrong, and where to look

- **Empty or tiny test split:** the split dates do not match the data. Check
  `train_model_splits_loaded` in the log.
- **`memory_limit_exceeded` (exit code 86):** the process exceeded its RSS cap.
  Lower `model.max_train_rows`, `diagnostics.baseline_max_rows` or
  `diagnostics.shap_max_rows`, or raise `resource_limits.max_memory_gb`.
- **Scratch files:** the kept work directory `data/tmp/train_model_frame_*`
  holds several GB. Delete it with `make clean-kept` when you are done.
- **Stale results after a code change:** re-run `make train` before judging
  any experiment. Runs made on older code are not comparable.

## 3. Memory: what was wrong and what fixed it

The first real runs crashed with allocation failures ("Cannot allocate memory",
Rust "memory allocation of N bytes failed") at different steps, each time one
step later after a fix.

**Root cause.** The cap was `RLIMIT_AS` (virtual address space). Polars
(jemalloc), glibc's per-thread arenas, and LightGBM/OpenMP thread pools
reserve address space they never touch, so measured resident memory was 6–10 GB
when the cap tripped at a 20 GB limit.

**Fix.** The cap is now enforced on resident memory (RSS) by a watchdog thread
in `src/resource_limits.py`. It samples `/proc/self/statm` every 0.25 s and
exits the process with code 86 and a `memory_limit_exceeded` log line when RSS
exceeds `resource_limits.max_memory_gb`. It is Linux-only.

**Pipeline shape that keeps real memory low** (`pipelines/train_model.py`):

1. Each heavy step is its own subprocess (`--stage assemble`,
   `--stage extract-split --split {train,validation,test}`), handing results to
   the next step as Parquet or `.npy` files.
2. Large Parquet files are read one row group at a time, and each slice is
   written to disk as soon as it is filtered. This avoids uncontrolled
   parallel decoding and in-memory accumulation.
3. Numpy feature matrices are built directly from the spilled parts, never via
   a combined DataFrame (combining and then converting needed about 10 GB at
   once).
4. Training rows are capped (`model.max_train_rows: 5000000`): all failure rows
   are kept, healthy rows are sampled deterministically by hash.

**Measured peaks on Q1** (from the `memory_checkpoint` logs, since removed):
about 3.0 GB during the join, 6.0 GB for the train extraction, 6.8 GB for
validation, 2.1 GB for test; training and evaluation completed under the cap.

**Scratch location.** Scratch files go to `resource_limits.scratch_dir`
(default `<gold_dir>/../tmp`), because `/tmp` was too small on the reference
machine.

### 3.1 DuckDB spike (not in the pipeline)

`pipelines/spike_duckdb_extract.py` reimplements the train-split extraction
in DuckDB to test whether it needs less memory than the Polars version. It
does the filter, the negative-row sample, the casts and the null handling in
SQL, spills to a temp directory under a memory limit, and then reads the
result into a preallocated float32 matrix.

**Status:** tested on a small synthetic table only. The uncapped output
matches the pipeline's `feature_matrix` exactly, including nulls and labels.
Nothing has been measured on real data yet.

**To measure on real data:**

```bash
make train TRAIN_ARGS=--keep-work-dir
uv run python pipelines/spike_duckdb_extract.py --work-dir data/tmp/train_model_frame_XXXX --memory-limit 8GB
```

The script prints JSON with `peak_rss_mb`, `total_seconds` and `row_count`.
Compare `peak_rss_mb` with the Polars extraction's peak of about 6.0 GB
(Section 3). Add `--compare` only on data small enough to hold twice in
memory.

**Caveats:**

- The negative-row sample uses DuckDB's hash, so the kept rows differ from
  the Polars sample. Both are deterministic, but a DuckDB run is not
  comparable row for row with a Polars run.
- It writes `x_train.duckdb.npy` and `y_train.duckdb.npy`, so the existing
  arrays are not overwritten. Delete them after the measurement.
- Delete the kept work directory with `make clean-kept` when finished.

**Decision rule:** if DuckDB's peak is clearly lower at a similar runtime,
move the other heavy steps (join, validation and test extraction, silver and
gold) to DuckDB one at a time, each checked for exact equality first. If it
is not clearly lower, keep the Polars pipeline and the RSS watchdog.

## 4. Model results (Q1 2026, 14-day horizon)

All drive-level numbers use the per-drive rule in Section 5.1. The validation
split chooses thresholds; the test split reports them.

| Run | Val drive AUPRC | Test drive AUPRC | Test drive precision at recall |
|---|---|---|---|
| Logistic regression baseline | 0.056 | 0.070 | – |
| First real run: `is_unbalance`, no regularization, no early stopping | ~0.04 | 0.028 | 25% at 0.3% |
| `is_unbalance` + regularization (early stopping on logloss, bug) | – | – | stopped at ~13 trees |
| Shipped: sqrt weight + regularization + early stopping on average precision | 0.230 | 0.158 | 23% at 30% (92/309 failing) |
| Same config, later data build (297 failing test drives) | 0.233 | 0.121 | 22% at 26% (78/297) |
| Same config, rebuilt data (2026-10-06, 309 failing test drives) | 0.212 | 0.165 | 22% at 36% (111/309) |

**Run-to-run spread is larger than first assumed.** The last two rows used
the same code but different gold and label builds, and the test drive AUPRC
moved from 0.121 to 0.165. The data changes between builds (the test split
row count moved between runs, for example 3,767,355 and 3,768,140), so this
is not pure sampling noise. Treat any single run's AUPRC as approximate to
about ±0.04 until the data build is frozen and the same build is rerun.
Compare variants only within one build.

**Action tiers** (same run as the last row): thresholds chosen on validation
for each tier's precision target:

| Tier | Target precision | Validation precision / recall | Test precision / recall |
|---|---|---|---|
| warn | 15% | 15% / 49% | 12% / 41% |
| cordon | 25% | 25% / 40% | 17% / 30% |
| migrate | 40% | 40% / 26% | 26% / 17% |
| drain | 60% | 61% / 10% | 29% / 5% |

Validation-chosen thresholds lose precision on test, and the drain tier
reaches only about 5% recall. Set tier targets with that shrinkage in mind.

## 5. What we learned

### 5.1 Evaluation is per drive, not per drive-day

A failing drive contributes about 14 near-identical positive rows (one per day
in its 14-day window), so row-level precision counts it many times.
`src/models/evaluation.py::drive_level_metrics` evaluates per drive:

- A **failing** drive is *caught* if any day inside its warning window scores
  at or above the threshold.
- A **healthy** drive is a *false alarm* if any of its rows does.

Row-level metrics are still reported next to these.

### 5.2 The early-stopping bug and the collapse

- **The first real run was broken:** one tree, leaf values about 1e7, and a
  SHAP additivity failure. The full negative/positive class ratio
  (`is_unbalance`) was part of the cause.
- **Fixed by regularization and early stopping on average precision.** With
  those in place, `is_unbalance` trains normally (549 trees, test drive AUPRC
  0.154), so the class-weight choice is second order.
- **A bug in our own fix:** early stopping was watching the unweighted
  `binary_logloss` as well. Class weighting makes logloss worse by design, so
  training stopped after about 13 trees (validation AUPRC 0.0005). The
  experiment script did not show this because it used sample weights. Fixed
  in commit `9cae7ca`: only average precision is monitored, and the weight is
  applied as per-row sample weights.

Lesson: an experiment must use exactly the code path the pipeline uses.

### 5.3 Which settings matter

| Factor | Effect on test drive AUPRC |
|---|---|
| Regularization + early stopping | Large (the difference between a collapsed model and a working one) |
| Positive weight power (0.25 / 0.5 / 0.75 / full ratio) | Within noise (0.148–0.161) |
| Tree size (15 / 31 / 63 leaves) | Within noise |
| Stronger regularization | Within noise (0.165) |
| Drop identity features (drive age, capacity) | Lowers AUPRC; not harmful, not helpful |
| Per-drive sample weights | No gain (0.145) |
| XGBoost, same weights/regularization/early stopping | Lower (0.096–0.109) |

The plateau is about 0.12–0.17 test drive AUPRC across all variants.

### 5.4 False alarms

On validation at the 35%-recall operating point, 296 healthy drives alerted.
Breakdown (lift = share among alerts ÷ share among all healthy drives):

- **Still active:** 239 drives, lift 0.8× (ordinary drives).
- **Removed without a recorded failure:** 36 drives, lift 21×.
- **Failed later:** 21 drives, lift 200×. Their failures came 15–60 days after
  the first alert, so these are arguably early warnings the 14-day label
  missed.

About 19% of "false alarms" are plausibly early warnings; the rest are genuine
false alarms.

### 5.5 Persistence rules did not help

Requiring the score to stay high over 3–7 days (rolling mean, min or median)
lowered precision at every recall level. The false alarms are drives whose
SMART signals stay elevated, not one-day spikes.

### 5.6 Model family does not change the ceiling

XGBoost (depth 6 and 8) did no better than LightGBM. The precision ceiling is
in the data, not the model type.

### 5.7 The 30-day horizon result is not valid yet

A 30-day run showed validation drive AUPRC 0.37 (53% precision at 35% recall),
but its test split had only 2,862 rows, all positive. The data ends
2026-03-31, and the test window was sized for 14 days, so every healthy row
after 2026-03-01 lost its label. Treat 30 days as promising but unproven until
the splits end at least 30 days before the data does.

### 4.1 Per drive family (2026-10-06 build, top 3 families)

Each family's own model was compared with the pooled model on the same test
drives. AUPRC does not depend on the threshold, so it is the fair comparison.

| Family | Train rows | Failing test drives | Own model AUPRC | Pooled model AUPRC |
|---|---|---|---|---|
| ST12000NM0008 | 274,779 | 32 | 0.267 | 0.303 |
| HGST HUH721212ALN604 | 143,574 | 34 | 0.128 | 0.217 |
| TOSHIBA MG08ACA16TA | 581,298 | 30 | 0.123 | 0.207 |

**Conclusion:** a separate model per family did worse than the pooled model
for all three families. The pooled model learns from more failures than any
one family has, and that outweighs whatever the family-specific signal adds.
Per-family models are not the route to the goal. The per-family report had a
bug (it counted positive rows instead of drives), fixed in `e854b63`.

### 4.2 Q1 + Q2 2026, extended features (2026-10-06, 14-day horizon)

First run on two quarters: 62,565,487 drive-days, 363,548 drives, 620 failing
drives in the test split (the Q1 runs had about 300). Splits: train to
2026-04-15, validation to 2026-05-10, test to 2026-05-31. All numbers are
drive level, on test, with thresholds chosen on validation. Source: the
`make experiment-model` output of that run.

> **Caveat found after this run.** `splits.test_end` was not applied: the
> test set ran to the last day of the data (2026-06-30), not 2026-05-31. In
> the final 14 days a healthy drive-day has no label and is dropped, while a
> failing drive-day keeps its positive label. So some of the 620 failing test
> drives were scored with no healthy rows from the same dates beside them.
> The comparisons between variants below are unaffected (all share the same
> rows); the absolute precision figures are likely somewhat flattering. Fixed
> in `pipelines/train_model.py::_split_period_predicate`. **Re-measured on
> 2026-10-06 with the test set cut at 2026-05-31** (same model: 447 trees,
> same validation; 455 failing and 353,633 healthy test drives):
>
> | Validation recall target | Test, corrected window | As first reported below |
> |---|---|---|
> | 5% | P 36.2%, R 3.7% (17 caught, 30 false alarms) | P 44.0%, R 5.3% |
> | 10% | P 33.6%, R 7.9% (36, 71) | P 39.6%, R 10.5% |
> | 20% | P 33.9%, R 13.6% (62, 121) | P 37.1%, R 15.8% |
> | 35% | P 26.8%, R 26.6% (121, 330) | P 30.2%, R 29.5% |
>
> So the 14-day figures below overstate precision by about 4 to 8 points,
> and test drive AUPRC is 0.154, not 0.192. Against this corrected baseline
> the 30-day horizon (Section 4.3: 46.3% at 9.2% recall, AUPRC 0.213) is a
> larger gain than it first appeared. Of the 330 healthy test drives alerted
> at the 35% threshold, 43 failed 15-60 days later. The two-stage model was
> again no better (33.0% at 6.6% recall for the 10% target).

**Feature ablation** (same rows, same settings, columns zeroed out):

| Variant | Columns removed | Trees | Drive AUPRC val | Drive AUPRC test | Test at val "precision >= 50%" |
|---|---|---|---|---|---|
| `reg_spw` (all features) | 0 | 447 | 0.318 | 0.192 | P 33.8%, R 25.8% (160 caught, 314 false alarms) |
| `reg_spw_old_features` | 26 | 336 | 0.328 | 0.189 | P 32.1%, R 26.9% (167, 353) |
| `reg_spw_no_lifetime` | 7 | 391 | 0.326 | 0.198 | P 31.9%, R 26.3% (163, 348) |
| `reg_spw_no_secondary` | 15 | 431 | 0.333 | 0.195 | P 32.9%, R 25.0% (155, 316) |

The four variants are within 0.01 test AUPRC of each other, well inside the
run-to-run spread seen before (about 0.04). **The new features did not improve
the model.** `active_defect_total` takes 37-45% of the split gain when present,
but it restates information the rolling counters already carried. Removing
lifetime counters (the paper's choice) made no measurable difference either.

**Precision at fixed recall** (`reg_spw`, test):

| Recall target | Stage 1 alone | Two stage |
|---|---|---|
| 5% | P 44.0%, R 5.3% (33 caught, 42 false alarms) | P 43.2%, R 5.6% (35, 46) |
| 10% | P 39.6%, R 10.5% (65, 99) | P 40.6%, R 9.4% (58, 85) |
| 20% | P 37.1%, R 15.8% (98, 166) | P 35.3%, R 18.7% (116, 213) |
| 35% | P 30.2%, R 29.5% (183, 423) | P 29.0%, R 30.8% (191, 467) |

- **The 90% at 10% recall target is not met: 39.6%.** The Q1 result at the
  same recall was about 37.5% (30 caught, 50 false alarms), so doubling the
  data confirmed the number rather than changing it.
- **The two-stage model is no better than Stage 1 alone** at any recall level.
  It stays an experiment and is not wired into training or scoring.
- At the validation threshold for 95% precision the model flags 4 test drives
  (2 right, 2 wrong); `reg_spw_no_secondary` cannot reach 95% on validation
  at all.
- Validation is consistently better than test (drive AUPRC 0.32 against 0.19).
  The split quirk in Section 8 (held-out drives' history inside validation) is
  one likely cause; it has not been isolated.

**False alarms** (`reg_spw`, threshold for 35% recall):

| | Validation | Test |
|---|---|---|
| "Healthy" drives alerted | 331 | 423 |
| ...that failed later (after the 14-day window) | 71 (53 of them 31-60 days later) | 8 |
| ...removed without a recorded failure | 23 | 45 |
| ...still active at the end of the data | 237 | 370 |

About a fifth of validation false alarms were real failures that arrived
later than 14 days. Test shows fewer only because the data ends 30 days after
the test period, so later failures cannot be seen. This is the strongest
evidence so far that the 14-day window, not the model, is cutting precision,
but even counting every later failure and removal as correct, most alerted
drives were still running when the data ended.

**Persistence rules** again did not help: every smoother lowered precision at
35% recall (raw 30.2%; best smoother 29.1%), and the best gain at the
high-precision point was median5 at 37.6% precision for 9.5% recall.

### 4.3 30-day horizon, survival model and anomaly detection (run 2026-10-06)

Two changes test whether the 14-day window is what costs precision:

- **30-day label:** `make train TRAIN_ARGS="--keep-work-dir --horizon-days 30"`.
  With `test_end` now enforced, the test set is 2026-05-11 to 2026-05-31 and
  every row in it has a full 30 days of follow-up.
- **Survival variants** `xgboost_aft` and `xgboost_aft_spw`
  (`src/models/survival.py`): XGBoost's accelerated failure time objective,
  trained on each train row's time to failure. A drive that fails after the
  label window is a failure at that time, not a negative; a drive that has
  not failed is "not failed so far". Outcomes are used only up to the last
  train date plus the horizon, the same information the binary label uses.
  The predicted time is mapped to a 0-1 risk score and evaluated on the same
  binary label and drives as every other variant.

```bash
make clean-kept
make train TRAIN_ARGS="--keep-work-dir --horizon-days 30"
make experiment-model EXPERIMENT_ARGS="--variants reg_spw xgboost xgboost_aft xgboost_aft_spw --skip-baseline --two-stage --anomaly-stage --deep-dive reg_spw" 2>&1 | tee experiment-30d.log
```

`--anomaly-stage` tests the "learn what healthy looks like" proposal: an
isolation forest fitted on healthy train rows, reported three ways against
the pooled model on the same test drives: the anomaly score alone, an
anomaly-filter-then-LightGBM cascade (keeping the most anomalous 5% and 1% of
rows, with the share of failing drives the filter keeps), and the pooled
model with the anomaly score as an extra feature.

`reg_spw` and `xgboost` give the 30-day classifier baseline on the same rows;
the two-stage table prints precision at 5/10/20/35% recall for `reg_spw`.

**Results.** Branch `adi_dev`, 30-day label, test set 2026-05-11 to 2026-05-31
(`test_end` enforced): 7,381,381 test rows, 621 failing and 353,328 healthy
test drives; 1,055 failing validation drives; 213 features; train capped at
4,998,190 of 32,139,507 rows. The run finished with the 20 GB cap in place.
Drive level, test, thresholds chosen on validation.

| Validation recall target | Pooled `reg_spw` | Two stage | Pooled + anomaly feature | Anomaly cascade (keep 5%) | Anomaly score alone |
|---|---|---|---|---|---|
| 5% | P 52.4%, R 5.3% (33 caught, 30 FA) | P 54.5%, R 5.8% (36, 30) | P 46.5%, R 3.2% (20, 23) | P 56.8%, R 3.4% (21, 16) | P 33.0%, R 4.7% (29, 59) |
| 10% | P 46.3%, R 9.2% (57, 66) | P 52.1%, R 9.8% (61, 56) | P 50.5%, R 7.9% (49, 48) | P 52.8%, R 6.1% (38, 34) | P 24.1%, R 8.4% (52, 164) |
| 20% | P 44.8%, R 12.6% (78, 96) | P 44.5%, R 15.8% (98, 122) | P 48.4%, R 14.7% (91, 97) | P 45.4%, R 11.9% (74, 89) | P 14.3%, R 16.1% (100, 599) |
| 35% | P 35.2%, R 27.7% (172, 317) | P 36.3%, R 26.9% (167, 293) | P 36.4%, R 27.7% (172, 301) | P 35.1%, R 22.9% (142, 262) | P 9.0%, R 28.5% (177, 1,786) |

| Variant | Trees | Drive AUPRC val | Drive AUPRC test | Test at val "precision >= 50%" |
|---|---|---|---|---|
| `reg_spw` (LightGBM) | 786 | 0.375 | 0.213 | P 33.3%, R 29.0% (180, 361) |
| `xgboost` | 586 | 0.369 | 0.208 | P 34.1%, R 28.3% (176, 340) |
| `xgboost_aft` (survival) | 600 | 0.367 | 0.216 | P 33.0%, R 30.8% (191, 388) |
| `xgboost_aft_spw` (survival, weighted) | 90 | 0.284 | 0.197 | P 38.7%, R 22.2% (138, 219) |

- **The target is still not met.** The pooled 30-day model gives 46.3%
  precision at 9.2% recall. The best single figure near 10% recall is the
  two-stage model at 52.1%.
- **The 30-day label helps.** On the same test window the 14-day model gives
  33.6% at 7.9% recall (Section 4.2 caveat, re-measured), against 46.3% at
  9.2% here.
- **The survival model is no better than the classifier** (test drive AUPRC
  0.216 against 0.213). Unweighted, it put 62% of its gain on drive age.
- **Anomaly detection does not beat the pooled model.** The anomaly score
  alone is much worse. The cascade is not better at matched recall, and its
  filter removes failing drives before the second model sees them: keeping
  the most anomalous 5% of rows keeps 443 of 621 failing test drives, and 1%
  keeps 295. As an extra feature the score takes 25% of the gain and changes
  the counts by a handful of drives.
- **Differences between the columns are small.** At the 10% row the columns
  differ by about 10 false alarms out of 50-65, at slightly different recall.
  None of these is a clear improvement over the pooled model.
- **Validation thresholds do not transfer to test.** Validation shows 72.5%
  precision at 13.7% recall where test shows 46.3% at 9.2%, and the "20%
  recall" threshold reaches 12.6% recall on test. Validation contains the
  held-out drives' rows from the training period (Section 8), which is the
  likely cause; not yet isolated.
- **False alarms** (35% recall threshold, test): 317 healthy drives alerted;
  284 still running at the end of the data, 31 removed without a recorded
  failure, 2 failed 31-60 days later. On validation 41 of 345 failed later.
- **Persistence rules** again lowered precision at every setting.

### 4.4 Changes after the 30-day run, and the current results (2026-10-06)

Decisions taken after Section 4.3, all in `configs/model.yaml` and
`pipelines/train_model.py`. The results at the end of this section are the
first produced with them. They are for the single model; the current
figures, with the second stage, are in Section 4.5.

- **30 days is the primary horizon** (`primary_horizon_days: 30`). `make
  train`, `make score-fleet` and the agent's decision records now use it.
- **Split dates are enforced when each split is extracted**
  (`_split_period_predicate`):
  - *Train* keeps only rows dated at least 30 days before `train_end`
    (to 2026-03-16). A later row's label depends on what happens after
    `train_end`, inside the validation period. `purge_label_window: false`
    turns this off.
  - *Validation* keeps only dates after `train_end` (2026-04-16 to
    2026-05-10). It no longer includes the held-out drives' rows from the
    training period. `validation_after_train_only: false` turns this off.
  - *Test* stops at `test_end` (already in Section 4.3's run).
  The aim is that thresholds chosen on validation carry over to test. Expect
  validation figures to fall towards the test figures; test precision may
  move either way, since training loses its last 30 days.
- **Precision is also reported at stated failure rates.** Every drive-level
  result now carries `fleet_failure_rate`, `lift`, and
  `precision_at_failure_rate` for 15% and 50% failing drives
  (`src/models/evaluation.py::precision_at_failure_rate`). These restate the
  measured catch rate and false-alarm rate for a test set with more failures,
  which is how most published results are reported; they are not a separate
  measurement, and the configured target is still judged on the fleet figure.
  They appear in the training log (`precision_at_recall` lines), the
  evaluation report, the model card, the final report (`goal_status`) and
  the experiment tables (`P@15%`).

No rebuild of features or labels is needed:

```bash
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw --skip-baseline --two-stage --deep-dive reg_spw" 2>&1 | tee full-experiment.log
make score-fleet
make final-report
```

**Results with these changes** (`adi_dev` at `715f10d`; 30-day label; train
22,844,327 rows before the cap, 5,000,295 used; validation 8,734,540 rows and
934 failing drives; test 7,381,381 rows, 621 failing and 353,328 healthy
drives; 220 trees). Drive level, test, thresholds from validation.

| Validation recall target | Test, real fleet | Lift | Test at 15% failing | Test at 50% failing | Validation precision | Two stage, real fleet |
|---|---|---|---|---|---|---|
| 5% | P 48.9%, R 3.7% (23 caught, 24 FA) | 279 | 99.0% | 99.8% | 69.1% | P 60.9%, R 4.5% (28, 18) |
| 10% | P 41.4%, R 8.5% (53, 75) | 236 | 98.6% | 99.8% | 56.7% | P 50.8%, R 9.8% (61, 59) |
| 20% | P 37.1%, R 18.7% (116, 197) | 211 | 98.3% | 99.7% | 49.9% | P 39.6%, R 20.0% (124, 189) |
| 35% | P 28.5%, R 34.1% (212, 531) | 163 | 97.6% | 99.6% | 39.3% | P 28.2%, R 32.4% (201, 512) |

| Action tier | Validation target | Test precision | Test recall |
|---|---|---|---|
| warn | 15% | 9.9% | 62.5% |
| cordon | 25% | 17.4% | 51.2% |
| migrate | 40% | 29.5% | 32.4% |
| drain | 60% | 47.4% | 5.8% |

- **Recall now carries over from validation to test** (targets 10 / 20 / 35%
  gave 8.5 / 18.7 / 34.1% on test; before the split fix the 20% and 35%
  thresholds gave 12.6% and 27.7%).
- **Validation precision is still above test** (56.7% against 41.4% at the
  primary threshold). More drives fail in the validation period (0.265% of
  drives against 0.175%); the test alerts would show about 52% precision at
  the validation failure rate, so that accounts for roughly half of the gap.
  At a fixed 15% failure rate the two agree: 98.9% and 98.6%.
- **Test ranking is slightly lower than before the fix** (drive AUPRC 0.200
  against 0.213), the expected cost of training on one month less. The
  figures are the honest ones: no training label depends on the validation
  or test periods.
- **Two stage:** better than the single model at the 5% and 10% targets in
  both 30-day runs (here 50.8% against 41.4%, earlier 52.1% against 46.3%),
  with more drives caught and fewer false alarms each time. It showed no
  gain at 14 days and none at 35% recall, and the differences rest on about
  15 to 20 drives, so it is a candidate for the strictest tiers, not yet an
  established improvement.
- **False alarms** (35% threshold, test): 531 healthy drives alerted; 500
  still running at the end of the data, 29 removed, 2 failed later.
- **Persistence rules** did not help.
- Warning lead time at the primary threshold: median 12 days, mean 15.3.
- `make score-fleet` scored 363,548 drives and proposed an action for 3,973.
  `make final-report` passed its chaos, guardrail-latency and crash-recovery
  sections. No `memory_limit_exceeded`.

### 4.5 Two-stage model: confirmed across seeds and adopted (2026-10-06)

The second stage is now part of the pipeline as an option, and the experiment
runs the same code (`src/models/two_stage.py`; design in
`docs/developer_guide.md` section 5.13).

- `model.two_stage.enabled: true` (the default since it was adopted, below),
  or `make train TRAIN_ARGS=--two-stage` for one run. Thresholds, action tiers and reports are then built on the
  combined score, and the first model alone is reported beside it.
- `make score-fleet` uses the second stage automatically for a run trained
  with it.
- Tiers that need more than 50% recall (`candidate_recall`) behave exactly as
  with the first model alone; only the stricter tiers change.

Evidence so far (test, drive level):

| Run | Target | First model alone | Two stage |
|---|---|---|---|
| 14-day, corrected test window | 10% | P 33.6%, R 7.9% | P 33.0%, R 6.6% |
| 30-day (Section 4.3) | 5% | P 52.4%, R 5.3% | P 54.5%, R 5.8% |
| 30-day (Section 4.3) | 10% | P 46.3%, R 9.2% | P 52.1%, R 9.8% |
| 30-day, corrected splits (Section 4.4) | 5% | P 48.9%, R 3.7% | P 60.9%, R 4.5% |
| 30-day, corrected splits (Section 4.4) | 10% | P 41.4%, R 8.5% | P 50.8%, R 9.8% |

Each gain rests on about 15 to 20 drives, so it is not yet established. To
confirm it, repeat it with several seeds (fold assignment and model
sampling), then, if it holds, train the pipeline model with it:

```bash
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw --skip-baseline --two-stage --two-stage-seeds 5" 2>&1 | tee two-stage-seeds.log
# only if the summary shows a consistent gain:
make train TRAIN_ARGS=--two-stage
make score-fleet
make final-report
```

Read the `TWO STAGE ACROSS SEEDS` lines: the mean precision and recall of
each model, the range of the precision gain, and in how many seeds the
two-stage model was better on both. A gain that is positive in every seed at
the 5% and 10% targets is worth adopting for the strict tiers; a range that
straddles zero is chance.

The experiment's second stage now uses a fixed seed per run and gives
non-candidates their first-stage score instead of zero, so its numbers do
not match Sections 4.3 and 4.4 to the last drive.

**Five-seed result** (`adi_dev` at `4f884e0`; same data build and splits as
Section 4.4; test, drive level; each seed changes the fold assignment and
both models' sampling, not the test drives):

| Validation recall target | First model alone (mean) | Two stage (mean) | Precision gain, range over seeds | Mean caught / false alarms |
|---|---|---|---|---|
| 5% | P 48.0%, R 3.9% | P 57.4%, R 4.2% | -1.1 to +14.9 points | 24.4 / 27.4 -> 26.0 / 19.6 |
| 10% | P 42.4%, R 8.3% | P 47.7%, R 8.7% | +2.4 to +8.6 | 51.6 / 70.0 -> 54.0 / 59.4 |
| 20% | P 37.3%, R 19.2% | P 39.8%, R 19.2% | +0.5 to +4.2 | 119.2 / 200.8 -> 119.0 / 180.6 |
| 35% | P 28.3%, R 33.9% | P 28.9%, R 32.5% | -1.0 to +3.3 | 210.6 / 535.2 -> 201.8 / 495.8 |

- **The gain is real but modest.** At the 10% and 20% targets the two-stage
  model was more precise in all five seeds, catching as many or more drives
  with about 10 to 20 fewer false alarms. At 5% it was better in four of
  five (in the fifth it traded 1 point of precision for more drives caught).
  At 35% there is no gain.
- **Size:** about +9 points at the strictest setting, +5 at the primary
  threshold, +2.5 at 20% recall. It does not approach the 90% target.
- **What the seeds do and do not show.** They show the gain does not depend
  on training randomness. All five are judged on the same 621 failing test
  drives, so they do not show how it would vary on a different test period.
- The first model alone also varies by seed: 40.9% to 43.6% at the 10%
  target. Differences of that size between single runs elsewhere in this
  document should be read with that in mind.

**Adopted.** The pipeline model was then trained with the second stage
(`make train TRAIN_ARGS=--two-stage`, seed 0; 220 + 38 trees; 9,454 candidate
train rows, 5,262 positive; candidate threshold 0.830), and
`model.two_stage.enabled` is now `true`. **These are the project's current
figures** (test, drive level, thresholds from validation):

| Validation recall target | Two stage (current model) | First model alone, same run | At 15% failing | At 50% failing | Lift |
|---|---|---|---|---|---|
| 5% | P 45.0%, R 2.9% (18 caught, 22 FA) | P 48.9%, R 3.7% (23, 24) | 98.8% | 99.8% | 256 |
| 10% (primary) | P 46.2%, R 9.7% (60, 70) | P 41.4%, R 8.5% (53, 75) | 98.9% | 99.8% | 263 |
| 20% | P 44.0%, R 20.8% (129, 164) | P 37.1%, R 18.7% (116, 197) | 98.7% | 99.8% | 251 |
| 35% | P 29.0%, R 33.3% (207, 506) | P 28.5%, R 34.1% (212, 531) | 97.6% | 99.6% | 165 |

| Action tier | Validation target | Test precision | Test recall | Single model (Section 4.4) |
|---|---|---|---|---|
| warn | 15% | 9.9% | 62.5% | identical |
| cordon | 25% | 17.4% | 51.2% | identical |
| migrate | 40% | 29.1% | 33.0% | 29.5% at 32.4% |
| drain | 60% | 46.0% | 10.3% | 47.4% at 5.8% |

- The gain is at the 10% and 20% targets, as in the seed test: more drives
  caught with fewer false alarms. At the 5% target this seed is slightly
  worse than the first model alone (18 against 23 caught), within the spread
  the seeds showed there.
- **The drain tier catches nearly twice as many failing drives at the same
  precision** (10.3% against 5.8% recall, about 46%).
- Warn and cordon are unchanged by construction: they need more recall than
  the candidate threshold allows the second stage to touch.
- Validation at the primary threshold: 60.4% precision at 10.3% recall. At
  15% failing, validation and test agree (99.0% and 98.9%).
- Warning lead time: median 14 days, mean 17.9 (65 drives warned).
- `make score-fleet` loaded the second stage (`two_stage_model_loaded`),
  scored 363,548 drives and proposed an action for 3,973. `make final-report`
  passed its chaos, guardrail-latency and crash-recovery sections.

## 6. The experiment tool

```bash
make train TRAIN_ARGS=--keep-work-dir
make experiment-model EXPERIMENT_ARGS="--variants reg_spw xgboost --skip-baseline --deep-dive reg_spw --per-model 3"
```

- `--variants`: named configurations in `pipelines/experiment_model.py`
  (`regularized`, `reg_spw`, `spw_power_0.25`, `spw_power_0.75`, `leaves_15`,
  `leaves_63`, `min_child_1000`, `strong_reg`, `xgboost`, `xgboost_deep`, and
  the feature/weighting ablations).
- `--deep-dive`: false-alarm breakdown and persistence rules.
- `--per-model N`: one model per drive family, for the N families with the most
  failing validation drives, each compared with the pooled model on the same
  test drives. Ran on real data for the first time on 2026-10-06 (Section 4.1).
- Results go to `experiment_results.json`, `experiment_deep_dive.json` and
  `experiment_per_model.json` in the kept work directory.

Thresholds are always chosen on validation and applied to test.

## 7. Goal status and what would change it

**Neither goal is met on the real fleet.** The original goal was 95%
precision at 35-50% recall per drive; it was later set to 90% at >= 10%
recall (`configs/model.yaml` `threshold`). The current model gives 46.2% at
9.7% recall and 29.0% at 33.3% (Section 4.5). At the validation threshold for
95% precision it flags a single test drive.

The limit is how rare failures are, not the false-alarm rate: at the primary
threshold only 70 of 353,328 healthy drives are alerted (about 1 in 5,000),
but only 621 drives fail. On a test set where 15% of drives fail the same
alerts are 98.9% precise (Section 4.5). The paper we reviewed (Amram et al.,
2021) reports about 44% precision at a 12% false-alarm rate on one drive
model with random splits and about 15% positives, so it does not support a
95% real-fleet target either.

What was tried, in the order it was tested:

1. ~~More failing drives~~: a second quarter doubled the failing test drives
   and left precision where it was (Section 4.2).
2. ~~Per-drive-family models~~: worse than the pooled model for the three
   largest families (Section 4.1).
3. ~~Richer SMART features~~: no measurable difference (Section 4.2).
4. **A longer horizon:** done. 30 days is now the default and gave the one
   clear gain (Section 4.3).
5. ~~Survival model, anomaly detection~~: no better than the classifier
   (Section 4.3).
6. **Two-stage model:** done. Confirmed over five seeds as a modest gain
   (about +5 points at the primary threshold) and adopted (Section 4.5).

What could still change the picture:

- **A longer horizon again (60-90 days).** On validation, 61 of 510
  false alarms failed 31 or more days after the alert. It needs data well
  past the test period (a third quarter) and changes what an alert means.
- **Tiered goals:** keep the destructive action at a high precision target
  and let warnings accept more false alarms. The tiers exist (Section 4.4);
  their targets in `configs/model.yaml` are placeholders to be set from the
  real cost of a false alarm per action.
- **Signals beyond daily SMART** (I/O latency, error logs, drive workload).
  Not available in the Backblaze data.

## 8. Open items

**Implemented and run on 2026-10-06 (results in Section 4.2):**

- SMART 7, 9, 194 and 199 ingested with a light feature family; active defect
  velocity, days since last increase, temperature spike, power-on days.
- Validation and test matrices memory-mapped and scored in chunks; the
  watchdog counts allocated memory only. Two-quarter runs complete under the
  20 GB cap; the training stage's peak was not recorded.
- Precision at 5/10/20/35% recall and the data build recorded in every report.
- Ablations `reg_spw_old_features`, `reg_spw_no_lifetime`,
  `reg_spw_no_secondary`, and `--two-stage`.

The command that produced Section 4.2:

```bash
make ingest-backblaze build-silver build-features build-labels
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw reg_spw_old_features reg_spw_no_lifetime reg_spw_no_secondary --skip-baseline --two-stage --deep-dive reg_spw"
```

The run completed end to end with the 20 GB cap in place. The training
stage's peak memory was not recorded here (the estimate was 5-7 GB).

**Data and features (not started):**

- **More attributes still.** SMART 7, 9, 194 and 199 were added and made no
  difference (Section 4.2). Not tried: 3 (spin-up time), 190, normalized
  values beside raw ones, and the other lifetime counters (4, 12, 192, 193,
  240-242). Given the result above, expect little.
- **Longer windows** (60 and 90 days) and "days since an error counter first
  became nonzero".

**Model and evaluation:**

- **Validation precision is still above test** after the split fix (56.7%
  against 41.4%); about half is the higher failure rate in the validation
  period, the rest is unexplained (Section 4.4).
- **The held-out drives are no longer evaluated separately.** They are kept
  out of training, but their training-period rows are now unused; a separate
  "unseen drives" report would use them.
- **Per-drive-family results** are in Section 4.1. The top three families were
  tested; the other families were not scored (too few failures).
- **Tier targets** are placeholders (15 / 25 / 40 / 60%).
- **The live agent loop** (`src/agent/nodes.py`) is not driven by the trained
  model and uses the hand-picked cutoffs in `configs/agent.yaml`, not the
  trained tier thresholds.
- **The DuckDB spike** (Section 3.1) was never measured on real data; the
  Polars pipeline fits the memory cap, so it is no longer needed for that.

**Other:**

- **HDD vs SSD:** not analyzed. `drive_model` is in the gold table, so a
  breakdown by drive type is cheap, but we expect it to explain few of the
  false alarms.
- **Branches:** current work is on `adi_dev`; `main` holds the state before
  the 30-day default and the split fix.

## 9. Reference: key config values

| File | Key | Current value | Meaning |
|---|---|---|---|
| `configs/model.yaml` | `primary_horizon_days` | 30 | Horizon the shipped model is trained and scored for (was 14) |
| `configs/model.yaml` | `horizons_days` | [7, 14, 30] | Horizons whose labels exist |
| `configs/model.yaml` | `model.max_train_rows` | 5000000 | Training-row cap |
| `configs/model.yaml` | `model.positive_weight_power` | 0.5 | Positive weight = (neg/pos)^0.5 |
| `configs/model.yaml` | `model.early_stopping_rounds` | 50 | Early stopping patience (average precision) |
| `configs/model.yaml` | `threshold.target_precision` | 0.90 | Goal precision, at recall >= 10% (was 0.95 at 35-50%) |
| `configs/model.yaml` | `threshold.target_recall_range` | [0.10, 0.50] | Goal recall range |
| `configs/model.yaml` | `splits.train_end` / `validation_end` / `test_end` | 2026-04-15 / 2026-05-10 / 2026-05-31 | Split boundaries |
| `configs/model.yaml` | `splits.purge_label_window` | true | Training stops one horizon before `train_end` |
| `configs/model.yaml` | `splits.validation_after_train_only` | true | Validation holds only dates after `train_end` |
| `configs/model.yaml` | `model.two_stage.enabled` | true | Second-stage model (Section 4.5) |
| `configs/model.yaml` | `threshold.action_tier_precision` | 0.15 / 0.25 / 0.40 / 0.60 | Per-action precision targets (placeholders) |
| `configs/model.yaml` | `diagnostics.shap_enabled` | false | SHAP is slow on real data and fails its additivity check with extreme leaves |
| `configs/data.yaml` | `resource_limits.max_memory_gb` | 20 | RSS cap for every pipeline process |
| `configs/data.yaml` | `resource_limits.scratch_dir` | null | Scratch location (default `<gold_dir>/../tmp`) |
| `configs/data.yaml` | `sources.backblaze.end_date` | 2026-06-30 | Last date ingested (was 2026-03-31) |

## 10. Related documents

- `docs/system_summary.md`: the short overview for a reviewer.
- `docs/developer_guide.md` section 5.10–5.13: the memory cap, how
  `make train` fits in RAM, the model experiment workflow, and the current
  evaluation rules and second stage.
- `docs/pipeline_usage.md`: every command and option.
- `docs/dataset_strategy.md` and `docs/project_plan.md`: the original goals.
