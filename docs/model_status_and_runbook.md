# Model status, results and runbook

Status as of 2026-10-06. This document records what was run on the real
Backblaze Q1 2026 data, what was learned, what is still open, and how to run
the pipeline end to end, including the Q1+Q2 setup. Numbers are copied from
the run logs and experiment output; where something was not measured on real
data, it says so.

## 1. Where things stand

- **The pipeline runs end to end on real Q1 data** within the 20 GB memory
  cap. The primary model trains, and evaluation, the model card and MLflow
  logging complete.
- **The model is useful but far from the stated goal.** The goal is 95%
  precision at 35–50% recall, per drive. The best result we measured is
  about 22–31% precision at 26–35% recall on the validation split and about
  22% precision at 26% recall on test (Section 4).
- **We do not expect 95% precision to be reachable with one quarter of SMART
  data.** Section 5 explains why, and Section 7 lists what would change that.
- **The next step is a larger data set** (Q1 and Q2 2026) plus a richer
  feature set. It has been planned, but not started (Section 8).

## 2. How to run the pipeline

### 2.1 Setup

```bash
make install
```

### 2.2 Q1 2026 only (the configuration the results in this document use)

`configs/data.yaml`: `sources.backblaze.end_date: "2026-03-31"`.
`configs/model.yaml`: `splits` with `train_end: "2026-02-22"`,
`validation_end: "2026-03-06"`, `test_end: "2026-03-17"`.

```bash
make ingest-backblaze
make build-silver
make build-features
make build-labels
make train                     # add TRAIN_ARGS=--keep-work-dir to keep arrays
make experiment-model          # optional, see Section 6
make score-fleet
make plots
```

### 2.3 Q1 and Q2 2026 together

Change, before running:

- `configs/data.yaml`: `sources.backblaze.end_date: "2026-06-30"` (or the last
  date in your Q2 files), and `download.backblaze.quarters: ["Q1_2026", "Q2_2026"]`.
- `configs/model.yaml`: move the splits so the test period ends at least 30
  days before the data does (a 30-day label needs that window). A suggested
  starting point is `train_end: "2026-04-15"`, `validation_end: "2026-05-10"`,
  `test_end: "2026-05-31"`. Check these against your files.

Then run the same command sequence as in 2.2. Expect silver, gold and the
training join to take roughly twice as long and to use about twice the disk.
The two-quarter run has not been done yet.

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

**The 95% precision goal is not reachable with the current data.** At 95%
precision the best measured operating point catches about 1% of failing drives
(3 of 309). At 35% recall the best precision is about 22–31%, and the
false-alarm rate would need to fall roughly 50× to reach 95%. Nothing tried
so far moved the curve by more than a small factor.

The paper we reviewed (Amram et al., 2021) reports about 44% precision at a
12% false-alarm rate on one drive model with random splits. Our false-alarm
rate is already about 100× lower, and our splits are time-based, so the paper
does not support a 95% target either.

What could change the picture, in order of expected effect:

1. **More failing drives:** a second quarter (in progress) and possibly more.
2. ~~Per-drive-family models~~: tested on 2026-10-06 and worse than the pooled
   model for the three largest families (Section 4.1).
3. **A longer horizon** (30 days), once the split is valid.
4. **Richer SMART features** (Section 8).
5. **Tiered goals:** keep the destructive action at a high precision target
   and let warnings accept more false alarms. The tiers already exist (Section
   4); the targets in `configs/model.yaml` are placeholders to be set from the
   real cost of a false alarm per action.

## 8. Open items

**Data and features (planned, not started):**

- **Ingest more SMART attributes.** Currently only 5, 187, 188, 197 and 198
  reach the model. The paper's most useful ones that we drop are 3 (spin-up
  time), 7 (seek error rate), 190 (temperature difference), and both raw and
  normalized values of each attribute.
- **Lifetime counters** (4, 9, 12, 192, 193, 240–242). The paper removed these
  because they mostly encode age. Ingest them as a separate group so the
  experiment can switch them on and off.
- **Longer windows** (60 and 90 days) and "days since an error counter first
  became nonzero" (the paper cites Google's finding that the first scan error
  makes a drive 39 times more likely to fail within 60 days).
- **Real power-on hours** (smart 9) in place of `drive_age_days`, which is
  measured from the start of the loaded data.
- **Feature-count risk.** The gold table would grow several times over. The
  memory design should hold, but it has not been run at that width.

**Model and evaluation:**

- **The split fix.** The 10% holdout drives' history is moved into validation,
  and there is no purge gap before validation. This is a known distortion, and
  fixing it needs a `make build-labels` re-run.
- **Per-drive-family results** are in Section 4.1. The top three families were
  tested; the other families were not scored (too few failures).
- **Tier targets** are placeholders (15 / 25 / 40 / 60%).
- **The live agent loop** (`src/agent/nodes.py`) still uses the hand-picked
  cutoffs in `configs/agent.yaml`, not the trained tier thresholds.

**Other:**

- **HDD vs SSD:** not analyzed. `drive_model` is in the gold table, so a
  breakdown by drive type is cheap, but we expect it to explain few of the
  false alarms.
- **Commits not pushed:** `main` is ahead of `origin/main` by the recent model
  commits (see `git log`).

## 9. Reference: key config values

| File | Key | Current value | Meaning |
|---|---|---|---|
| `configs/model.yaml` | `primary_horizon_days` | 14 | Horizon the shipped model is trained and scored for |
| `configs/model.yaml` | `horizons_days` | [7, 14, 30] | Horizons whose labels exist |
| `configs/model.yaml` | `model.max_train_rows` | 5000000 | Training-row cap |
| `configs/model.yaml` | `model.positive_weight_power` | 0.5 | Positive weight = (neg/pos)^0.5 |
| `configs/model.yaml` | `model.early_stopping_rounds` | 50 | Early stopping patience (average precision) |
| `configs/model.yaml` | `threshold.target_precision` | 0.95 | Goal precision (kept as the target) |
| `configs/model.yaml` | `threshold.target_recall_range` | [0.35, 0.50] | Goal recall range |
| `configs/model.yaml` | `threshold.action_tier_precision` | 0.15 / 0.25 / 0.40 / 0.60 | Per-action precision targets (placeholders) |
| `configs/model.yaml` | `diagnostics.shap_enabled` | false | SHAP is slow on real data and fails its additivity check with extreme leaves |
| `configs/data.yaml` | `resource_limits.max_memory_gb` | 20 | RSS cap for every pipeline process |
| `configs/data.yaml` | `resource_limits.scratch_dir` | null | Scratch location (default `<gold_dir>/../tmp`) |
| `configs/data.yaml` | `sources.backblaze.end_date` | 2026-03-31 | Last date ingested |

## 10. Related documents

- `docs/developer_guide.md` section 5.10–5.12: the memory cap, how
  `make train` fits in RAM, and the model experiment workflow.
- `docs/dataset_strategy.md` and `docs/project_plan.md`: the original goals.
