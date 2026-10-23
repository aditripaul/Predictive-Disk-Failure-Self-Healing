# Pipeline usage: the standard (Polars) pipeline and the DuckDB spike

This document says how to run the complete standard pipeline, and how to use
the DuckDB spike. The two are not interchangeable:

- **The standard pipeline** (Polars) is complete. It runs from raw CSVs to a
  trained model, scored fleet and plots. Every number in
  `docs/model_status_and_runbook.md` came from it.
- **The DuckDB pipeline does not exist yet.** What exists is a spike,
  `pipelines/spike_duckdb_extract.py`, which reimplements one step (the
  train-split extraction) to measure its memory. Its output is not used by
  `make train`, `make score-fleet` or any other target.

Background on the memory design, and why the DuckDB spike exists, is in
`docs/model_status_and_runbook.md` section 3.

## 1. Setup (both pipelines)

```bash
make install
```

DuckDB is already a dependency (`pyproject.toml`), so nothing else needs
installing.

## 2. The standard pipeline

### 2.1 Choose the data period

The period is set in two files. Change both before running.

**`configs/data.yaml`** controls which raw files are ingested and which
quarters are downloaded:

```yaml
sources:
  backblaze:
    start_date: "2025-10-01"      # Q4 2025 onwards (the repo default)
    end_date: "2026-06-30"        # through Q2 2026; "2026-03-31" for Q1 only
download:
  backblaze:
    quarters: ["Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]
```

The download list is deliberately wider than what is ingested: Q3 2025 is on
disk for the parked 60-day horizon experiment but `start_date` excludes it.
Bronze reflects whatever was last ingested and `make build-silver` reads
Bronze rather than the raw CSVs, so narrowing `start_date` does not narrow
what the pipeline sees until Bronze is rebuilt (`rm -rf
data/bronze/backblaze`).

**`configs/model.yaml`** controls the train, validation and test periods. The
test period must end at least 30 days before the last day of data, because a
30-day label needs that window.

```yaml
splits:
  strategy: chronological
  train_end: "2026-02-14"         # the repo default: Q4 2025 + Q1 2026 train
  validation_end: "2026-03-31"    # and validate, Q2 2026 tests
  test_end: "2026-05-31"
  sealed_start: "2026-07-01"      # read only by `make evaluate-frozen`
```

`sealed_start` holds a final window back from every modelling choice. Set it
to a date your data actually reaches (or `null` to disable it); rows from
there on are labelled `split=sealed` and are scored once, later, by
`make evaluate-frozen` (section 2.5).

`make train` applies these dates when it extracts each split: training uses
rows dated at least one horizon (30 days) before `train_end`, validation uses
only dates after `train_end`, and test stops at `test_end`
(`splits.purge_label_window` and `splits.validation_after_train_only`, both
`true`; see `docs/model_status_and_runbook.md` section 4.4).

One quarter is too short for the 30-day default: after the purge, training
would have about three weeks of data. For Q1 only, set
`primary_horizon_days: 14` and use `train_end: "2026-02-22"`,
`validation_end: "2026-03-06"`, `test_end: "2026-03-17"`. Check the split
dates against the last date in your files before running.

### 2.2 Run the steps in order

```bash
make download-backblaze           # skip if the quarters are already in data/raw/backblaze/
make ingest-backblaze             # raw CSVs -> data/bronze/
make build-silver                 # cleaned drive-day and canonical tables -> data/silver/
make build-features               # gold feature table -> data/gold/features/
make build-labels                 # labels and splits for horizons 7, 14, 30 -> data/gold/labels/
make train                        # trains the primary model and writes the model card
make score-fleet                  # scores the latest row for each drive
make plots                        # performance plots from the audit reports
```

Use `make train TRAIN_ARGS=--keep-work-dir` if you plan to run the experiment
tool, which needs the kept arrays.

Each step needs the one before it. If a step fails because an input is missing,
it says which earlier step to run.

### 2.3 Run everything in one command

Once the period is set (section 2.1) and the raw files are downloaded:

```bash
make full-pipeline
```

It runs, in order: `ingest-backblaze`, `build-silver`, `build-features`,
`build-labels`, `full-experiment` (which clears old kept arrays, trains with
them kept, and runs the per-family, false-alarm and persistence comparison),
`score-fleet` and `plots`. It stops at the first failing step. It does not
download data and does not run `clean-kept` at the end, so run `make clean-kept`
when you are done with the arrays.

Expect it to take well over an hour on the Q1 data, and longer for Q1 and Q2.

### 2.4 Trial runs and the experiment tool

One command runs the whole comparison (clears old kept directories, trains
with the arrays kept, then runs the per-family, false-alarm and persistence
analyses):

```bash
make full-experiment
```

Change the experiment with `FULL_EXPERIMENT_ARGS`, for example
`make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw xgboost --per-model 5"`.
Afterwards run `make clean-kept` when you are finished with the arrays.

The individual steps are still available:

```bash
# A different horizon (must be one of horizons_days in configs/model.yaml).
# The run is logged with its horizon; score-fleet only uses runs for the
# primary horizon (primary_horizon_days in configs/model.yaml, now 30).
make train TRAIN_ARGS="--horizon-days 30"

# Compare model variants on the kept arrays (run make train with --keep-work-dir first).
make experiment-model EXPERIMENT_ARGS="--variants reg_spw xgboost --skip-baseline --deep-dive reg_spw --per-model 3"
```

Delete the kept work directories afterwards with `make clean-kept`
(it removes `data/tmp/train_model_frame_*` and nothing else).

### 2.5 Scoring a sealed or external split, once

`configs/model.yaml`'s `splits.sealed_start` (2026-07-01 by default) labels
every row from that date on `split=sealed`. No training, validation or test
step reads it. SMART-Z rows are sealed the same way, as
`split=external_smartz`.

To score one of them, you need a training run that was frozen *before* that
data arrived. `make train` logs everything such an evaluation needs as a
`frozen_spec.json` artifact on the run - the feature list, the drive
threshold, the per-tier thresholds and the fitted calibrator - so nothing is
re-chosen at evaluation time:

```bash
make evaluate-frozen ARGS="--run-id <mlflow run id> --split sealed"
make evaluate-frozen ARGS="--run-id <mlflow run id> --split external_smartz"
```

It writes `data/audit/frozen_evaluations/<split>__<run id>.json` and **refuses
to run again** for the same split and run, because a second look is a second
comparison. It needs an assembled frame to read the split's rows from, so run
`make train TRAIN_ARGS=--keep-work-dir` first (or pass `--frame`).

Options: `--run-id` (required), `--split` (`sealed` or `external_smartz`),
`--n-boot` (bootstrap draws, default 1000), `--frame` (otherwise the newest
`data/tmp/train_model_frame_*/frame.parquet`).

### 2.6 Where the outputs go

| Output | Location |
|---|---|
| Silver tables | `data/silver/` |
| Gold features | `data/gold/features/part.parquet` |
| Labels and splits | `data/gold/labels/part.parquet` |
| Evaluation report | `data/audit/data_quality_reports/model_evaluation_report.json` |
| Action tiers | `data/audit/data_quality_reports/action_tiers.json` (also in MLflow) |
| Model card | `data/audit/model_cards/v<version>_h<horizon>d.md` |
| Fleet predictions | `data/audit/predictions/<date>.json` |
| MLflow runs | `mlflow/mlflow.db` |
| Frozen-split results | `data/audit/frozen_evaluations/<split>__<run id>.json` |
| Log of every test-split computation | `data/audit/test_access_log.jsonl` (append-only) |
| Approval queue and decision trail | `data/audit/audit.sqlite` (the running API; `AUDIT_DB_PATH` overrides) |
| Kept training arrays | `data/tmp/train_model_frame_*` (only with `--keep-work-dir`) |
| LSTM report, scores, weights | `data/audit/data_quality_reports/sequence_evaluation_report[_<tag>].json`; `data/tmp/sequence_model/scores_h<horizon>[_<tag>].npz` and `lstm_h<horizon>[_<tag>].pt` |
| LSTM vs LightGBM comparison | `data/audit/data_quality_reports/sequence_vs_tree_seeds.json` |

### 2.7 Resource limits

Every pipeline process is capped at `resource_limits.max_memory_gb`
(`configs/data.yaml`, default 20). The cap is enforced on resident memory: a
process that exceeds it logs `memory_limit_exceeded` and exits with code 86.
Scratch files go to `resource_limits.scratch_dir` (default
`<gold_dir>/../tmp`).

## 2A. Make command reference

Every command below is run from the repository root. Variables after the
target name change its behavior; the others take no arguments.

### Setup and checks

| Command | What it does |
|---|---|
| `make install` | Installs the Python environment (`uv sync --extra dev`). |
| `make lint` / `make format` | Runs ruff (check, or format). |
| `make test` | Runs the full test suite. |
| `make test-unit` | Runs only the unit tests. |
| `make ci` | Lint and tests, as the continuous-integration job runs them. |
| `make clean` | Removes Python caches and tool caches. Keeps data. |
| `make clean-data` | Deletes regenerated outputs in `data/bronze`, `silver`, `gold` and `audit`. Never touches `data/raw`. |

### Data pipeline (in order)

| Command | Reads | Writes |
|---|---|---|
| `make download-backblaze` | quarters in `configs/data.yaml` | `data/raw/backblaze/` |
| `make ingest-backblaze` | raw CSVs inside `sources.backblaze.start_date`/`end_date` | `data/bronze/` |
| `make build-silver` | bronze | `data/silver/` |
| `make build-features` | silver | `data/gold/features/` |
| `make build-labels` | gold features | `data/gold/labels/` (horizons 7, 14, 30; splits) |
| `make train` | gold and labels | MLflow run, model card, evaluation and action-tier reports |
| `make score-fleet` | latest row per drive, latest MLflow run for the primary horizon (30 days) | `data/audit/predictions/<date>.json` |
| `make plots` | audit reports | `data/audit/plots/` |

Each step needs the ones before it.

### Training options

`make train` takes its options through `TRAIN_ARGS`:

| Option | Effect |
|---|---|
| `TRAIN_ARGS=--keep-work-dir` | Keeps the training arrays in `data/tmp/train_model_frame_*` for the experiment tool. Delete them with `make clean-kept`. |
| `TRAIN_ARGS="--horizon-days 14"` | Trains for a horizon in `horizons_days` other than the default. The run is logged with that horizon; `score-fleet` only uses runs for the primary horizon (`primary_horizon_days`, now 30). |
| `TRAIN_ARGS=--save-scores` | Also writes the final validation and test scores (`drive_id`, `date`, `label`, `score`; two-stage included) to `data/tmp/lightgbm_scores_h<horizon>.parquet`, so another model can be compared or blended on the same drives (`scripts/compare_sequence_vs_tree.py`). |
| `TRAIN_ARGS=--two-stage` | Forces the second stage on for this run. It is already on by default (`model.two_stage.enabled: true`); set that to `false` for the single model. The second stage is a model that re-ranks the first model's highest-scoring rows (`src/models/two_stage.py`). Thresholds, tiers and reports are then built on the combined score, with the first model alone logged beside it (`stage1_alone_precision_at_recall`). `make score-fleet` uses the second stage automatically for a run trained with it. Same as `model.two_stage.enabled: true`. Adds three extra fits of the first model, so roughly doubles the model-fitting time. |

Examples:
```bash
make train
make train TRAIN_ARGS=--keep-work-dir
make train TRAIN_ARGS="--two-stage --keep-work-dir"
make train TRAIN_ARGS="--horizon-days 14"
```

Every drive-level result in the log and reports carries, beside its
precision on the real fleet, `lift` and `precision_at_failure_rate` (the same
alerts on test sets where 15% and 50% of drives fail). The configured target
is judged on the real-fleet figure.

### Sequence model (LSTM)

`make train-sequence` needs the torch extra. `pyproject.toml` resolves torch from
PyTorch's CPU index (the PyPI Linux wheel pulls about 3 GB of CUDA libraries that
this CPU-only run never uses), so install with:

```bash
uv sync --extra dev --extra torch
```

A plain `uv sync` removes torch again; `make train-sequence` runs `uv run --extra
torch`, which keeps it. A full run takes about 35 minutes on 14 CPU threads
(training about 5 minutes per epoch on a 3,000,000-row sample, then scoring 15.4M
validation and 21.3M test drive-days) and fits in the 20 GB memory cap.

| Option (`SEQUENCE_ARGS=...`) | Effect |
|---|---|
| `--drive-fraction 0.02 --max-epochs 2 --max-train-rows 200000` | A smoke run on 2% of drives, about 2 minutes. Writes `*.smoke.*` files and logs no test access. Its metrics are noise; it checks that the pipeline runs. |
| `--seed N --tag sN` | Another seed. The seed changes the initial weights, batch order and training negatives; the early-stopping validation subsample stays fixed. `--tag` keeps the report, scores and weights from overwriting the untagged run. |
| `--learning-rate`, `--dropout`, `--hidden-size` | Override `sequence_model.*` in `configs/model.yaml` for one run. |
| `--train-only --tag t1` | Trains and stops, writing `sequence_tuning_t1.json` (validation average precision per epoch). Reads no test rows and logs no test access: for choosing settings on validation. |
| `--horizon-days N` | A horizon other than the primary one. |

A full run logs one test-split access. To compare with, or blend into, LightGBM
on the same drives:

```bash
make train TRAIN_ARGS=--save-scores     # LightGBM scores; overwrites the evaluation report, so back it up
make train-sequence                     # the untagged run (called s0 below)
make train-sequence SEQUENCE_ARGS="--seed 1 --tag s1"
uv run --extra torch python scripts/compare_sequence_vs_tree.py --tags s0 s1
```

`compare_sequence_vs_tree.py` matches rows on (drive, date), refuses to run if the
row order or labels do not line up, and reports each LSTM run alone and blended
with LightGBM (rank average, weight chosen on validation only), their ensemble,
the drives each catches at matched false-alarm rates, and a paired bootstrap on
blend minus LightGBM. It logs one test-split access. Results: runbook sections
5.9 to 5.12.

### Model comparison

| Command | What it does |
|---|---|
| `make experiment-model` | Runs the comparison on kept arrays from an earlier `make train TRAIN_ARGS=--keep-work-dir`. Options go in `EXPERIMENT_ARGS`. |
| `make full-experiment` | Clears old kept arrays, trains with `--keep-work-dir`, then runs the comparison with `FULL_EXPERIMENT_ARGS` (default: `--variants reg_spw --skip-baseline --per-model 3 --deep-dive reg_spw`). |
| `make clean-kept` | Removes `data/tmp/train_model_frame_*` and nothing else. |

Options for `EXPERIMENT_ARGS` and `FULL_EXPERIMENT_ARGS`:

| Option | Effect |
|---|---|
| `--variants NAME ...` | The named configurations to compare (see `pipelines/experiment_model.py`). |
| `--skip-baseline` | Skips the logistic-regression reference, which takes about two minutes. |
| `--deep-dive [NAME]` | Prints the false-alarm breakdown and persistence rules for one variant (default `reg_spw`). |
| `--two-stage` | Runs the same second-stage code as `make train TRAIN_ARGS=--two-stage` and compares it with the first model alone at 5/10/20/35% recall on the same test drives. |
| `--two-stage-seeds N` | With `--two-stage`: repeats it with N seeds and prints the mean, the range of the precision gain, and in how many seeds the two-stage model was better on both precision and recall. Use it to tell a real gain from chance; each seed takes a few minutes. |
| `--anomaly-stage` | Fits an isolation forest on healthy train rows and reports the anomaly score alone, an anomaly-filter-then-classify cascade, and the score as an extra feature, each against the pooled model. |
| `--per-model [N]` | Trains one model per drive family for the N families with the most failing validation drives (default 3), each compared with the pooled model on the same test drives. |

Examples:
```bash
make experiment-model EXPERIMENT_ARGS="--variants reg_spw xgboost --skip-baseline"
make full-experiment
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw xgboost --per-model 5"
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw --skip-baseline --two-stage --two-stage-seeds 5"
```

Variants worth knowing (`--variants`): `reg_spw` is the pipeline's own
configuration; `xgboost` and `xgboost_deep` swap the model family;
`xgboost_aft` and `xgboost_aft_spw` are survival models trained on time to
failure; `reg_spw_old_features`, `reg_spw_no_lifetime` and
`reg_spw_no_secondary` remove feature groups. A kept work directory made
before 2026-10-06 lacks the columns the survival variants need; rebuild it
with `make train TRAIN_ARGS=--keep-work-dir`.

### Whole pipeline

| Command | What it does |
|---|---|
| `make full-pipeline` | Runs ingest, silver, features, labels, `full-experiment`, `score-fleet` and `plots`, in that order, stopping at the first failure. Does not download data or remove kept arrays at the end. |

Expect it to take well over an hour on Q1 alone.

### Other targets

| Command | What it does |
|---|---|
| `make experiment-model` | See above. |
| `make build-sequences`, `make train-lstm` | Optional LSTM branch (one window per drive; a rough comparison only). Needs `uv sync --extra dev --extra torch` first. |
| `make train-sequence` | LSTM over each drive's raw 30-day history, evaluated exactly like `make train` (same labelled drive-days, split rules and drive-level metrics), so the two can be compared. Options go in `SEQUENCE_ARGS`; see below. |
| `make plot-late-hits` | Draws what became of the drives the model flagged: three figures in `data/audit/plots/` (`late_hits_outcomes.png`, `late_hits_lead_time.png`, `late_hits_timelines.png`) plus `late_hits_summary.json`. Shows that alerts the 30-day label counts as false alarms often failed in days 31 to 60. Needs `make agent-pit` first; options in `ARGS` (`--tier`, `--max-lanes`). |
| `make evaluate-frozen` | Scores one sealed or external split, once, against a frozen run (section 2.5). `ARGS="--run-id <id> --split sealed"`. |
| `make final-report` | Writes `data/audit/data_quality_reports/final_evaluation_report.json`: the model's drive-level results and action tiers, a goal-status section (target met or not, and the gap), the data period, and the chaos, latency and crash-recovery test results. Runs those tests, so it takes a few minutes. |
| `make agent-demo` | Runs the decision-agent demo once. |
| `make api` / `make dashboard` | Starts the FastAPI service / the Streamlit dashboard. |

## 3. The DuckDB spike

### 3.1 What it does

`pipelines/spike_duckdb_extract.py` performs the train-split extraction of
`make train` in DuckDB instead of Polars:

1. Counts the labels in the train split.
2. Computes the negative-row sample (same rule as the pipeline, but using
   DuckDB's hash).
3. Writes the filtered rows, cast to float32 with nulls set to zero, to a
   temporary Parquet file. DuckDB spills to a temporary directory under a
   memory limit.
4. Reads that file in batches into a preallocated float32 matrix and saves
   `x_train.duckdb.npy` and `y_train.duckdb.npy` next to the frame.

It reports `peak_rss_mb`, `total_seconds` and `row_count` as JSON.

### 3.2 What it does not do

- It is **not** a pipeline. No other step reads its output, and nothing in
  `make` calls it.
- It does **not** change the model. The model still trains on the Polars
  arrays.
- It does **not** cover silver, gold, labels, validation or test extraction,
  or training.
- Its sample is not row-for-row identical to the Polars sample (different
  hash), so DuckDB and Polars runs are not directly comparable.

### 3.3 How to run it

Create the kept work directory first (it holds the joined frame):

```bash
make train TRAIN_ARGS=--keep-work-dir
```

Then run the spike on it:

```bash
uv run python pipelines/spike_duckdb_extract.py \
    --work-dir data/tmp/train_model_frame_XXXX \
    --memory-limit 8GB
```

Options:

- `--memory-limit`: DuckDB's memory limit, for example `8GB`.
- `--temp-dir`: where DuckDB spills. Defaults to `<work-dir>/_duckdb_spill`.
- `--max-train-rows`: negative-row cap, default 5000000 (matches the pipeline).
- `--uncapped`: keep every train row.
- `--compare`: also check the output against the Polars feature matrix. Only use
  it on data small enough to hold twice in memory, since it builds the Polars
  reference as well.

### 3.4 Reading the result

Compare `peak_rss_mb` with the Polars extraction, which peaked at about 6.0 GB
on the Q1 data. A clearly lower peak at a similar `total_seconds` is the
signal that DuckDB is worth more work. A lower peak with a much longer runtime
is not.

Afterwards, delete the spike's output and the kept work directory:

```bash
make clean-kept
```

## 4. Choosing between them

| Question | Use |
|---|---|
| Train, score or report on the data | The standard pipeline (section 2) |
| Test whether DuckDB would reduce memory | The spike (section 3) |
| Use DuckDB for real results | Not yet available |

If the spike shows a clear memory saving, the next step is to port the heavy
steps one at a time, each checked for exact equality against the Polars output
before it replaces that step. That work has not started.

## 5. Related documents

- `docs/model_status_and_runbook.md`: results, the memory design, the goal
  assessment and open items.
- `docs/developer_guide.md` sections 5.10–5.14: memory cap, how `make train`
  fits in RAM, the experiment tool, the current evaluation rules and second
  stage, and the sealed split and frozen evaluation.
