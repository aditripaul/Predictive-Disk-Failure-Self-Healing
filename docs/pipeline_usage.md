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

### 2.8 Running and measuring the decision loop

Needs a trained run (`make train`) and the gold tables.

```bash
make agent-pit                               # the figures to quote
make agent-pit ARGS="--step-days 3 --bootstrap 2000"
make family-calibration                      # per-family recalibration test
make agent-fleet ARGS="--start-date 2026-06-01 --end-date 2026-06-23 --cycles 200"
```

`make agent-pit` snapshots the fleet at weekly dates across the test window:
every drive with a feature row on that date, scored once, judged on the label
at that same row. It prints pooled precision and recall per tier with
drive-cluster bootstrap intervals, the base rate precision lives at, a
per-date table, and one look against two at a matched false-alarm rate. It
saves the scored snapshots to `data/audit/agent_point_in_time_observations.parquet`
so `scripts/analyze_false_positives.py` can profile them without re-scoring.
Peak memory is about 6 GB (one date's fleet at a time) and a run takes about a
minute. The dates are test-split data, so each run is recorded in the
test-access log.

`make agent-fleet` exercises the loop itself and writes
`data/audit/agent_fleet_run.json`. With `--score` it also joins the decisions
to the labels, but those figures look ahead twice - maturity comes from the
dataset's last date, and "latest row" is a dying drive's last day - and the
run warns when it scores a window that ends before the data does. Use
`agent-pit` for anything you intend to quote (ADR 0003). The checkpoint
defaults to a temporary file; `--checkpoint-path` overrides it.

The decision loop never touches hardware: the executor and validator are
simulated and logged.

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
| `TRAIN_TAG=h30_seed0` | Appends `_h30_seed0` to every file the run writes (`model_evaluation_report`, `action_tiers`, `shap_feature_importance`, the model card, and the `--save-scores` parquet) and names the MLflow run after it, so it does not replace another run's. A tagged run refuses to start if any of those files exists; add `--overwrite` to `TRAIN_ARGS` to replace them. MLflow artifacts keep their canonical names (`action_tiers.json`, `frozen_spec.json`), because `score-fleet` and `evaluate-frozen` load them by name. Without a tag the fixed names are used and overwritten on each run, as before. |
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

`make train-sequence` refuses to start if the report, scores or weights for its
`--tag` (or the untagged names) already exist, so a repeat run needs a new
`--tag` or `--overwrite`. It checks before any training, so a collision fails in
seconds.

A full run logs one test-split access. To compare with, or blend into, LightGBM
on the same drives:

```bash
make train TRAIN_ARGS=--save-scores     # LightGBM scores; overwrites the evaluation report, so back it up
                                        # (or add TRAIN_TAG=<tag> to keep the existing files)
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

### Merged approach (`make train-merged`)

Trains XGBoost AFT and Cox survival models on the kept training arrays and
compares them, side by side and as validation-rank blends, with the two-stage
LightGBM and the LSTM and TCN scores, on the same rows, split dates and
drive-level metrics (`pipelines/train_merged_models.py`). It trains no
LightGBM, LSTM or TCN itself, so those run first, each with the same tag so nothing
is overwritten. One command does all four steps:

```bash
make train-all RUN_TAG=h30_seed0       # about 2.5 to 3 hours; HORIZON=30 by default
```

which is exactly:

```bash
make train TRAIN_TAG=h30_seed0 TRAIN_ARGS="--horizon-days 30 --keep-work-dir --save-scores"
make train-sequence SEQUENCE_ARGS="--arch lstm --horizon-days 30 --tag h30_seed0"
make train-sequence SEQUENCE_ARGS="--arch tcn --horizon-days 30 --tag h30_seed0"
make train-merged MERGED_TAG=h30_seed0 MERGED_ARGS="--lightgbm-scores data/tmp/lightgbm_scores_h30_h30_seed0.parquet --lstm-scores data/tmp/sequence_model/scores_h30_h30_seed0.npz --tcn-scores data/tmp/sequence_model/scores_h30_tcn_h30_seed0.npz"
```

`train-all` stops at the first failing step. To resume, run the remaining steps from
the list above: a tagged step refuses to overwrite its own earlier outputs, so
re-running `train-all` with the same tag stops at step 1. It leaves the kept training
arrays (several GB) in `data/tmp/`; `make clean-kept` removes them.

The three score paths are passed explicitly because a tagged run names its outputs
with the tag (the defaults are the untagged names). A sequence-score file written
before the merged approach has no day indices and is rejected before any model
trains. The kept work dir is the newest `train_model_frame_*` unless `--work-dir`
is given.

| Option | Effect |
|---|---|
| `MERGED_TAG=name` | Appends `_name` to every output: `merged_evaluation_report`, `merged_scores_h30.parquet`, `agent_thresholds_h30.json`, and the trained boosters `aft_h30.json` and `cox_h30.json`. Reports go to `data/audit/data_quality_reports/`, the rest to `data/tmp/merged/`. |
| `MERGED_ARGS=--overwrite` | Replaces the tag's existing outputs. Without it the run refuses to start if any exists. |
| `MERGED_ARGS=--smoke` | Small-subset run with `.smoke` outputs, no MLflow run and no test access recorded. Not a result. |

A run that is not `--smoke` logs to MLflow (the experiment in `configs/model.yaml`):
parameters, per-model and per-ensemble drive-level metrics, and the report, scores,
thresholds and boosters as artifacts. The run id is written into the report as
`mlflow_run_id`, and the run records one test-split access.

To act on the blended scores, give the agent its own output file, since
`make agent-fleet` refuses to overwrite an existing one:

```bash
make agent-fleet ARGS="--scores-path data/tmp/merged/merged_scores_h30_h30_seed0.parquet --out data/audit/agent_fleet_merged_h30_seed0.json"
```

#### Packing a score table for git

A `train-merged` score table is about 1 GB, over GitHub's 100 MB per-file limit,
and git history keeps every byte pushed. To commit one (a frozen run's, say), pack
it, and rebuild it when needed:

```bash
make pack-merged-scores SCORES=data/tmp/merged/merged_scores_h30_h30_seed0.parquet \
    PACK_DIR=frozen_runs/<run id>/scores
make unpack-merged-scores PACK_DIR=frozen_runs/<run id>/scores \
    OUT=data/tmp/merged/merged_scores_rebuilt.parquet
```

`pack` writes `part-NN.parquet` files of at most about 45 MB and a `MANIFEST.json`
(columns, dtypes, the blend recipes, per-part row counts). To get there it:

1. drops the rank-blend columns, which are a deterministic function of the member
   scores and their validation scores, and recomputes them on rebuild, summing in
   the recorded member order;
2. stores the columns in `--float32` (default `lstm tcn`, the sequence nets) as
   float32, and refuses a column that is not exactly representable;
3. rebuilds the table from the parts and fails unless it equals the original.

Output is about 40% smaller (423 MB for the 1 GB table in the frozen run), not
small: the scores are high-entropy floats. Both commands refuse to overwrite
existing output; add `PACK_ARGS=--overwrite`. `PACK_ARGS` also takes
`--rows-per-part N`, `--float32 col ...` and `--ensemble NAME=m1,m2,...`
(default: `merged_models.ensembles` in `configs/model.yaml`; the pack records its
own copy, so later config changes do not alter a rebuild). The rebuilt file
equals the original on every value but is not byte-identical, so compare values,
not file hashes.

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
| `make train-merged` | AFT and Cox survival models plus rank blends with LightGBM, LSTM and TCN, compared on the same rows. `MERGED_TAG` and `MERGED_ARGS`; see "Merged approach" above. |
| `make train-all RUN_TAG=<tag>` | The four merged-approach steps under one tag: `train` (keeping arrays, saving scores), `train-sequence` for the LSTM and the TCN, then `train-merged` on those outputs. About 2.5 to 3 hours; see "Merged approach" above. |
| `make pack-merged-scores`, `make unpack-merged-scores` | Pack a `train-merged` score table into parts under 50 MB for git, and rebuild it exactly (see "Packing a score table for git"). |
| `make agent-fleet` | Runs the decision loop on the real gold features and the trained model, executor simulated (section 2.8). Refuses to overwrite `--out` (default `data/audit/agent_fleet_run.json`); give a new `--out` or add `--overwrite`. |
| `make agent-pit` | Measures the flagging policy at fixed dates (section 2.8). |
| `make family-calibration` | Per-family recalibration, fitted on validation snapshots and judged on test (section 2.8). |
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
