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
    start_date: "2026-01-01"
    end_date: "2026-06-30"        # Q1 and Q2 (the repo default); "2026-03-31" for Q1 only
download:
  backblaze:
    quarters: ["Q1_2026"]         # or ["Q1_2026", "Q2_2026"]
```

**`configs/model.yaml`** controls the train, validation and test periods. The
test period must end at least 30 days before the last day of data, because a
30-day label needs that window.

```yaml
splits:
  strategy: chronological
  train_end: "2026-04-15"         # Q1 + Q2 settings (the repo default)
  validation_end: "2026-05-10"
  test_end: "2026-05-31"
```

For Q1 only, use `train_end: "2026-02-22"`, `validation_end: "2026-03-06"`,
`test_end: "2026-03-17"` (valid for the 14-day horizon). Check the split dates
against the last date in your files before running.

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

### 2.3 Trial runs and the experiment tool

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

### 2.4 Where the outputs go

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
| Kept training arrays | `data/tmp/train_model_frame_*` (only with `--keep-work-dir`) |

### 2.5 Resource limits

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
| `TRAIN_ARGS="--horizon-days 30"` | Trains for a horizon in `horizons_days`. The run is logged with that horizon; `score-fleet` only uses runs for the primary horizon (`primary_horizon_days`, now 30). |

Examples:
```bash
make train
make train TRAIN_ARGS=--keep-work-dir
make train TRAIN_ARGS="--horizon-days 30"
```

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
| `--two-stage` | Trains a second model on the rows the pooled model flags (out-of-fold scores on train) and compares it with the pooled model alone at 5/10/20/35% recall on the same test drives. An experiment only: nothing else uses it. |
| `--per-model [N]` | Trains one model per drive family for the N families with the most failing validation drives (default 3), each compared with the pooled model on the same test drives. |

Examples:
```bash
make experiment-model EXPERIMENT_ARGS="--variants reg_spw xgboost --skip-baseline"
make full-experiment
make full-experiment FULL_EXPERIMENT_ARGS="--variants reg_spw xgboost --per-model 5"
```

### Whole pipeline

| Command | What it does |
|---|---|
| `make full-pipeline` | Runs ingest, silver, features, labels, `full-experiment`, `score-fleet` and `plots`, in that order, stopping at the first failure. Does not download data or remove kept arrays at the end. |

Expect it to take well over an hour on Q1 alone.

### Other targets

| Command | What it does |
|---|---|
| `make experiment-model` | See above. |
| `make build-sequences`, `make train-lstm` | Optional LSTM branch. Needs `uv sync --extra torch` first. |
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
- `docs/developer_guide.md` sections 5.10–5.12: memory cap, how `make train` fits
  in RAM, and the experiment tool.
