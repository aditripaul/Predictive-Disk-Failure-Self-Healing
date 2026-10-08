.PHONY: install lint format test test-unit test-integration test-chaos test-golden test-property \
	data-check smoke coverage ci clean clean-data \
	download-backblaze download-smartz \
	ingest-backblaze ingest-smartz ingest-synthetic-stub build-silver build-features build-labels train experiment-model clean-kept full-experiment full-pipeline \
	build-sequences train-lstm score-fleet plots \
	agent-demo dashboard api final-report

# --- setup -----------------------------------------------------------------

install:
	uv sync --extra dev

# --- code quality ------------------------------------------------------------

lint:
	uv run ruff check .
	uv run mypy src data_contracts

format:
	uv run ruff format .
	uv run ruff check --fix .

# --- tests -------------------------------------------------------------------
# `test` runs everything; the test-* targets run one layer at a time (see
# docs/developer_guide.md section 12 for what each layer actually exercises).

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test-chaos:
	uv run pytest tests/chaos

test-golden:
	uv run pytest tests/golden

test-property:
	uv run pytest tests/property

data-check: test-golden

smoke:
	uv run pytest tests/smoke

coverage:
	uv run pytest --cov=src --cov=data_contracts --cov=pipelines \
		--cov-report=term-missing --cov-report=html

# `ci` is what .github/workflows/ci.yml runs; use it to reproduce a CI
# failure locally before pushing.
ci: lint test

# --- housekeeping --------------------------------------------------------------
# `clean` only removes caches/build artifacts - never data or MLflow/agent
# runtime state. `clean-data` removes regenerated pipeline outputs
# (bronze/silver/gold/audit) but never raw source data under data/raw/.

clean:
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache htmlcov .coverage

clean-data:
	find data/bronze data/silver data/gold data/audit -type f -not -name ".gitkeep" -delete
	find data/bronze data/silver data/gold data/audit -mindepth 1 -type d -empty -delete

# Removes the kept training work directories left by
# `make train TRAIN_ARGS=--keep-work-dir` (several GB each). Only these
# scratch directories are touched; data/tmp/ otherwise is left alone.
clean-kept:
	rm -rf data/tmp/train_model_frame_*

# The whole comparison in one command: clear old kept work directories, train
# once keeping the arrays, then run the variant, false-alarm and per-family
# comparison on them. Override the experiment with EXPERIMENT_ARGS=... .
FULL_EXPERIMENT_ARGS ?= --variants reg_spw --skip-baseline --per-model 3 --deep-dive reg_spw
full-experiment: clean-kept
	$(MAKE) train TRAIN_ARGS=--keep-work-dir
	$(MAKE) experiment-model EXPERIMENT_ARGS="$(FULL_EXPERIMENT_ARGS)"

# The complete run, raw bronze CSVs to fleet scores and plots, with the model
# comparison included. Assumes the raw files are already downloaded (run
# `make download-backblaze` first). Stops at the first failing step.
full-pipeline:
	$(MAKE) ingest-backblaze
	$(MAKE) build-silver
	$(MAKE) build-features
	$(MAKE) build-labels
	$(MAKE) full-experiment
	$(MAKE) score-fleet
	$(MAKE) plots

# --- data download ---------------------------------------------------------
# Configurable time period, no default quarter is hardcoded: set
# download.backblaze.quarters (or start_quarter/end_quarter) in
# configs/data.yaml, or override per-run with ARGS, e.g.:
#   make download-backblaze ARGS="--quarters Q1_2025 Q2_2025"
#   make download-backblaze ARGS="--start-quarter Q1_2025 --end-quarter Q4_2025"

download-backblaze:
	uv run python pipelines/download_backblaze.py $(ARGS)

download-smartz:
	uv run python pipelines/download_smartz.py $(ARGS)

# --- data pipeline -------------------------------------------------------------

ingest-backblaze:
	uv run python pipelines/ingest_backblaze.py $(ARGS)

ingest-smartz:
	uv run python pipelines/ingest_smartz.py

ingest-synthetic-stub:
	uv run python pipelines/ingest_synthetic_stub.py

build-silver:
	uv run python pipelines/build_silver.py

build-features:
	uv run python pipelines/build_gold_features.py

build-labels:
	uv run python pipelines/build_labels.py

# `make train TRAIN_ARGS=--keep-work-dir` keeps the scratch arrays for
# `make experiment-model` (pipelines/experiment_model.py).
train:
	uv run python pipelines/train_model.py $(TRAIN_ARGS)

experiment-model:
	uv run python pipelines/experiment_model.py $(EXPERIMENT_ARGS)

# Optional LSTM comparison branch (docs/dataset_strategy.md section 16.2).
# Requires `uv sync --extra torch` first - not part of the default install.
build-sequences:
	uv run python pipelines/build_sequences.py

train-lstm:
	uv run python pipelines/train_lstm.py

score-fleet:
	uv run python pipelines/score_fleet.py

# Single-shot evaluation of a frozen run on the sealed quarter or SMART-Z:
#   make evaluate-frozen ARGS="--run-id <mlflow run id> --split sealed"
evaluate-frozen:
	uv run python pipelines/evaluate_frozen.py $(ARGS)

plots:
	uv run python pipelines/generate_performance_plots.py

final-report: plots
	uv run python pipelines/generate_final_report.py

# --- services --------------------------------------------------------------

agent-demo:
	uv run python -m src.agent.demo

dashboard:
	uv run streamlit run src/dashboards/app.py

api:
	uv run python pipelines/run_api.py
