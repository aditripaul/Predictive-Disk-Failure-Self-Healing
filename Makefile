.PHONY: install lint format install-hooks check-paths agent-pit-validation agent-knowledge pack-knowledge unpack-knowledge test test-unit test-integration test-chaos test-golden test-property \
	data-check smoke coverage ci clean clean-data \
	download-backblaze download-smartz \
	ingest-backblaze ingest-smartz ingest-synthetic-stub build-silver build-features build-labels train experiment-model clean-kept full-experiment full-pipeline \
	build-sequences train-lstm train-sequence train-merged train-all agent-models merged-tables tune-blend-weights recall-policy pack-merged-scores unpack-merged-scores score-fleet plots \
	agent-demo agent-fleet agent-pit family-calibration plot-late-hits dashboard api final-report

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

# No absolute paths in commits (scripts/check_no_absolute_paths.py).
# `install-hooks` points git at scripts/githooks (pre-commit, commit-msg);
# `check-paths` checks the staged changes, or a range: make check-paths RANGE=main..HEAD
install-hooks:
	git config core.hooksPath scripts/githooks

check-paths:
	python3 scripts/check_no_absolute_paths.py $(if $(RANGE),--range $(RANGE),)

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

# Regenerable data only. test_access_log.jsonl is deliberately spared: it is the
# append-only record of every test-split computation (src/models/access_log.py),
# so the comparison count behind a reported result is a fact rather than an
# estimate. Deleting it cannot be undone by re-running the pipeline, and would
# silently reset that count to zero.
clean-data:
	find data/bronze data/silver data/gold data/audit -type f -not -name ".gitkeep" -not -name "test_access_log.jsonl" -delete
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
# `make train TRAIN_TAG=h30_seed0 TRAIN_ARGS="--keep-work-dir --save-scores"` suffixes
# every file the run writes (report, action tiers, SHAP, model card, saved scores) and
# names the MLflow run, so it does not replace the untagged run's. A tagged run refuses
# to start if its files exist (--overwrite in TRAIN_ARGS to replace). Feed its scores
# to the merged run with MERGED_ARGS="--lightgbm-scores data/tmp/lightgbm_scores_h30_h30_seed0.parquet".
train:
	uv run python pipelines/train_model.py $(if $(TRAIN_TAG),--tag $(TRAIN_TAG)) $(TRAIN_ARGS)

# Refuses to overwrite an existing report/scores/weights; give each run its own
# `--tag` (SEQUENCE_ARGS="--arch tcn --tag s1") or add --overwrite.
train-sequence:
	uv run --extra torch python pipelines/train_sequence_model.py $(SEQUENCE_ARGS)

# `make train-merged MERGED_TAG=h30_seed0` tags every output (report, scores,
# thresholds, boosters) with _h30_seed0 so it does not overwrite another run's, and
# names the MLflow run after it. Extra flags (e.g. --smoke): MERGED_ARGS=... .
# It refuses to start if any output for that tag exists (add --overwrite via
# MERGED_ARGS to replace), and never writes the files `make train` produces.
train-merged:
	uv run python pipelines/train_merged_models.py $(if $(MERGED_TAG),--tag $(MERGED_TAG)) $(MERGED_ARGS)

# A train-merged score table is about 1 GB, over GitHub's 100 MB file limit. Pack it
# into parts under 50 MB (drops the recomputable blend columns, stores the sequence
# nets' scores as exact float32) and rebuild it; the pack is verified equal to the
# original before it finishes. Both refuse to overwrite existing output (add
# --overwrite via PACK_ARGS).
#   make pack-merged-scores SCORES=data/tmp/merged/merged_scores_h30_h30_seed0.parquet \
#       PACK_DIR=frozen_runs/<run id>/scores
#   make unpack-merged-scores PACK_DIR=frozen_runs/<run id>/scores \
#       OUT=data/tmp/merged/merged_scores_rebuilt.parquet
pack-merged-scores:
	@test -n "$(SCORES)" -a -n "$(PACK_DIR)" || { echo "usage: make pack-merged-scores SCORES=<parquet> PACK_DIR=<dir>"; exit 2; }
	uv run python scripts/pack_merged_scores.py pack $(SCORES) $(PACK_DIR) $(PACK_ARGS)

unpack-merged-scores:
	@test -n "$(PACK_DIR)" -a -n "$(OUT)" || { echo "usage: make unpack-merged-scores PACK_DIR=<dir> OUT=<parquet>"; exit 2; }
	uv run python scripts/pack_merged_scores.py rebuild $(PACK_DIR) $(OUT) $(PACK_ARGS)

# The whole merged-approach comparison under one tag: LightGBM (keeping the training
# arrays and saving scores), the LSTM, the TCN, then train-merged on exactly those
# outputs. Each step is the individual target with the tag threaded through, so the
# steps can also be run one at a time (to resume after a failure, run the remaining
# ones: a tagged step refuses to overwrite its own earlier outputs).
#   make train-all RUN_TAG=h30_seed0              # about 2.5 to 3 hours on this data
#   make train-all RUN_TAG=h30_seed0 HORIZON=30   # HORIZON defaults to 30
# Leaves the kept training arrays (several GB) in data/tmp/; `make clean-kept` removes them.
HORIZON ?= 30
train-all:
	@test -n "$(RUN_TAG)" || { echo "usage: make train-all RUN_TAG=<tag>  (e.g. h30_seed0)"; exit 2; }
	$(MAKE) train TRAIN_TAG=$(RUN_TAG) TRAIN_ARGS="--horizon-days $(HORIZON) --keep-work-dir --save-scores"
	$(MAKE) train-sequence SEQUENCE_ARGS="--arch lstm --horizon-days $(HORIZON) --tag $(RUN_TAG)"
	$(MAKE) train-sequence SEQUENCE_ARGS="--arch tcn --horizon-days $(HORIZON) --tag $(RUN_TAG)"
	$(MAKE) train-merged MERGED_TAG=$(RUN_TAG) MERGED_ARGS="--lightgbm-scores data/tmp/lightgbm_scores_h$(HORIZON)_$(RUN_TAG).parquet --lstm-scores data/tmp/sequence_model/scores_h$(HORIZON)_$(RUN_TAG).npz --tcn-scores data/tmp/sequence_model/scores_h$(HORIZON)_tcn_$(RUN_TAG).npz"

# The agent loop (MAPE-K) once per model of a train-merged run, each with the action tiers
# tuned for that model, then the tables. Window: the test split. One snapshot, no clock,
# so it looks ahead and is not point-in-time (runbook Section 4.7); it records a test access.
#   make agent-models REPORT=data/audit/data_quality_reports/merged_evaluation_report_<tag>.json \
#       SCORES=data/tmp/merged/merged_scores_h30_<tag>.parquet OUT_DIR=data/audit/agent_models_<tag>
#   make merged-tables REPORT=... [AGENT_DIR=data/audit/agent_models_<tag>]
agent-models:
	@test -n "$(REPORT)" -a -n "$(SCORES)" -a -n "$(OUT_DIR)" || { echo "usage: make agent-models REPORT=<report.json> SCORES=<scores.parquet> OUT_DIR=<dir>"; exit 2; }
	uv run python scripts/run_agent_models.py $(REPORT) $(SCORES) $(OUT_DIR) $(AGENT_MODELS_ARGS)

merged-tables:
	@test -n "$(REPORT)" || { echo "usage: make merged-tables REPORT=<report.json> [AGENT_DIR=<dir>]"; exit 2; }
	uv run python scripts/report_merged_tables.py $(REPORT) $(if $(AGENT_DIR),--agent-dir $(AGENT_DIR)) $(TABLES_ARGS)

# Search the weights of a train-merged rank blend on VALIDATION scores (no retraining; reads a
# packed score table). Reads no test data unless TUNE_ARGS=--evaluate-test, which spends one
# recorded test look on the recommended weights.
#   make tune-blend-weights PACK_DIR=frozen_runs/<run id>/scores OUT_DIR=data/audit/blend_weights_<tag>
tune-blend-weights:
	@test -n "$(PACK_DIR)" -a -n "$(OUT_DIR)" || { echo "usage: make tune-blend-weights PACK_DIR=<dir> OUT_DIR=<dir>"; exit 2; }
	uv run python scripts/tune_blend_weights.py $(PACK_DIR) $(OUT_DIR) $(TUNE_ARGS)

# A day-by-day replay of a recall-first alerting policy over a packed train-merged score table: a
# WARN threshold set on validation for RECALL_ARGS="--recall-target 0.6" (a catch needs a warning
# at least --lead-min days before the failure), precision-tuned cordon/migrate/drain tiers with
# --hysteresis k consecutive days. One recorded test look. It picks a looser operating point and
# measures its cost; it does not improve the model's ranking.
#   make recall-policy PACK_DIR=frozen_runs/<run id>/scores OUT_DIR=data/audit/recall_policy_<tag>
recall-policy:
	@test -n "$(PACK_DIR)" -a -n "$(OUT_DIR)" || { echo "usage: make recall-policy PACK_DIR=<dir> OUT_DIR=<dir>"; exit 2; }
	uv run python scripts/run_recall_policy.py $(PACK_DIR) $(OUT_DIR) $(RECALL_ARGS)

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

# The MAPE-K loop against the real gold features and trained model, rather
# than agent-demo's two hardcoded drives (src/agent/fleet.py). Execution is
# still simulated. Pick the window with ARGS:
#   make agent-fleet ARGS="--start-date 2026-06-01"                  # inference
#   make agent-fleet ARGS="--start-date 2026-04-01 --end-date 2026-05-31 --cycles 50"
# The second is a backtest on the test split and records a test-split
# access (src/models/access_log.py).
# Refuses to overwrite --out (default data/audit/agent_fleet_run.json): give a
# run on merged scores its own, e.g. ARGS="--scores-path ... --out data/audit/agent_fleet_merged_h30_seed0.json".
agent-fleet:
	uv run python pipelines/run_agent_fleet.py $(ARGS)

# Fixed-date snapshots: the flagging policy measured the way it would run, on
# the drives reporting that day. Use this, not agent-fleet --score, for any
# precision or recall you intend to quote - the latter looks ahead (runbook
# 4.7). Test-split data, so it is recorded in the test-access log.
#   make agent-pit
#   make agent-pit ARGS="--step-days 3 --bootstrap 2000"
agent-pit:
	uv run python pipelines/run_point_in_time.py $(ARGS)

# Point-in-time baseline on VALIDATION drives for the knowledge-base screen
# (docs/agent_knowledge_base.md, Phase 0): 9 weekly snapshots 2026-01-05..03-02 at the
# 60-day horizon (every horizon ends by 2026-05-01, the end of validation), validation
# holdout drives only, so no test access. Four arms: the frozen stacked Stage 2 (also
# cross-checked against its model on 2026-02-23), the LightGBM and Cox Stage 2 of the
# same refit, and the frozen 60-day LightGBM cascade 804a20ce.
PIT_H60 := --horizon-days 60 --start-date 2026-01-05 --end-date 2026-03-02 --step-days 7 \
	--drives-from-split validation --outcome-cutoff 2026-05-01
STACK_DIR := frozen_runs/stack_h60_seed0
STACK_SCORES := data/tmp/stack_scores/scores_h60_seed0.parquet
PIT_OUT := data/audit/agent_pit_validation_h60
agent-pit-validation:
	uv run python pipelines/run_point_in_time.py $(PIT_H60) --label stack_h60_seed0 \
		--scores-path $(STACK_SCORES) --score-column score --model-dir $(STACK_DIR) \
		--cross-check-dates 2026-02-23 \
		--thresholds-path $(STACK_DIR)/agent_thresholds_stack_h60_seed0.json \
		--out $(PIT_OUT)/stack.json --observations-dir $(PIT_OUT)/stack \
		--observations-out $(PIT_OUT)/stack_observations.parquet $(ARGS)
	uv run python pipelines/run_point_in_time.py $(PIT_H60) --label lightgbm_h60_seed0 \
		--scores-path $(STACK_SCORES) --score-column score_lightgbm \
		--thresholds-path $(STACK_DIR)/agent_thresholds_lightgbm_h60_seed0.json \
		--out $(PIT_OUT)/lightgbm.json --observations-dir $(PIT_OUT)/lightgbm \
		--observations-out $(PIT_OUT)/lightgbm_observations.parquet $(ARGS)
	uv run python pipelines/run_point_in_time.py $(PIT_H60) --label cox_h60_seed0 \
		--scores-path $(STACK_SCORES) --score-column score_cox \
		--thresholds-path $(STACK_DIR)/agent_thresholds_cox_h60_seed0.json \
		--out $(PIT_OUT)/cox.json --observations-dir $(PIT_OUT)/cox \
		--observations-out $(PIT_OUT)/cox_observations.parquet $(ARGS)
	uv run python pipelines/run_point_in_time.py $(PIT_H60) --label cascade_804a20ce \
		--scores-path data/tmp/lightgbm_scores_h60_h60_q1q2_q3.parquet --score-column score \
		--thresholds-path frozen_runs/804a20cec10048abb1bc3ccf510f96e1/action_tiers.json \
		--out $(PIT_OUT)/cascade.json --observations-dir $(PIT_OUT)/cascade \
		--observations-out $(PIT_OUT)/cascade_observations.parquet $(ARGS)

# The agent's knowledge store (docs/agent_knowledge_base.md): build it, pack it into the
# tracked deterministic gzip (data/audit/agent_knowledge_h60/agent_knowledge.sqlite.gz, with its
# SHA-256 and content fingerprint in agent_knowledge_pack.json), or restore and verify it.
agent-knowledge:
	uv run python scripts/build_agent_knowledge.py build --overwrite $(ARGS)
	uv run python scripts/build_agent_knowledge.py check

pack-knowledge:
	uv run python scripts/build_agent_knowledge.py pack

unpack-knowledge:
	uv run python scripts/build_agent_knowledge.py unpack $(ARGS)

# Per-family recalibration, fitted on VALIDATION-window snapshots and judged on
# the test snapshots (runbook 4.10). Makes the validation snapshots first, into
# separate files so the test ones are not overwritten.
family-calibration:
	uv run python pipelines/run_point_in_time.py --start-date 2026-02-15 --end-date 2026-03-31 \
		--out data/audit/agent_pit_validation.json \
		--observations-out data/audit/agent_pit_validation_observations.parquet
	uv run python scripts/evaluate_family_calibration.py $(ARGS)

# Figures showing that many alerts the 30-day label calls false were early: the
# drive failed in days 31-60 (runbook 4.9). Needs `make agent-pit` first.
#   make plot-late-hits
#   make plot-late-hits ARGS="--tier migrate --max-lanes 20"
plot-late-hits:
	uv run python scripts/plot_late_hits.py $(ARGS)

dashboard:
	uv run streamlit run src/dashboards/app.py

api:
	uv run python pipelines/run_api.py
