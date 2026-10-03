# Developer Guide

## AI-Based Autonomous System Validation & Reliability Checker
### Predictive Disk-Failure Self-Healing Agent for Server Fleets

**Audience:** engineers reading, extending, or maintaining this codebase.
**Companion documents:** `docs/design_goal.md` (why the system is built this
way), `docs/dataset_strategy.md` (data/feature rules), `docs/project_plan.md`
(phase history). This guide is about the code as it exists today — where
things live, how the pieces fit together, and how to change them safely.
For running the system as an operator rather than modifying it, see
`docs/user_guide.md`.

---

## 1. Highlevel overview

A `FleetSimulator` (or, in production, a real fleet) reports drive state. A
LangGraph **MAPE-K agent** (Monitor → Analyze → Plan → Execute → Validate)
turns that into a p_fail-ranked action proposal. A **guardrail engine**
either lets the proposal through, blocks it (routing to human review), or
leaves it non-destructive. A **reliability checker** scores every decision
(trust score, 0–1, with hard vetoes) and explains it in plain English. A
FastAPI **orchestrator** ties all of this to a persistent approval queue and
audit trail that a Streamlit dashboard renders.

Everything downstream of "predict p_fail" is deliberately swappable: the
agent, guardrails, and reliability checker never call a real ML model or a
real fleet directly — they call whatever `AgentDependencies` you hand them.

---

## 2. Repository layout

```text
configs/            YAML settings, loaded at runtime via src/config.py — not decorative
data_contracts/     Pydantic models shared across every layer (the "wire format")
src/
  ingest/           Bronze landing: Backblaze, SMART-Z, synthetic stub, profiling
  preprocess/       Silver harmonization: identifiers, SMART mapping, gaps, maturity, QC
  features/         Gold feature engineering: windows, derivatives, events, confidence, registry
  labels/           Failure-horizon labeling, event classification, splits, imbalance
  models/           Training, threshold tuning, evaluation, action-tier policy
  agent/            The MAPE-K LangGraph: state, nodes, graph, deps, orchestrator
  guardrails/       Rule catalog, engine, context, operational state, agent adapter
  simulator/        Fleet state machine, action execution + compensation, chaos injectors
  reliability/      Trust score, checks, multipliers, audit, batch resolution, drift
  api/              FastAPI approval/audit service + in-memory store
  dashboards/       Streamlit UI
  config.py         Typed loaders over configs/*.yaml
pipelines/          One script per pipeline stage; each is a `make <target>`
tests/
  unit/             One file per src module, mirroring its name
  integration/      Cross-module: agent+guardrails, agent+simulator, agent+API
  chaos/            Full-stack chaos scenarios + guardrail latency benchmark
  golden/           Synthetic golden dataset + regression smoke test
```

If you're looking for a specific behavior, the module names above are literal
— there's no hidden indirection layer. `src/agent/nodes.py` is exactly the
five MAPE-K node functions; `src/guardrails/rules.py` is exactly the rule
catalog from `configs/guardrails.yaml`.

---

## 3. Environment

```bash
make install      # uv sync --extra dev - installs runtime + dev deps into .venv
make lint          # ruff + mypy (both must be clean; both run in CI too)
make test          # full pytest suite
```

Python 3.11+ is declared in `pyproject.toml`; mypy is configured for 3.12
syntax (`[tool.mypy] python_version`) because some dependency stub files
(numpy) require it — this doesn't affect the actual supported runtime
version, only what mypy parses in stub files.

Never invoke `python`, `ruff`, `mypy`, `pytest`, `streamlit`, or `uvicorn`
directly in this repo — always through `uv run` (or a `make` target, which
already does this). The Makefile learned this the hard way twice: every
target used to call these tools bare (silently failing outside an activated
venv), and separately `install` used to run plain `uv sync` without
`--extra dev`, so a fresh clone's very first `make install && make lint`
would fail with "ruff: command not found."

### 3.1 Every `make` target

| Target | What it does |
|---|---|
| `make install` | `uv sync --extra dev` |
| `make lint` | `ruff check .` + `mypy src data_contracts` |
| `make format` | `ruff format .` then `ruff check --fix .` |
| `make test` | Full pytest suite (unit + integration + chaos + golden + property + smoke) |
| `make test-unit` | `tests/unit/` only |
| `make test-integration` | `tests/integration/` only |
| `make test-chaos` | `tests/chaos/` only |
| `make test-golden` / `make data-check` | `tests/golden/` only (same thing, two names) |
| `make test-property` | `tests/property/` only — Hypothesis property-based tests |
| `make smoke` | `tests/smoke/` only — see §12 |
| `make coverage` | Full suite under `pytest-cov`; writes `htmlcov/index.html` |
| `make ci` | `lint` + `test` — exactly what CI runs |
| `make clean` | Removes caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `htmlcov`, `.coverage`) — never touches data or runtime state |
| `make clean-data` | Removes regenerated pipeline outputs (`data/bronze`, `data/silver`, `data/gold`, `data/audit/*/*`), preserving every `.gitkeep` placeholder — never touches `data/raw/` source data |
| `make download-backblaze` / `download-smartz` | Configurable raw-data download (§5.0) |
| `make ingest-backblaze` / `ingest-smartz` / `ingest-synthetic-stub` | Bronze ingestion (§5) |
| `make build-silver` / `build-features` / `build-labels` | Silver/Gold pipeline stages (§5) |
| `make train` | Trains the model, tunes threshold, computes SHAP importance (§5.1) |
| `make build-sequences` / `train-lstm` | Optional LSTM comparison branch (§5.7) — `train-lstm` requires `uv sync --extra torch` |
| `make score-fleet` | Batch-scores the current fleet with the latest MLflow model; writes `data/audit/predictions/` (§5.8) |
| `make plots` | Renders performance-metric plots (calibration, SHAP, class imbalance, metric comparison) to `data/audit/plots/` (§5.9) |
| `make final-report` | `pipelines/generate_final_report.py` — aggregates chaos/latency/model reports |
| `make agent-demo` | One MAPE-K cycle against the hardcoded demo fleet (§6) |
| `make dashboard` | Streamlit UI (§10) |
| `make api` | FastAPI service (§10) |

`clean`/`clean-data` are intentionally split: `clean` is always safe to run
without thinking; `clean-data` deletes real (if synthetic-derived) pipeline
output, so it's separate and never invoked by anything else automatically.

---

## 4. The configuration system

**Before your first real run**, see `README.md`'s "Configure Before
Running" table for which `configs/*.yaml` settings need a real value
(e.g. `download.backblaze.quarters`) versus which ones are safe defaults.
Nothing needs to change to run `make smoke`/`make agent-demo`/the
synthetic-data walkthrough (§12.1) — this section is about how config
loading works in code, not what to set.

`src/config.py` defines three dataclasses — `AgentSettings`,
`GuardrailSettings`, `ModelSettings` — each with a `.load()` classmethod that
reads the corresponding `configs/*.yaml` file. This is the **only** correct
way to get a threshold into running code. Do not hardcode a second copy of
`configs/agent.yaml`'s `action_thresholds` somewhere else — that was a real
bug this codebase had until it was fixed (thresholds silently diverging from
the YAML).

```python
from src.config import AgentSettings, GuardrailSettings

settings = AgentSettings.load()
settings.action_thresholds        # {"warn": 0.30, "cordon": 0.60, "migrate": 0.80, "drain": 0.90}
settings.min_confidence_for_destructive_action  # 0.80
```

Functions that need config values (`make_plan_node`, `build_guardrail_evaluator`)
accept them as keyword arguments defaulting to `None`, and resolve from
`.load()` lazily inside the function body — never at import time, and never
as a mutable default argument. This keeps tests able to override individual
values without needing the YAML files present, and keeps a stale config
object from being cached across calls.

`GuardrailSettings.prediction_threshold` is deliberately sourced from
`action_thresholds["migrate"]`, not `["drain"]` — it's the lowest p_fail at
which *any* destructive tier can be proposed, and setting it any tighter
would make `PRED_THRESHOLD` block every legitimate migrate action.

---

## 5. Data pipeline (Phases 1–5)

### 5.0 Configurable raw-data download

`src/ingest/download.py` (`make download-backblaze` / `make
download-smartz`, `pipelines/download_backblaze.py` /
`pipelines/download_smartz.py`): Backblaze publishes its quarterly Hard
Drive Stats archives as ZIP files at a stable URL
(`https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_<quarter>.zip`).
**No quarter is ever hardcoded as a real default anywhere in this
codebase** - the time period is entirely up to you, configured either way:

```yaml
# configs/data.yaml
download:
  backblaze:
    quarters: ["Q1_2025", "Q2_2025"]   # an explicit list, or...
    start_quarter: "Q1_2025"           # ...an inclusive range (used only
    end_quarter: "Q4_2025"             # if `quarters` is empty)
```

or per-run without touching the config file:

```bash
make download-backblaze ARGS="--quarters Q1_2025 Q2_2025"
make download-backblaze ARGS="--start-quarter Q1_2025 --end-quarter Q4_2025"
```

`--quarters`/`quarters` always wins over the range if both are given.
`generate_quarter_range`/`parse_quarter`/`format_quarter` implement the
range expansion (e.g. `("Q3_2025", "Q2_2026")` ->
`["Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]`), and `resolve_quarters`
is the single place that merges CLI args and config, raising a clear
error if neither a list nor a range is configured, rather than silently
downloading nothing. Downloads are streamed to disk in fixed-size chunks
(never held fully in memory) and are idempotent per quarter - a
`.{quarter}.downloaded` marker file skips a quarter that's already
present, and if the ZIP itself is already on disk (e.g. an extraction
that got interrupted) it's reused rather than re-fetched over the
network. **These are multi-hundred-MB-to-multi-GB downloads that expand
to many GB uncompressed** - make sure you have disk headroom before
running this against more than one or two quarters.

SMART-Z has no public bulk-download API (docs/dataset_strategy.md
section 3.2); `download_smartz` requires a directly-configured
`download.smartz.url` (set only once you've requested access) and raises
a clear, actionable `ValueError` otherwise instead of silently doing
nothing - the same "no real code path defaults to anything you haven't
configured" principle.

`make ingest-backblaze` has an analogous, independent restriction:
`sources.backblaze.start_date`/`end_date` (or `--start-date`/`--end-date`)
limits which already-downloaded raw CSVs (named `YYYY-MM-DD.csv`) get
ingested, without touching what's on disk in `data/raw/backblaze/`. This
is separate from which quarters you've downloaded - useful for bounding
how much data a single `make build-silver` run has to handle (e.g. while
validating against a small slice before scaling up), or for re-ingesting
just a subset after a schema change. Unset (the default) ingests every
CSV `raw_glob` matches, same as before this option existed:

```yaml
# configs/data.yaml
sources:
  backblaze:
    start_date: "2026-01-01"
    end_date: "2026-01-31"
```

```bash
make ingest-backblaze ARGS="--start-date 2026-01-01 --end-date 2026-01-31"
```

```text
make ingest-backblaze / ingest-smartz / ingest-synthetic-stub   → data/bronze/
make build-silver                                               → data/silver/
make build-features                                              → data/gold/features/
make build-labels                                                 → data/gold/labels/
make train                                                        → MLflow run + data/audit/.../model_evaluation_report.json
make build-sequences / train-lstm (optional, §5.7)              → data/gold/sequences/ + lstm_evaluation_report.json
```

Each stage is a thin `pipelines/*.py` script that reads YAML config, calls
into `src/<stage>/`, and writes Parquet + a JSON report under `data/audit/`.
The actual logic lives in `src/`, is unit-tested independently of the
pipeline scripts, and never touches disk paths directly (paths are passed
in) — so you can call `build_silver(...)`, `compute_labels(...)`, etc. from
a notebook or a different orchestrator without going through `pipelines/`.

**Leakage rules are enforced structurally, not by convention.** Every
rolling/window operation partitions by `drive_id` and orders by `date`
(`src/features/windows.py`, `derivatives.py`, `events.py`); labels only look
forward from `date` to `date + horizon_days` (`src/labels/labeling.py`); an
ambiguous or right-censored outcome produces `label = None`, never a
default `0` (`src/labels/labeling.py::compute_labels_for_horizon`). If
you're adding a new feature or label rule, follow the same
partition-by-drive_id / no-future-data pattern — the golden dataset test
(`tests/golden/`) and leakage-focused unit tests
(`tests/unit/test_labels.py`) are there to catch regressions, but they can't
catch a leakage bug in logic they don't exercise, so add a test alongside
any new rule.

**Gold labels schema validation** (`src/labels/schema_validation.py`,
docs/project_plan.md Phase 1 "Data validation: Pandera with Polars
backend, or native Polars schema checks + pytest"): `pandera` was a
declared but entirely unused dependency — `src/preprocess/
quality_checks.py` already covers the "native Polars checks" half of
that either/or for bronze/silver invariants (null dates, duplicate
drive-day-attribute rows, capacity, failure-date ordering), but nothing
used Pandera itself, and nothing validated the gold label table's schema
at all. `validate_gold_labels` runs a `pandera.polars.DataFrameSchema`
(dtypes, nullability, and `isin` checks for `label`/`split`/
`split_strategy`/`horizon_days`) with `lazy=True`, so a violation raises
`pandera.errors.SchemaErrors` listing every failing check at once, not
just the first. `pipelines/build_labels.py` calls it right before writing
the label Parquet file. Writing this schema caught a real, if harmless,
inconsistency: `compute_labels_for_horizon` built `horizon_days` via
`pl.lit(horizon_days)`, which Polars infers as `Int32` for a bare Python
int, while every other integer column in the table used `Int64` -
`src/labels/labeling.py` now pins it to `Int64` explicitly.

**SMART-Z is an external validation split, never train/validation/test**
(`src/labels/splits.py::apply_vendor_holdout`, `docs/dataset_strategy.md`
section 9.3/9.4): `pipelines/build_labels.py` runs
`add_chronological_split` → `apply_drive_level_holdout` →
`apply_vendor_holdout`, and that last call unconditionally reassigns every
row whose `source_dataset == "smartz"` to `split="external_smartz"` /
`split_strategy="vendor_holdout"`, overriding whatever the chronological
split computed for it. This exists to measure Backblaze-specific
overfitting and cross-vendor generalization, which is only a meaningful
test if SMART-Z never leaks into training. On a Backblaze-only build no row
has `source_dataset == "smartz"`, so the reassignment matches nothing and
the split is effectively untouched; the function is also a structural
no-op if `source_dataset` isn't present on the frame at all (e.g. a
hand-built test fixture).

**Cross-vendor SMART harmonization** (`src/preprocess/smart_mapping.py`):
SMART attribute IDs are standardized by spec (ID 5 is always "Reallocated
Sector Count"), but the *column name* a source uses for "the raw value of
attribute 5" varies — Backblaze uses `smart_5_raw`, SMART-Z is expected to
use `smart_5_normalized` (`SOURCE_COLUMN_TEMPLATES`, best-effort until real
SMART-Z files are ingested — verify and adjust the template then).
`melt_smart_attributes` melts each `source_dataset` value with its own
template before concatenating, so a mixed Backblaze+SMART-Z Bronze frame
harmonizes into the same canonical attribute names — this is what makes the
cross-vendor generalization objective (`docs/dataset_strategy.md` section 2)
actually reachable once real SMART-Z data exists, rather than SMART-Z rows
silently having no SMART columns after harmonization.

**Feature Family E — cross-vendor ratios and model-family z-scores**
(`src/features/cross_vendor.py`, `docs/dataset_strategy.md` section 10.5):
`add_attribute_ratios` computes vendor-agnostic ratios
(`pending_to_reallocated_ratio`, `reallocated_per_capacity`,
`uncorrectable_per_power_on_hour`) that normalize away absolute-count
differences in reporting granularity across sources; each ratio is a no-op
unless both of its input columns are present, so it degrades gracefully for
whichever priority SMART attributes a given build actually has.
`add_model_family_zscores` re-expresses each attribute's `{window}d_mean`
rolling feature as a z-score relative to its own `model_family` (not the
whole fleet), because "some drive models naturally report higher error
counts" — a drive that is unremarkable for its model should not look
anomalous just because other models run cleaner. Both are wired into
`pipelines/build_gold_features.py::build_gold_features` right after
`add_lifecycle_features`, and both are registered in
`src/features/registry.py::build_registry` for auditability like every
other feature family.

**Slope is a secant approximation, not least-squares** (`src/features/derivatives.py`):
`slope_W = delta_W / W`. This is a documented, deliberate simplification for
speed; if validation ever shows it matters, replace it with a proper rolling
regression — the function signature won't need to change.

**Real data gap:** the ingestion/silver/gold/label pipelines are fully
implemented and tested against synthetic fixtures, but nobody has run them
against real Backblaze/SMART-Z archives in this repo. `make ingest-backblaze`
and `make ingest-smartz` print a clear message and exit if
`data/raw/{backblaze,smartz}/` is empty. `make train` fails with a clear
`ValueError` (not a cryptic LightGBM stack trace) if the assembled training
frame has zero non-censored rows for the primary horizon; against the
synthetic stub (`src/ingest/synthetic_stub.py`, §12.1) that no longer
happens by default - the stub now spans multiple years with real degrading
failure trajectories, so `make train` genuinely completes end-to-end
offline. The error path itself is still real and still worth knowing about
(e.g. if you narrow `configs/model.yaml`'s split boundaries against a
small real dataset and end up with an empty split).

**Model selection: LightGBM or XGBoost** (`configs/model.yaml`'s
`model.type`, docs/design_goal.md/docs/project_plan.md "ML models: XGBoost
/ LightGBM"): `xgboost` was a declared but entirely unused dependency, and
`model.type` in the config was read nowhere - `pipelines/train_model.py`
always trained LightGBM regardless of what it said. `model.type: xgboost`
now trains via `src/models/xgboost_training.py::train_xgboost`, which
computes `scale_pos_weight = negative_count / positive_count`
(docs/dataset_strategy.md section 15's XGBoost-specific imbalance
mitigation, the counterpart to LightGBM's `is_unbalance`) directly from
the training labels rather than requiring it as a param. Everything
downstream (`predict_proba_positive`, SHAP's `TreeExplainer`, MLflow
logging, the model card) works with either model type; the Optuna search
(§5.5) remains LightGBM-only and raises a clear error if enabled together
with `model.type: xgboost`, rather than silently ignoring the setting.

**Logistic Regression sanity baseline**
(`src/models/logistic_regression_baseline.py`, docs/project_plan.md Phase
6 Model Candidates: "Logistic Regression | Interpretable sanity
baseline"): `pipelines/train_model.py` always trains this alongside the
primary model (never config-gated - it's cheap and central to what a
"sanity baseline" is for) and logs its validation/test AUPRC to MLflow
and the model card. It standardizes features first via a
`StandardScaler` (logistic regression is scale-sensitive, unlike the
tree-based models, and the gold feature columns span very different raw
ranges) and uses `class_weight="balanced"` for the same imbalance the
tree models handle via `is_unbalance`/`scale_pos_weight`. It is never
used for production decisions - its only job is letting a training run
confirm the primary model is actually beating a simple linear one,
instead of assuming so.

**Subsampled SMOTE comparison** (`src/models/smote.py`,
docs/dataset_strategy.md section 15 Imbalance Mitigations #3: "SMOTE may
be used, but only on training data and only if it improves AUPRC...
apply SMOTE only to a stratified training sample; never apply SMOTE to
validation or test data; compare against class weighting; prefer class
weighting if SMOTE does not clearly improve validation performance").
`imbalanced-learn` wasn't even a declared dependency before this.
`apply_smote` stratified-subsamples the training partition (SMOTE's
nearest-neighbor search is the expensive part at scale) and then
oversamples the minority class to parity. Gated by
`configs/model.yaml`'s `smote_comparison.enabled` (default `false`):
when enabled, `pipelines/train_model.py` trains a **second, comparison-
only** model on the SMOTE-resampled data (with class weighting turned
off, since the data is already balanced) and reports both validation
AUPRCs plus a `recommendation` field side by side - it never replaces the
primary (class-weighted) model automatically, matching the docs' explicit
default-to-class-weighting guidance.

### 5.1 Explainability (SHAP)

> **Configurable, off by default.** `make train` only computes SHAP when
> `configs/model.yaml` has `diagnostics.shap_enabled: true`. On real fleet
> data it takes ~30 minutes and can fail SHAP's own additivity check when
> the model has extreme leaf values (seen with a raw score near -10,000,000).
> When disabled, no `shap_feature_importance.json` is written (a stale one
> is deleted), the model card's Explainability section says SHAP was
> skipped, and `make plots` warns that the SHAP report is missing.

`src/models/explainability.py` wraps `shap.TreeExplainer` (exact and fast
for LightGBM) around the trained model:

```python
explainer = build_explainer(model, background)         # background: a sample of training rows
shap_values = compute_shap_values(explainer, x_val)     # (n_rows, n_features)
ranking = global_feature_importance(shap_values, feature_columns)  # sorted, mean |SHAP|
explanation = top_contributing_features(shap_values[i], feature_columns, k=5)  # one row
```

`pipelines/train_model.py` calls this after training: it builds the
explainer from a 100-row random sample of the training split (fast,
representative), computes SHAP values over the validation split, and writes
the top-20 global ranking to both `data/audit/data_quality_reports/
model_evaluation_report.json` (under `shap_global_feature_importance`) and a
standalone `shap_feature_importance.json`, plus logs the latter as an MLflow
artifact. This is the *global* (training-time) explainability story.

**Not yet wired**: `top_contributing_features` (the *per-prediction*
explainer) isn't called anywhere at serving time — the agent's lean-state
design (§6.1) keeps bulk feature vectors out of `AgentState`, so a real
integration would need its own feature-vector lookup path (e.g. keyed by
`drive_id` + `as_of` timestamp against the online feature store described in
`docs/dataset_strategy.md` §20), not a field threaded through the graph.
`data_contracts.schemas.PredictionOutput.top_contributing_features` exists
and is ready to receive this once that lookup path is built.

### 5.2 Model card

`src/models/model_card.py` (docs/project_plan.md Phase 6 Key Task 7 /
Deliverables "Model card"): `build_model_card` assembles model details,
intended use/out-of-scope, training data provenance (feature registry
version, dataset version — see §5.3 below), the threshold policy and
validation/test metrics, the SHAP top-20 global ranking, and known
limitations into one dict; `render_model_card_markdown` renders it to a
human-readable Markdown file. `pipelines/train_model.py` writes both
`data/audit/model_cards/v{model_version}_h{horizon}d.{json,md}` and logs
them as MLflow artifacts on every training run, so a reviewer can read one
file to understand what a specific model version is and is not validated
for, rather than reconstructing that from scattered metrics and params.
`dataset_version` reads `"unversioned"` if `make build-labels` was never
run (no version record exists yet to read back).

### 5.3 Dataset versioning

`src/labels/dataset_version.py` (docs/dataset_strategy.md section 4,
`audit/dataset_versions/`): every `make build-labels` run writes an
immutable, timestamped record — `version_id` (a UTC timestamp), the
feature-registry and model-config versions used, and a `sha256` content
hash + row count + date range + per-split row counts for the gold
features/labels files that went into it — to
`data/audit/dataset_versions/{version_id}.json`. `latest_dataset_version`
reads the most recent one back (`version_id` sorts chronologically);
`pipelines/train_model.py` calls it to stamp the model card with the
dataset version actually used, so "what data trained this model" is
answerable from a file, not from memory of when the pipelines were last
run.

### 5.4 Secondary evaluation metrics

`src/models/evaluation.py` (docs/dataset_strategy.md section 15
"Secondary metrics"), beyond AUPRC and the confusion-matrix-derived
precision/recall:

- **Calibration** (`compute_calibration`): bins predictions into 10
  equal-width buckets over `[0, 1]` and compares each bucket's mean
  predicted probability to its observed fraction of positives.
  `expected_calibration_error` is the bin-count-weighted mean absolute
  gap; `brier_score` is the mean squared error of the raw probabilities.
  Folded into `evaluate_at_threshold`'s return dict under `"calibration"`.
- **Precision at top-K** (`precision_at_k` / `precision_at_k_fractions`):
  out of the K highest-scored drive-days, what fraction actually failed —
  a ranking-quality metric independent of the chosen operating threshold.
  `evaluate_at_threshold` includes it for the top 1%/5%/10% of whatever
  population it's called on.
- **Warning lead time** (`compute_warning_lead_time_days`): for drives
  with a genuine failure event (`event_type` in
  `src/labels/event_types.py::FAILURE_EVENT_TYPES` — `pipelines/
  train_model.py` filters the test split to these before calling), the
  number of days before the actual failure that the score first crossed
  the operating threshold, aggregated as mean/median/min/max plus
  `warning_coverage` (the fraction of failed drives that got any warning
  at all). Reported separately from `evaluate_at_threshold` because it
  needs per-drive-day `days_to_event`/`event_type`, not just
  `(y_true, y_scores)` arrays.

All three are written to `model_evaluation_report.json` and surfaced in
the model card (§5.2).

### 5.5 Optuna hyperparameter search

`src/models/hyperparameter_tuning.py` (docs/project_plan.md Phase 6
"Threshold tuning: Use Optuna. Subsample for search. Retrain best
configuration on full training partition."; RAM Practices "Use subsampled
Optuna search."): `optuna` was a declared but entirely unused dependency.
`tune_lightgbm_hyperparameters` runs a `TPESampler` study over a
configurable `search_space` (`{param: [low, high]}`; an all-int range
samples `suggest_int`, otherwise `suggest_float`), training each trial on
a random *subsample* of the training partition and scoring AUPRC on the
*full*, never-subsampled validation split, so the objective reflects real
generalization rather than a subsample artifact. It never retrains on the
full partition itself - `pipelines/train_model.py` does that with the
returned `best_params`, per the doc's explicit "retrain on full partition"
instruction. Gated by `configs/model.yaml`'s `hyperparameter_search.enabled`
(default `false`, so `make train` stays fast/deterministic); when enabled,
the search result (`best_params`, `best_value`, trial count) is written to
`model_evaluation_report.json` and the tuned params are logged to MLflow
under a `tuned_` prefix.

### 5.6 Structured logging

`src/logging_config.py` (docs/design_goal.md / docs/project_plan.md
"Logging / audit: structlog / JSON logs" — "Per-decision rationale,
guardrail status, explainability... query-friendly, append-only"):
`structlog` was a declared but entirely unused dependency — every pipeline
script and `src/agent/demo.py` used plain `print()`, which can't be
filtered, queried, or shipped to a log aggregator as structured events.
`configure_logging(json_output=False)` sets up the process-wide structlog
pipeline (console renderer by default, for interactive `make` runs;
`json_output=True` renders newline-delimited JSON, the audit-friendly
format for a real deployment); `get_logger(__name__)` returns a bound
logger. Every `pipelines/*.py` entry point calls `configure_logging()` at
the top of `main()` and replaces its `print(...)` calls with
`logger.info(event_name, **fields)` / `logger.warning(...)`; so does
`src/agent/demo.py`, and `AgentOrchestrator._handle_result`'s slow-cycle
signal (§6.4) is now `logger.warning("mapek_cycle_exceeded_target", ...)`
instead of a bare print.

### 5.7 Optional LSTM / sequence branch

docs/dataset_strategy.md section 10.7 (Feature Family G) and section 16.2
describe an **optional** deep-learning comparison branch, explicitly
"compared against the tree-based baseline, not assumed to be superior."
It is intentionally kept out of the default install and the primary
training pipeline:

- `src/features/sequences.py` (pure Polars/NumPy, no new dependency):
  `build_sequence_tensors` turns the gold wide-features table into one
  `[time_steps, features]` sequence per drive (its last `time_steps` days
  of the configured attributes, sorted by date; a drive with less history
  is left-padded with zeros rather than dropped, since a young drive is
  exactly the case early signal matters most for). `compute_time_decay_
  weights`/`apply_time_decay_weights` implement the doc's `weight_t =
  exp(-lambda * age_in_days)` formula directly on the tensor, damping
  older days before the sequence ever reaches a model.
  `write_sequence_tensors`/`load_sequence_tensors` persist it as a
  memory-mapped `.npy` file (never a large in-memory array, per the doc's
  storage recommendation) plus a small Parquet/JSON metadata sidecar.
  `make build-sequences` (`pipelines/build_sequences.py`) writes
  `data/gold/sequences/`, configured by `configs/features.yaml`'s
  `sequences:` section.
- `src/models/lstm.py` **requires `uv sync --extra torch`** (`torch` is a
  `pyproject.toml` optional extra, not part of the default install) and
  is never imported by `pipelines/train_model.py` or anything else in the
  default dependency closure - only by its own entry point. `SequenceLSTM`
  is a small `nn.LSTM` + dropout + linear classifier; `train_lstm` uses
  `BCEWithLogitsLoss(pos_weight=...)` for the same class imbalance the
  tree models handle via `is_unbalance`/`scale_pos_weight`.
- `make train-lstm` (`pipelines/train_lstm.py`) joins the sequence
  tensors back to the gold labels for the primary horizon (by
  `drive_id`+`as_of_date`), applies the time-decay weights, trains on the
  `train` split, tunes a threshold and evaluates on `validation`/`test`
  exactly like the primary pipeline, and writes
  `data/audit/data_quality_reports/lstm_evaluation_report.json` with a
  `recommendation` field comparing its test AUPRC against the primary
  model's (`model_evaluation_report.json`, if one exists) - it never
  replaces the primary model.
- `tests/unit/test_lstm.py` guards its `torch` import with
  `pytest.importorskip`, so `make test` (the default dev install) skips
  it cleanly rather than failing; it only actually runs once `torch` is
  installed. `tests/unit/test_sequences.py` has no such guard since
  `src/features/sequences.py` has no torch dependency.

### 5.8 Batch fleet scoring — where `PredictionOutput`/`ActionProposal` are actually used

`data_contracts.schemas.PredictionOutput`, `ActionProposal`, and
`DecisionAuditRecord` were declared but never imported anywhere in real
code. The reason isn't an oversight: `PredictionOutput.feature_confidence`
is a full `FeatureConfidence` object requiring
`telemetry_coverage_30d`/`hours_since_last_telemetry`/
`attribute_coverage_factor`, and the live MAPE-K loop's `AgentState` is
*deliberately* lean (§6.1, docs/design_goal.md section 5.6) - it never
carries those fields, by design, so LangGraph checkpoints stay small.
Forcing the live loop to populate these types would mean reversing that
design decision.

Instead, `src/models/serving.py` + `pipelines/score_fleet.py`
(`make score-fleet`) is a genuinely new, standalone capability where the
full contract applies for real: an offline batch report, independent of
the live agent, that has the *entire* gold-feature row on hand (not a
lean cycle state).

- `score_latest_drive_day` takes each drive's most recent gold-feature
  row (`latest_row_per_drive`, joined with `join_feature_maturity` since
  `feature_maturity` lives only in Silver's `drive_metadata`, never the
  Gold features table) plus a trained model, and returns one validated
  `PredictionOutput` per drive - `FeatureConfidence` is built from real
  columns the gold-feature pipeline already computes
  (`src/features/confidence.py`, `src/preprocess/feature_maturity.py`),
  not fabricated data.
- `propose_actions` runs the same `determine_action_tier` policy the live
  agent uses over each `PredictionOutput`, producing an `ActionProposal`
  (with a real rationale string and the full `PredictionOutput` embedded)
  for anything above MONITOR.
- `pipelines/score_fleet.py` loads the latest MLflow run for the
  configured experiment (`mlflow.lightgbm.load_model`/
  `mlflow.xgboost.load_model`, matching whichever flavor
  `pipelines/train_model.py` logged), scores the current fleet, and
  writes both lists to `data/audit/predictions/{date}.json`.
- `DecisionAuditRecord` remains genuinely unused in the live loop - this
  is the considered decision, not an unfixed gap. Its docstring in
  `data_contracts/schemas.py` explains why and points to this section;
  `InMemoryAuditStore` (`src/api/store.py`) persists the live loop's
  equivalent information as a plain dict, and `DecisionAuditRecord`
  documents that dict's fully-specified shape as a schema reference.

### 5.9 Performance-metric plots

`src/reporting/plots.py` (`make plots`, `pipelines/generate_performance_plots.py`):
a pure visualization layer over the JSON reports `make train`/`make
build-labels` already write — it computes nothing new, it just renders
what's already in `data/audit/data_quality_reports/`. Uses matplotlib's
headless `"Agg"` backend (set before `pyplot` is imported) since this
always runs from a script, never an interactive session with a display.

| Plot | Source report | Output |
|---|---|---|
| `plot_calibration_curve` | `model_evaluation_report.json`'s `validation_metrics`/`test_metrics.calibration` | `calibration_validation.png` / `calibration_test.png` — reliability diagram (mean predicted probability vs. observed fraction of positives per bin) |
| `plot_metric_comparison` | same, `.auprc`/`.precision`/`.recall`/`.false_positive_rate`/`.false_negative_rate` | `metric_comparison.png` — validation vs. test bars |
| `plot_precision_at_k` | same, `.precision_at_top_{K}pct` | `precision_at_k_test.png` |
| `plot_shap_feature_importance` | `shap_feature_importance.json` | `shap_feature_importance.png` — top-20 global SHAP bar chart |
| `plot_class_imbalance` | `label_imbalance_report.json`'s `class_distribution` | `class_imbalance.png` — failure rate per split, grouped by horizon |

Each plot's data comes from a report `pipelines/generate_performance_plots.py`
checks for individually; a missing report (e.g. `make plots` run before
`make train`) logs a clear warning and skips just that plot rather than
failing the whole run. Plots are written to `data/audit/plots/`
(gitignored like the rest of `data/audit/*/*`) and are safe to regenerate
at any time — nothing else reads them back.

---

### 5.10 Memory: the `build_silver` peak-RAM bug, and the process-wide cap

`make build-silver` crashed with an out-of-memory kill against real
fleet-scale Backblaze data (32GB RAM), even though the same pipeline was
fine against the synthetic fixtures. Two independent things were wrong,
and both matter for anyone touching a pipeline that scans `data/bronze/`:

**The bug (fixed in `build_silver()`, `pipelines/build_silver.py`).**
The original implementation (a) read every Bronze Parquet file eagerly
with `pl.read_parquet` and concatenated them into one in-memory
`DataFrame` before doing anything else, and (b) kept every intermediate
`DataFrame` (`wide`, `drive_day`, `canonical_long`, ...) alive
simultaneously as it worked through Silver's normalize → derive → gap
detection → melt steps. Real fleet data has enough distinct
`(drive_id, date)` rows that holding two or three full copies of it in
memory at once is what actually exhausted 32GB — the synthetic fixture
(a handful of drives, ~1 year) never got big enough to expose it, and
even the fixed-up eager version still didn't survive a real ~30M-row
Backblaze export: a multi-column sort keyed partly on the string
`drive_id` (row-encoding a string column for `arg_sort_multiple` is
expensive) crashed first, and once that was fixed, just holding the
fully-materialized `drive_day` table (or `canonical_long`, ~10x bigger
after `melt_smart_attributes` unpivots to one row per
drive-day-*attribute*) in memory at fleet scale crashed next regardless
of how cheaply it was computed.

The real fix ended up being structural, not just an eager-vs-lazy
cleanup: `build_silver` now never materializes either `drive_day` or
`canonical_long` as an in-memory `DataFrame` at all, AND `canonical_long`
no longer carries every drive-day column through the melt.
- The scan → normalize → failure-date → telemetry-gap chain is built as
  one `LazyFrame` and sunk straight to Parquet (`data/silver/drive_day/`,
  now a real, persistent Silver output, not a temp file) via
  `LazyFrame.sink_parquet`, which streams the computation *and* the
  output in row-group-sized batches — unlike `.collect(engine="streaming")`,
  which still has to land the whole result in memory as one `DataFrame`
  once it's done, `sink_parquet` never does.
- `build_drive_metadata` then `scan_parquet`s that file lazily rather
  than taking `drive_day` as an in-memory argument (it now accepts either
  a `DataFrame` or a `LazyFrame` for exactly this reason).
- `melt_smart_attributes` is called with `id_columns=["drive_id", "date"]`
  explicitly (its default, `id_columns=None`, still carries every
  non-SMART column through, for callers that want that - e.g. tests).
  Carrying all ~17 of `drive_day`'s other columns (drive_model,
  capacity_gb, telemetry-gap flags, ...) through the melt would duplicate
  every one of them once per SMART attribute for no reason, since none of
  them vary by attribute - at fleet scale that's a genuinely large,
  entirely avoidable cost on top of the row expansion itself. The melt's
  output (just `drive_id`/`date`/the attribute columns) is sunk straight
  to `canonical_telemetry/part.parquet`.
- `src/features/pivot.py::pivot_badness_wide` (used by `make
  build-features`) now takes an optional `drive_day` argument and
  left-joins its other columns back in *after* pivoting - i.e. at the
  drive-day grain again, not the ~10x-larger melted one, where the join
  is cheap. It explicitly excludes the raw bronze `smart_<id>_raw`/
  `_normalized` columns from that join (`ALL_BRONZE_SMART_COLUMNS` in
  `src/preprocess/smart_mapping.py`) - `drive_day` still has them (the
  melt only reads it, never mutates it), but they're superseded there by
  the harmonized `smart_badness_value` columns the pivot itself already
  produces; a first version of this join didn't exclude them and
  silently reintroduced them into `gold`, caught only by diffing the
  final feature table against the pre-split implementation's output.
  `pipelines/build_gold_features.py` reads `drive_day/part.parquet`
  alongside `canonical_telemetry/part.parquet` and passes it through.
- Only `drive_metadata` (one row per drive) is still ever a real
  in-memory `DataFrame` — it's inherently tiny regardless of fleet size,
  so there's nothing to gain by streaming it too.
- Compute-side, `compute_telemetry_gaps` and `pivot_badness_wide` both
  sort by `date` alone now, not `[drive_id, date]` — a global sort by
  `date` alone still guarantees every drive's own rows land in
  non-decreasing date order (all `.over("drive_id")`/
  `.rolling(group_by="drive_id")` window features need), without ever
  row-encoding the string `drive_id` column into the sort key.
- Quality checks (`run_all_checks`) run against `drive_day`, not
  `canonical_long`, since `capacity_gb`/`failure_date` only exist at the
  drive-day grain now - checking `drive_day` for duplicate
  `(drive_id, date)` rows is equivalent to (and cheaper than) checking
  `canonical_long` for duplicate `(drive_id, date, smart_attribute_name)`
  rows, since a duplicate drive-day row produces duplicates for every
  melted attribute.

Every one of these changes was checked for exact output equivalence
before being kept: the new implementation's `drive_day`,
`canonical_telemetry`, `drive_metadata`, quality-check results, and
finally the full `make build-features` output (`gold`) were diffed
directly against the pre-restructure implementation on the synthetic
fixture and found identical, aside from `ingested_at` (a real wall-clock
ingestion timestamp, expected to differ between separate runs) -
`tests/unit/test_pipelines_build_silver.py`,
`tests/unit/test_preprocess.py`'s lazy/eager quality-check and
narrowed-`id_columns` tests, `tests/unit/test_features.py`'s
scrambled-input and drive_day-join-back pivot tests. There's a
regression test for the Bronze-accumulation behavior the pipeline
depends on in `tests/unit/test_ingest_common.py`; there's no dedicated
peak-memory test (polars/pytest don't make that cheap to assert), so if
`build_silver` or `pivot_badness_wide` are touched again, re-check that
no step re-introduces a full materialization of `drive_day` or
`canonical_long`, or reintroduces a column into `gold` that wasn't there
before.

**The cap (`src/resource_limits.py`, new).** Fixing the one known bug
doesn't rule out a different pipeline, a bigger fleet, or a future
regression doing the same thing — so every pipeline and service entry
point now also applies a hard, configurable ceiling on its own process's
memory at startup, as a last line of defense: a bug degrades into a clean
process abort instead of silently swapping the host to a crawl or taking
down other processes. Configured by `configs/data.yaml`:

```yaml
resource_limits:
  max_memory_gb: 20   # null (or omit the whole block) disables the cap
  feature_batch_target_rows: 2000000
```

**Batched feature computation (`make build-features`).** The gold feature
table is ~196 columns wide, which works out to ~950 bytes per row — about
10GB for a single month of real Backblaze data (10.46M drive-days), before
counting the intermediates each feature family allocates on top of it. No
amount of join hygiene makes that fit under the cap, so
`pipelines/build_gold_features.py` does not build it as one frame:

1. `pivot_badness_wide` produces the wide (~22 column) frame as before.
2. `_write_wide_batches` splits it into batches of whole drives — batch
   membership is `hash(drive_id) % n_batches`, so every row of a given
   drive lands in exactly one batch — and spills each batch to its own
   temp Parquet file, then frees the pivoted frame.
3. Each batch is read back and run through `build_gold_features`
   independently, and its result is written to a temp Parquet file and
   freed before the next batch starts.
4. The output file is assembled with a `pyarrow.parquet.ParquetWriter`,
   one batch per row group, so the finished table is never held in memory
   — not even once, at the end.

**Why batching alone wasn't enough, and each batch is its own process.**
Splitting into batches of whole drives bounds how much *live* data any one
step holds — but on real data, even a single ~1.75M-row batch (1/6 of the
fleet) started crashing partway through its own feature computation, well
under what that data volume should need. The reason: Polars is built on
jemalloc (confirmed via `strings` on the compiled extension —
`_rjem_je_*` symbols, from the `tikv-jemallocator` crate), which on 64-bit
Linux defaults to *retaining* freed virtual memory for reuse instead of
returning it to the OS. `madvise(MADV_DONTNEED)` drops the physical pages
(so RSS goes down), but the address-space mapping itself stays reserved.
`RLIMIT_AS` — the cap this pipeline *used to* run under (now RSS-based, see below) — constrains
mapped address space, not resident memory, so within a single process it
tracks the *high-water mark of everything that process has ever
allocated*, not what's currently live. `del` and `gc.collect()` free the
data but never lower that high-water mark. This is why every earlier
round of narrowing joins and freeing intermediates bought progress to the
next pipeline stage without ever eliminating the crash outright — the
cap wasn't tracking the size of any one step, it was accumulating across
the whole run.

The only thing that actually resets it is a process boundary: exiting a
process unconditionally unmaps its entire address space, regardless of
what the allocator inside it was retaining. So `_stage_prepare` (the
canonical_long/drive_day read + pivot + batch split) and `_stage_batch`
(one call per batch) each run in their own subprocess, invoked via
`_run_stage` — `main()` re-invokes this same script with `--stage ...`.
The top-level orchestrator (`main()` with no `--stage`) never itself
touches a large Polars frame, so its own address space stays flat
regardless of how many batches there are, and each batch subprocess
starts from a genuinely clean slate rather than inheriting whatever the
previous batch's allocator was still holding onto.

Batching on whole drives is what makes this invisible to the result:
every feature family except one is computed strictly per drive
(`.rolling(index_column="date", group_by="drive_id")` in
`src/features/windows.py` and `src/features/events.py`,
`.over("drive_id")` in `src/features/derivatives.py` and
`src/features/lifecycle.py`, row-wise arithmetic in
`src/features/confidence.py`), so a drive's features depend only on rows
that are all present in its own batch.

The exception is `add_model_family_zscores`, which compares a drive to
every other drive of the same `model_family`. That one is split in two:
`model_family_zscore_stats` aggregates the per-family mean/std off the
batch files (one row per model family, so nothing needs materializing),
and `add_model_family_zscores_from_stats` applies them per batch while
the output file is being written. It joins with `nulls_equal=True`
because `.over("model_family")` treats a null family as its own group
rather than as unmatched — `test_model_family_zscores_from_global_stats_match_the_whole_fleet_computation`
pins that equivalence, including a null family and a single-row family
(whose std is null, not `0.0`).

Verified against the pre-batching implementation on the synthetic
fixture: all 181 non-z-score columns are bit-identical, and the output is
byte-for-byte identical across different batch counts (7 batches vs 3 vs
1), i.e. the batch size genuinely does not affect the result. The 15
z-score columns differ by at most 1 ULP of float32 (~1.2e-7 relative),
because a per-family `group_by` aggregate accumulates its sum in a
different order than `.over("model_family")` does — the same 1-ULP
difference appears with a single batch, so it comes from the aggregation
form, not from batching.

`feature_batch_target_rows` (default 2,000,000 — roughly 2GB of feature
table per batch) is the knob: lower it if `make build-features` still
hits the cap, raise it for fewer, larger batches.

`apply_memory_limit_from_config()` reads this value and starts a daemon
watchdog thread that samples the process's **resident** memory (RSS, from
`/proc/self/statm`) every 0.25s and stops the process — logging
`memory_limit_exceeded`, exit code 86 — once RSS exceeds the cap. It's
called as the first thing in every pipeline's `main()` (right after
`configure_logging()`), so the limit is in force before any data is
loaded. Things to know before relying on it:

- **Why RSS and not `RLIMIT_AS`.** The cap used to be `RLIMIT_AS`
  (`ulimit -v`), which limits *virtual address space*. That turned out to
  be the wrong quantity: jemalloc (Polars) keeps freed ranges mapped,
  glibc malloc reserves a 64MB arena per thread, and LightGBM/OpenMP and
  Polars thread pools each reserve stacks — all address space that is
  never touched. On real data, `make train` aborted on allocations of a
  few hundred KB while measured peak RSS was ~6–10GB under a 20GB cap,
  and the gap grows with core count. The historical notes above about
  jemalloc "high-water marks" describe that old mechanism; the
  per-process stage split they motivated is kept because it still lowers
  real peak memory.
- **Linux-only.** Where `/proc/self/statm` is unreadable (macOS,
  Windows), `apply_memory_limit_gb` logs a warning instead of raising,
  so the pipeline still runs, just unlimited.
- **It's polled, not a hard allocation limit.** A single very fast
  allocation can overshoot the cap briefly before the watchdog notices,
  so keep the cap comfortably below physical RAM. For a strict kernel-
  enforced limit, run the target under a cgroup instead, e.g.
  `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 make train`.
- **`src/api/main.py` deliberately does NOT call it.** FastAPI's
  `TestClient` really executes the app's `lifespan` context manager (test
  dependency overrides don't prevent that), so a call placed there would
  apply a real memory cap to the pytest process running the test suite,
  not to a real deployment. The API's entry point is
  `pipelines/run_api.py` instead (now what `make api` runs) — it applies
  the limit and then hands off to uvicorn, and it's never imported by
  tests. `src/dashboards/app.py` applies the limit directly at module
  level, since it's the opposite case: `tests/smoke/test_smoke.py` only
  `ast.parse`s that file's source to check for syntax errors and never
  imports/executes it, so a real side effect there is safe.

---

### 5.11 How `make train` fits in RAM (the resolution)

**Symptom.** On real Backblaze Q1 data (25.7M drive-days, ~200 columns,
32 Parquet row groups), `make train` kept aborting with allocation failures
("Cannot allocate memory", Rust "memory allocation of N bytes failed"),
each time one step later after every fix, and finally on ~940KB requests.

**Root cause: the cap measured the wrong thing.** The cap was `RLIMIT_AS`
(virtual address space). Polars/jemalloc keep freed ranges mapped, glibc
reserves an arena per thread, and LightGBM/OpenMP/Polars thread pools
reserve stacks — address space that is never touched. Measured peak RSS at
each crash was ~6–10GB under a 20GB cap. **Fix (`src/resource_limits.py`):
the cap is now enforced on resident memory (RSS)** by a watchdog thread
(§5.10), so only RAM actually in use counts. That alone is what made the
last crashes disappear.

**Real memory is also kept low.** Three measures in
`pipelines/train_model.py` bound the *actual* peak, so the RSS cap has
headroom:

1. *Row-group-at-a-time reads, spilled to disk.* The 25.7M-row join
   (`join_by_native_row_groups`) and each split's extraction
   (`_row_group_parts`) read one Parquet row group with pyarrow, keep only
   that split's rows, write them to a small part file and drop them.
   Measured RSS stays flat at ~2–3GB across all 32 groups.
2. *Build the numpy matrix directly.* `_build_feature_arrays`
   pre-allocates the final float32 matrix and fills it part by part.
   Combining parts into one DataFrame and then converting held both copies
   at once (~6GB + ~3.7GB).
3. *Training-row cap.* `model.max_train_rows` (5M) keeps every failure row
   and hash-subsamples the healthy rows deterministically (16.1M -> 5.0M
   train rows), because one dense float32 copy of the full train split no
   longer fits comfortably.

Each heavy step (join; train / validation / test extraction) runs as its
own subprocess (`--stage assemble`, `--stage extract-split --split X`),
communicating through `.npy`/Parquet files in a scratch dir. This keeps the
orchestrator small and frees each step's memory unconditionally on exit.

**Disk, not RAM.** Scratch files (the joined frame is tens of GB) go to
`resource_limits.scratch_dir`, default `<gold_dir>/../tmp` — not `/tmp`,
which was too small on the reference machine.

**Measured on the real run** (peak RSS per stage, from temporary checkpoints since
removed): join ~ flat; train extraction 6.0GB; validation 6.6GB; test
2.0GB; training + baseline in the orchestrator completed under the 20GB
cap.

**If it still hits the cap:** the process logs `memory_limit_exceeded`
with the RSS and exits with code 86. Lower `model.max_train_rows`,
`diagnostics.baseline_max_rows` or `diagnostics.shap_max_rows`, or raise
`resource_limits.max_memory_gb`. For a strict kernel-enforced limit run
under a cgroup: `systemd-run --user --scope -p MemoryMax=20G make train`.

**SHAP is optional and off by default** (`diagnostics.shap_enabled` in
`configs/model.yaml`, see §5.1).

---

### 5.12 Judging and improving the model: drive-level metrics and `make experiment-model`

The model goal (precision >= 95%, recall 35-50%, `configs/model.yaml`) is
stated per *drive*, but row-level metrics count a failing drive's ~14
near-identical positive rows 14 times. `src/models/evaluation.py::
drive_level_metrics` reports it per drive: a failing drive is *caught* if
any row inside its prediction window scores >= threshold, a healthy drive
is a *false alarm* if any of its rows does. `make train` now logs it
(`validation_drive_level`, `test_drive_level`), writes it to the evaluation
report and model card, and logs `*_drive_precision/recall/auprc` to MLflow.
Row-level numbers are still reported next to it.

To try modelling ideas without re-running the whole join:

```bash
make train TRAIN_ARGS=--keep-work-dir   # once; keeps x_*.npy, *_ids.parquet, test_df.parquet
make experiment-model                   # compares variants (pipelines/experiment_model.py)
```

For the precision gap specifically, add `--deep-dive`:

```bash
make experiment-model EXPERIMENT_ARGS="--variants reg_spw --deep-dive reg_spw"
```

It prints (1) what the "false alarm" drives actually are, via
`src/models/alerting.py::analyze_false_alarms` (failures just beyond the
14-day horizon, drives pulled without failing, or still-active ones, each
with a lift over its base rate - separates model error from label and
evaluation artifacts), and (2) whether a *persistence rule* helps:
`smooth_scores` replaces each drive-day's score with a causal rolling
mean / min / median over the drive's last 3-7 observations ("alert only if
it stayed high"), with thresholds chosen on validation and applied to
test. The persistence rule is not in the shipped pipeline yet; scoring
(`pipelines/score_fleet.py`) would need each drive's recent history, not
just its latest row, if it proves worthwhile.

Each variant is early-stopped on a validation subsample and scored at row
and drive level; thresholds are chosen on validation and applied to test.
Excluded features are zeroed in place and restored, so a variant costs no
extra copy of the matrices. Delete the kept `data/tmp/train_model_frame_*`
directory when finished (it holds multi-GB arrays).

**Real-data findings (Q1 2026, 14-day horizon, 345k drives, 337 failing in
validation, 309 in test).** All rows below use early stopping on average
precision only (see `train_lightgbm`); test thresholds are chosen on
validation.

| variant | trees | drive AUPRC val / test | test drive precision @ recall |
|---|---|---|---|
| logistic baseline | - | 0.056 / 0.070 | - |
| first real run: `is_unbalance`, no regularization, no early stopping | 1 | ~0.04 / 0.03 | 25% @ 0.3% |
| `is_unbalance` + regularization | 549 | 0.256 / 0.154 | 33% @ 16%; 16% @ 39% |
| **sqrt weight + regularization (shipped)** | 434 | 0.230 / 0.158 | 40% @ 13%; 17% @ 38% |
| weight power 0.25 | 453 | 0.241 / 0.161 | 35% @ 17%; 16% @ 39% |
| weight power 0.75 | 507 | 0.209 / 0.148 | 36% @ 9%; 17% @ 36% |
| 63 leaves | 303 | 0.240 / 0.158 | 38% @ 14%; 16% @ 37% |
| stronger regularization | 361 | 0.223 / 0.165 | 32% @ 10%; 18% @ 41% |

Takeaways:
- **Regularization + proper early stopping is what fixed the collapse.**
  The class-weight choice is second order: `is_unbalance` works fine once
  regularized. (An earlier table here blamed the full-ratio weight; that was
  an artifact of early stopping on unweighted logloss.)
- **Tuning has plateaued.** Every variant lands within ~0.15-0.17 test drive
  AUPRC, inside run-to-run noise at ~300 failing drives. Dropping identity
  features (age, capacity) or per-drive weighting did not help either.
- **False alarms are mostly real, persistently "sick-looking" drives.**
  `--deep-dive` at the 35%-recall operating point (validation, 296 alerting
  "healthy" drives): 239 are still running (lift 0.8x, i.e. ordinary
  drives), 36 were removed without a recorded failure (lift 21x) and 21
  failed later, 15-60 days after the first alert (lift 200x). So ~19% of
  "false alarms" are arguably early warnings, but most are not label
  artifacts.
- **Persistence rules do not help.** Rolling mean/min/median over 3-7 days
  lowered precision at every recall level; the false alarms are drives whose
  SMART signals stay elevated, not one-day spikes.

**Trying another horizon.** `make train TRAIN_ARGS="--horizon-days 30"`
trains for any horizon in `horizons_days` (labels for 7/14/30 already exist,
no `build-labels` re-run). The run is logged with its `horizon_days` and
the model card is named `v<version>_h<horizon>d`; `make score-fleet` only
loads runs whose horizon equals `primary_horizon_days`, so a trial run never
becomes the production model by accident. Metrics at different horizons are
not comparable (a longer horizon labels more rows positive); judge them by
what the agent can do with the warning.

**Goal status: not met, and not reachable by tuning.** Shipped model, test,
drive level: 23% precision at 30% recall (92 of 309 failures caught, 309
healthy drives alerting out of 345k; validation chose the threshold at
29% / 36%), 40% at 13%, and >= 95% only at ~1% recall. The 95% target stays
in `configs/model.yaml` and the model card records the gap
(`target_met: false`). The remaining levers change the data or the
objective: more history (more failing drives to learn from), a longer
horizon (the 21 late failures above become hits at 30 days), and tiered
operating points per agent action instead of one 95% target.

---

## 6. The MAPE-K agent (`src/agent/`)

### 6.1 State (`state.py`)

`AgentState` is a `TypedDict` and deliberately thin — ids, small dicts,
booleans. **Nothing in it may be a raw enum or other non-JSON-safe object**:
LangGraph checkpoints this state via a JSON/msgpack serializer, and passing
it a `FeatureMaturity` enum instance directly produces a
"will be blocked in a future version" warning today and a hard failure
later. If you write a new predictor or node that puts a Pydantic model or
enum into state, normalize it to a plain string/int/float/bool/dict/list
first — see `_normalize_prediction_summary` in `nodes.py` for the pattern.

### 6.2 Dependencies (`deps.py`)

`AgentDependencies` is the seam between the agent and everything real:

| Field | Type | Real implementation lives in |
|---|---|---|
| `fleet_state_provider` | `() -> dict` | `src/simulator/fleet.py::FleetSimulator.snapshot`, or a real fleet API client |
| `predictor` | `(dict) -> dict` | Phase 5 model inference (not yet wired to a real model — see §11) |
| `guardrail_evaluator` | `(dict) -> dict` | `src/guardrails/adapter.py::build_guardrail_evaluator` |
| `post_action_guardrail_evaluator` | `(dict, dict) -> dict` | `src/guardrails/adapter.py::build_post_action_guardrail_evaluator` |
| `executor` | `(dict) -> dict` | `src/simulator/actions.py::apply_action` |
| `validator` | `(dict) -> dict` | `src/simulator/actions.py::validate_action` |
| `action_ledger` | `ActionLedger` protocol | `InMemoryActionLedger` (tests) or `SqliteActionLedger` (real use) |

**`Executor`'s contract**: its return dict *must* include `drive_id` and
`proposed_action` alongside `success`/`compensating_action_triggered`/`error`
— the post-action guardrail check needs both, and a test executor that omits
them will raise `KeyError` inside the Validate node, not silently pass.
`src/simulator/actions.py::apply_action` satisfies this contract; if you
write your own executor (e.g. to hit a real fleet API), copy its return
shape.

**Use `SqliteActionLedger`, not `InMemoryActionLedger`, for anything that
needs to survive a process restart.** The in-memory ledger is fine for a
single test or a single `.invoke()` sequence within one process, but it
defeats the entire purpose of idempotent action IDs the moment the process
restarts — there is no way to tell, from a *fresh* in-memory ledger, that an
action already ran. `src/agent/demo.py::build_demo_dependencies` and
`src/api/main.py`'s lifespan both use `SqliteActionLedger` backed by
`configs/agent.yaml`'s `checkpoint.action_ledger_path`.

### 6.3 Nodes (`nodes.py`) and graph (`graph.py`)

```text
monitor → analyze → plan ─┬─(guardrail passed, non-HUMAN_REVIEW tier)─→ execute → validate → END
                           └─(guardrail blocked, or tier == HUMAN_REVIEW)─→ human_review ─┬─(approved)─→ execute → validate → END
                                                                                            └─(rejected)──────────────────→ END
```

One `.invoke()` runs one cycle. "Return to Monitor for the next cycle" is
modeled as the *caller* re-invoking with the same `thread_id`, not an
internal graph edge — this keeps each checkpointed step small.

**`human_review` genuinely pauses the graph**, using LangGraph's native
`interrupt()`/`Command(resume=...)` mechanism (`make_human_review_node` in
`nodes.py`):

```python
first = app.invoke({}, config=config)          # {'...', '__interrupt__': [Interrupt(value={'proposal': ..., ...})]}
second = app.invoke(
    Command(resume={"approved": True, "operator_id": "op-1", "reason_code": "CONFIRMED"}),
    config=config,                              # same thread_id
)
```

`app.invoke(...)`'s mypy overload resolution is overly strict about this
exact (and LangGraph-recommended) calling pattern — see the two
`# type: ignore[call-overload]` comments in `src/agent/orchestrator.py` for
why they're there and why they're safe to keep.

**Idempotency**: `_deterministic_action_id(run_id, drive_id, tier)` hashes
those three values. `run_id` is generated once per thread (`state.get("run_id")
or str(uuid.uuid4())` in `make_monitor_node`) and persists across cycles on
that thread via the checkpoint, so re-invoking the same thread after a
"crash" reproduces the same `action_id`, and `execute` checks
`action_ledger.is_executed(action_id)` before calling the real executor.

**Hysteresis & cooldown** (`hysteresis.py`): `configs/agent.yaml`'s
`hysteresis_cycles_required`/`cooldown_seconds` are enforced here, not just
loaded and ignored. Plan tracks a per-drive `consecutive_escalation_cycles`
counter in `AgentState.drive_risk_state` (small counters, not telemetry —
still lean) and caps an escalation tier (cordon/migrate/drain) at WARN until
it's been proposed that many consecutive cycles; Execute records a
`last_action_at` timestamp after any escalation-tier action, and Plan
downgrades to WARN again if the same drive is re-proposed within the
cooldown window. `tests/unit/test_hysteresis.py` and
`tests/integration/test_agent_graph.py::test_real_hysteresis_config_caps_first_cycle_and_escalates_on_the_second`/
`::test_cooldown_prevents_re_drain_on_the_cycle_immediately_after_one` cover
this against the real `configs/agent.yaml` defaults, not just a relaxed
test override.

### 6.4 Orchestrator (`orchestrator.py`)

`AgentOrchestrator` is the only thing that should call `.invoke()` in
production code — it's the glue between the compiled graph and the API's
`InMemoryAuditStore`:

- `run_cycle(thread_id)`: invokes; if the result contains `"__interrupt__"`,
  records a `PendingAction` (with `thread_id`, so it can be resumed later)
  in the store and returns `{"status": "pending_review", ...}`. Otherwise,
  builds a provisional trust-score assessment
  (`src/reliability/audit.py::build_provisional_assessment`) and records a
  decision.
- `resume_after_decision(action_id, approve=..., ...)`: looks up the
  `PendingAction`'s `thread_id`, marks it decided in the store, then
  actually resumes the LangGraph thread via `Command(resume=...)`.

Both methods time their `.invoke()` call (`time.perf_counter()`) and surface
`cycle_duration_seconds` in the outcome dict and the decision record —
docs/design_goal.md section 26's "< 5 minutes" target was previously
unmeasured anywhere. A cycle exceeding `AgentSettings.target_cycle_time_seconds`
(from `configs/agent.yaml`'s `loop.target_cycle_time_seconds`) prints a
warning; it's a soft signal, not a hard failure.

Every decision record includes `date` (the cycle's timestamp date — this
system doesn't yet track a prediction's "as of" drive-day separately from
wall-clock time, so this is an approximation), `horizon_days` (from
`ModelSettings.load().primary_horizon_days`), `safety_violation`, and
`guardrail_severity` — these four fields are what
`src/reliability/batch.py::resolve_final_trust_scores` needs to join a
decision back to its resolved label later.

---

## 7. Guardrail engine (`src/guardrails/`)

`configs/guardrails.yaml` is the catalog; `rules.py` implements every rule
in it (11 rules — if you add a rule to the YAML, add the matching function
here, or the catalog and the code will silently diverge again). Each rule is
a pure function `RuleContext -> GuardrailViolation | None`.

```text
ALL_RULES (pre-action, run in Plan):
  PRED_THRESHOLD, FEATURE_CONFIDENCE, TELEMETRY_FRESHNESS   — HARD
  HARD_NO_LAST_NODE, HARD_QUORUM, HARD_MAX_DRAINS           — HARD
  SOFT_HIGH_IO, OPS_MAINTENANCE, OPS_RATE_LIMIT             — SOFT

POST_ACTION_RULES (run in Validate, after execution):
  POST_DATA_INTEGRITY, POST_SERVICE_CONTINUITY              — HARD
```

`GuardrailEngine.evaluate(ctx)` runs `ALL_RULES`; `.evaluate_post_action(ctx)`
runs `POST_ACTION_RULES` against a `PostActionContext` (needs
`data_integrity_ok`/`service_continuity_ok` from the validator's output, not
the pre-action context). Both share `_run_rules`, which sorts violations
hard-before-soft (the "safety > operational > efficiency" conflict
resolution from the design doc is implemented as this sort order — a hard
violation always blocks regardless of how many soft ones also fired).

`src/guardrails/adapter.py` bridges this typed engine to the agent's
dict-based `GuardrailEvaluator`/`PostActionGuardrailEvaluator` seams.
`InMemoryOperationalState` (concurrent-drain count, rolling actions-per-hour)
is correct for a single process; `src/guardrails/redis_operational_state.py::
RedisOperationalState` implements the same interface (a `sadd`/`srem` set for
active drains, a `zadd`/`zremrangebyscore`/`zcard` sorted set for the rolling
hour of action timestamps) against a real Redis client (`redis` was
previously a declared but entirely unused dependency), so the counters are
genuinely shared across multiple API/agent process instances rather than
resetting per process. `build_guardrail_evaluator` picks between them via
`configs/guardrails.yaml`'s `operational_state.backend` (`in_memory` by
default; `redis` reads `operational_state.redis_url`) unless a caller passes
`operational_state=` explicitly, which always wins - `GuardrailEngine`
itself never changes either way.

**Defense in depth, on purpose**: a low-confidence or stale-telemetry
destructive action is caught *twice* — once at the Plan/action-tier layer
(`src/models/action_tiers.py::determine_action_tier` downgrades it to
`cordon` before guardrails even run) and again at the guardrail layer
(`FEATURE_CONFIDENCE`/`TELEMETRY_FRESHNESS`, in case the first layer is ever
bypassed or misconfigured). `tests/chaos/test_chaos_scenarios.py` documents
and exercises this explicitly — don't "simplify" it down to one layer.

---

## 8. Fleet simulator (`src/simulator/`)

`FleetSimulator` holds `Drive`/`Node`/`ReplicationGroup` and a state machine
(`DriveState`: HEALTHY → DEGRADED → PREDICTED_FAILURE → CORDONED/MIGRATED/
DRAINED → REPLACED/RECOVERED, or → FAILED). `is_last_healthy_node_in_domain`
and `quorum_ok_after_drain` mirror exactly what the guardrail engine's
`HARD_NO_LAST_NODE`/`HARD_QUORUM` rules check — when wiring a real fleet
provider, these two methods (or their equivalents) are what feed those
guardrails.

`apply_action(simulator, action, fail_hook=...)` executes cordon/migrate/
drain and rolls back to the prior state on an injected `ExecutionError`
(the compensating action). `fail_hook` is the chaos-injection seam:
`src/simulator/chaos.py` provides `make_action_timeout_hook` (fails specific
`action_id`s) and `make_flaky_hook` (fails with a probability, seeded via a
`numpy.random.Generator` for determinism in tests).

---

## 9. Reliability checker (`src/reliability/`)

```text
BaseScore   = 0.40·Correctness + 0.30·Timeliness + 0.30·Necessity
TrustScore  = BaseScore × SafetyMultiplier × GuardrailMultiplier
SafetyMultiplier    = 0 on data-loss/quorum breach, else 1
GuardrailMultiplier = 0 hard violation, 0.5 soft violation, 1.0 none
```

- `checks.py`: `check_correctness`/`check_necessity`/`check_timeliness`
  return `None` (never a default `0`/`1`) when the label is censored —
  callers must handle `None` explicitly, not coerce it.
- `multipliers.py`: the two multipliers above, plus
  `guardrail_severity_from_violations` (worst-of a violation list) and
  `safety_violation_from_validation`.
- `trust_score.py`: `compute_provisional_trust_score` uses the model's own
  `p_fail` as a *necessity proxy* (true necessity/correctness/timeliness
  aren't knowable until the label horizon resolves).
  `compute_final_trust_score` uses the real checks — **it vetoes on the
  guardrail severity recorded at decision time, even if a human later
  approved and executed the action anyway.** This is intentional: trust
  score measures the automated decision's quality, not whether a human
  overrode it. If you're surprised a trust score is `0.0` for an executed
  action, check `guardrail_result` for a hard violation before assuming a
  bug.
- `audit.py::build_provisional_assessment` merges pre-action and post-action
  guardrail violations before computing severity, and generates the
  human-readable explanation lines shown in the dashboard/audit trail.
- `batch.py::resolve_final_trust_scores` is the "async evaluation" step —
  run it periodically against accumulated decisions once labels resolve; it
  joins on `(drive_id, date, horizon_days)` and silently skips anything
  without a matching non-censored label (they stay provisional-only until a
  later run finds one).
- `drift.py`: PSI-based feature/prediction drift (`population_stability_index`,
  `build_drift_report`), with the conventional 0.1/0.25 thresholds and a
  `retrain_recommended` flag. Not wired to any scheduled job yet — call it
  from whatever retraining cadence you set up.

---

## 10. API & dashboard (`src/api/`, `src/dashboards/`)

`src/api/main.py`'s `lifespan` handler compiles the agent graph **once**,
against `AgentSettings.load().checkpoint_path` (a real SQLite file, not
`:memory:` — a paused thread must survive an API process restart). It
defaults to `build_demo_dependencies()` (the same hardcoded two-drive demo
fleet as `make agent-demo`); swap `app.state.orchestrator` for a real
`AgentOrchestrator` (built with real `fleet_state_provider`/`predictor`) to
run against a real fleet/model.

`InMemoryAuditStore` (`store.py`) is explicitly a stand-in for Redis (hot
state) + DuckDB/Postgres (durable audit trail) — nothing here persists to
disk. If you need the audit trail to survive an API restart, that's the
next thing to build; the store's interface is small enough to reimplement
against a real backend without touching `main.py`.

Endpoints:

| Method & path | Purpose |
|---|---|
| `POST /api/v1/agent/run-cycle` | Runs one MAPE-K cycle; records the outcome |
| `GET /api/v1/fleet/state` | Last reported fleet snapshot (404 if none yet) |
| `GET /api/v1/predictions/latest` | Last cycle's per-drive predictions |
| `GET /api/v1/actions/pending` | Approval queue |
| `POST /api/v1/actions/{id}/approve` `/reject` | Resumes the paused thread |
| `GET /api/v1/audit/decisions` | Full decision history |
| `GET /api/v1/audit/decisions/export?format=csv\|parquet` | Decision history as a downloadable file |
| `GET /api/v1/analytics/failure-rate-by-model-family?horizon_days=N` | DuckDB analytics over the gold Parquet layer |
| `GET /api/v1/reliability/trust-trend` | Provisional/final trust scores over time |
| `GET /api/v1/guardrails/violations` | Flattened violation log |

**Audit export** (`src/reliability/audit_export.py`,
docs/project_plan.md Phase 10 Key Task 6): decisions can carry nested,
heterogeneous fields (`guardrail_result`, `execution_result`, `None` for
either) that don't map cleanly onto a single flat row schema, so
`decisions_to_dataframe` JSON-serializes every non-scalar field to a string
column (keeping it `None` where the field itself is `None`) before handing
the frame to Polars' `write_csv`/`write_parquet` — this is what lets a
single export cover decisions from a mix of pending-review-only and
fully-executed cycles without either format choking on inconsistent
nested shapes.

**DuckDB analytics** (`src/reliability/analytics.py`,
docs/project_plan.md Phase 10 Key Task 3 "Read analytics from DuckDB"):
`duckdb` was a declared but entirely unused dependency — every other read
path in this codebase goes through Polars or the in-memory
`InMemoryAuditStore`. `failure_rate_by_model_family` is the one place it's
used for real: it joins the gold features and labels Parquet files
directly on disk via DuckDB's `read_parquet(...)` (out-of-core - neither
file is loaded into a Polars/pandas DataFrame first) and groups by
`model_family`, which is exactly the "out-of-core SQL over Parquet" role
the design docs describe for it. `GET /api/v1/analytics/
failure-rate-by-model-family` exposes it; it returns `[]` before `make
build-features`/`make build-labels` have been run, rather than erroring.

`src/dashboards/app.py` reads these endpoints via `requests` (not covered by
automated tests — Streamlit apps aren't meaningfully testable under pytest;
verified manually that `streamlit run` boots and serves the page). It is a
separate process from the API by design (`API_BASE_URL` env var).

---

## 11. Extension recipes

**Add a guardrail rule**: add it to `configs/guardrails.yaml`'s `rules:`
list, write the matching function in `src/guardrails/rules.py`, add it to
`ALL_RULES` or `POST_ACTION_RULES`, and add a unit test in
`tests/unit/test_guardrails.py` covering both the fire and no-fire case.

**Add a gold feature**: add the computation to the relevant
`src/features/*.py` module (or a new one, following the
partition-by-drive_id/time-ordered pattern), wire it into
`pipelines/build_gold_features.py::build_gold_features`, and add an entry to
`src/features/registry.py::build_registry` so it's versioned/audited.

**Swap in a real fleet + model**: write a `fleet_state_provider` returning
`{"fleet_snapshot_id": ..., "drives": [...]}` and a `predictor` returning
`{"drives": [{"drive_id", "p_fail", "feature_confidence", "feature_maturity",
"stale_telemetry", ...}]}` matching the shape `src/agent/demo.py` produces,
then build an `AgentDependencies` with real `executor`/`validator` (calling
your real fleet's cordon/migrate/drain APIs, returning the same shape as
`src/simulator/actions.py::apply_action`). Use `SqliteActionLedger`.

**Add an API endpoint**: follow the existing pattern in `src/api/main.py` —
depend on `get_store`/`get_orchestrator`, never reach into
`app.state.orchestrator` directly from a route (tests override the
dependency, not the module-level `app` object).

---

## 12. Testing strategy

```text
tests/unit/          one file per src module — pure functions, no I/O beyond tmp_path
tests/integration/   agent+guardrails, agent+simulator, agent+API-store; real LangGraph graphs, InMemorySaver/SqliteSaver
tests/chaos/         full-stack scenarios (stale telemetry, correlated failure, action timeout,
                      human-review SLA timeout) + guardrail latency benchmark (<500ms budget)
tests/golden/        synthetic 100-healthy/100-failing dataset; regenerate via
                      tests/golden/generate_golden_dataset.py if the schema changes
tests/property/       Hypothesis property-based tests — invariants (hysteresis streaks,
                      trust-score bounds, calibration bin accounting) checked against
                      arbitrarily generated inputs, not just the hand-picked examples
                      tests/unit/ covers
tests/smoke/         boots the real default wiring (build_demo_dependencies) end-to-end and
                      walks every documented API endpoint once — "does the shipped system work at all"
```

Run everything: `make test`. Run one layer: `make test-unit` /
`make test-integration` / `make test-chaos` / `make test-golden` (alias:
`make data-check`) / `make test-property` / `make smoke`. `make ci` runs exactly what
`.github/workflows/ci.yml` runs (`lint` + `test`), so you can reproduce a CI
failure locally before pushing. `make coverage` runs the full suite under
`pytest-cov` and writes an HTML report to `htmlcov/` (open
`htmlcov/index.html`) — there's no enforced coverage threshold yet, this is
a local diagnostic only.

Every new module should get a same-named test file; every new cross-module
behavior (agent talking to guardrails, guardrails talking to the simulator,
API talking to the orchestrator) should get an integration test that
exercises the *real* collaborators, not mocks of them — the codebase leans
heavily on real `GuardrailEngine`/`FleetSimulator`/`compiled_agent` instances
in tests specifically to catch integration bugs that mocks would hide (this
is how the "human review was a dead end" and "action ledger wasn't
persistent" bugs were actually caught).

**The golden dataset tests a weaker guarantee than the name suggests.**
`tests/golden/test_golden_dataset.py` only checks that
`build_golden_dataset()` itself produces the right shape/labels and
round-trips through Parquet — it does **not** run the golden data through
the real `src/features/*`/`src/labels/*` pipeline and diff against a
checked-in expected-output snapshot. It currently provides zero protection
against a silent change in feature/label values over time. If you need that
guarantee, add a snapshot test that runs `build_gold_features`/
`compute_labels` against the golden dataset and asserts against a committed
expected-output file.

**Smoke tests never touch real project files.** `tests/smoke/test_smoke.py`
calls `build_demo_dependencies(action_ledger=InMemoryActionLedger())` and
uses the default `:memory:` LangGraph checkpoint — running `make smoke`
repeatedly (including in CI) writes nothing to `mlflow/`. If you write a new
test that calls `build_demo_dependencies()`, always pass an explicit
in-memory `action_ledger` unless you specifically intend to exercise the
real persistent one.

### 12.1 End-to-end walkthrough: synthetic data vs. real data

The automated suite (§12 above) is the fast, CI-safe way to know the
system works. This section is for manually walking the *data pipeline*
end-to-end yourself — either entirely offline with synthetic data, or
against real Backblaze/SMART-Z data.

#### A. Synthetic data (fast, offline, no download)

**Agent/API loop only, no data pipeline needed:**

```bash
make smoke        # one real MAPE-K cycle + every documented API endpoint, in-memory
make agent-demo   # one real MAPE-K cycle against a hardcoded 2-drive fleet, printed live
```

Both use the real guardrail engine, trust-score math, and LangGraph
graph — just synthetic/hardcoded fleet data instead of a trained model's
real predictions. `make smoke`/`make test` never write to `mlflow/`;
`make agent-demo` does (it uses the real persistent `SqliteActionLedger`
and checkpoint, by design — see `src/agent/demo.py`).

**Full data pipeline, bronze through a real trained model:**

```bash
make ingest-synthetic-stub   # 15 drives spanning configs/model.yaml's splits -> data/bronze/synthetic/
make build-silver            # -> data/silver/{drive_day,canonical_telemetry,drive_metadata}/
make build-features          # -> data/gold/features/part.parquet (~196 columns)
make build-labels            # -> data/gold/labels/part.parquet + dataset_versions/*.json
make train                   # trains, tunes threshold, logs to MLflow, writes a model card
make score-fleet             # batch-scores the current fleet -> data/audit/predictions/
make plots                   # renders calibration/SHAP/imbalance plots -> data/audit/plots/
```

Each stage logs a structured `*_written` event with the row/column count
and output path (`src/logging_config.py`) - use that to confirm each
stage actually produced rows before moving to the next. **This actually
completes end-to-end**, including `make train`: `src/ingest/synthetic_stub.py`
generates 6 healthy drives and 9 failing drives (real degrading
reallocated/pending-sector trajectories, mirroring
`tests/golden/generate_golden_dataset.py`). It **reads
`configs/model.yaml`'s `splits`/`primary_horizon_days` directly** (rather
than hardcoding a second copy of those dates) to derive its own date
range and to place failure dates so every split - `train`, `validation`,
*and* `test` - ends up with both positive and negative labels:
`_failure_dates_covering_every_split` allocates most failures across
`train` (by far the largest bucket, since telemetry starts
`LEAD_DAYS_BEFORE_TRAIN_END` = 400 days before `train_end`) and anchors
(at least) one each at the very end of `validation` and `test`, since a
day-count-proportional spread starves the smaller buckets of any failure
at all - this is exactly what happened the first time `configs/model.yaml`'s
defaults were changed without updating the stub to match, before this
derivation existed. Since the trajectories are noise-free by design,
expect a suspiciously perfect AUPRC (~1.0) — that confirms the *pipeline
plumbing* is correct, not that the model is any good; it isn't a
substitute for validating against real data's actual noise and ambiguity.

Two things this rewrite deliberately fixed, worth knowing about:
- **`failure_date` derivation** (`src/preprocess/failure_events.py::
  derive_failure_date`, wired into `pipelines/build_silver.py`): Backblaze
  (and now the synthetic stub) reports failure as a per-day `0`/`1` flag,
  but `src/labels/event_types.py::classify_event_types` and
  `build_drive_metadata` need a single `failure_date` per drive. Nothing
  converted one into the other before this - meaning a real Backblaze
  failure would never have been classified as `CONFIRMED_FAILURE`, so
  every label would have come out `0` even with a dataset full of real
  failures. This was caught by actually running the pipeline end-to-end,
  not by unit tests exercising each stage in isolation with hand-built
  fixtures.
- **`make score-fleet` needs `feature_maturity` joined in**
  (`src/models/serving.py::join_feature_maturity`): it's a per-drive
  classification that lives only in Silver's `drive_metadata` table, never
  in the Gold features table `score_latest_drive_day` otherwise reads from
  - also only caught by running the real pipeline, since the unit tests
  for `src/models/serving.py` used a hand-built fixture that already had
  the column.

For a synthetic dataset with a different (larger, gold-feature-shaped)
degrading trajectory for ad hoc notebook exploration, see
`tests/golden/generate_golden_dataset.py` (100 healthy + 100 failing
drives, 60 days each) - `make test-golden` is the automated shape-only
regression check on it (§12); it does not run through
`src/features/*`/`src/labels/*`, so it isn't a substitute for the
bronze-through-gold walkthrough above.

Clean up between runs: `make clean-data` (removes everything under
`data/bronze`, `data/silver`, `data/gold`, `data/audit/*/*`, preserving
every `.gitkeep`; never touches `data/raw/` or `mlflow/`). MLflow/agent
runtime state lives under `mlflow/` and is gitignored - remove it by hand
(`rm -rf mlflow/mlruns mlflow/*.db mlflow/mlartifacts`) if you want a
fully clean slate between experiments.

#### B. Real data (Backblaze, optionally SMART-Z)

1. **Configure a time period** (§5.0) - nothing downloads by default:

   ```yaml
   # configs/data.yaml
   download:
     backblaze:
       quarters: ["Q1_2025"]   # start with ONE quarter - see the warning below
   ```

2. **Run the full pipeline:**

   ```bash
   make download-backblaze   # streams + extracts the configured quarter(s)
   make ingest-backblaze     # -> data/bronze/backblaze/
   make build-silver
   make build-features
   make build-labels
   make train                # LightGBM/XGBoost, threshold tuning, SHAP, model card, MLflow run
   make score-fleet          # batch-scores the current fleet -> data/audit/predictions/{date}.json
   make plots                # renders calibration/SHAP/imbalance plots -> data/audit/plots/
   ```

   Real Backblaze CSVs must carry `date`, `serial_number`, `model`,
   `capacity_bytes`, `failure` (`src/ingest/backblaze.py::REQUIRED_COLUMNS`)
   - true for every quarterly archive Backblaze has published.

3. **`make score-fleet` output is a standalone batch report, not wired
   into `make agent-demo`/`make api`.** Those still run against the
   hardcoded demo fleet regardless of whether you've trained a real model
   - see §5.8 for exactly why (the live loop's lean `AgentState` can't
   carry `PredictionOutput`'s full nested `FeatureConfidence`). To see the
   trained model's real predictions, read
   `data/audit/predictions/{date}.json` or the MLflow run directly
   (`configs/model.yaml`'s `mlflow.tracking_uri`).

4. **Disk space warning, from experience**: a single Backblaze quarterly
   archive is roughly 1-1.5GB compressed and expands to several GB of CSV
   across 90+ daily files once extracted. Check `df -h` before configuring
   more than one or two quarters. `make clean-data` deliberately never
   touches `data/raw/` (it's expensive to re-download) - if you need to
   reclaim space from raw archives, remove them manually:
   `rm -rf data/raw/backblaze/_archives data/raw/backblaze/*.csv`.

5. **SMART-Z**: `make download-smartz` requires a directly-configured
   `download.smartz.url` (§5.0 - there's no public bulk-download API).
   Once ingested via `make ingest-smartz`, its rows flow through the same
   silver/gold pipeline and are automatically excluded from
   train/validation/test as an external validation split
   (`split=external_smartz` - `src/labels/splits.py::apply_vendor_holdout`).

---

## 13. Known gaps (don't be surprised by these)

Data / model:

- No real Backblaze/SMART-Z data has been ingested in this repo — the data
  pipeline is real and tested against synthetic fixtures only.
- `predictor` is never backed by the trained Phase 5 model in `demo.py`/the
  API's default wiring — it's a hardcoded two-drive fixture. Wiring a real
  MLflow-registered model in is the natural next step (see §11).
- `src/reliability/batch.py` and `drift.py` aren't wired to any scheduled
  job — they're ready to call, but nothing calls them periodically yet, and
  `drift.py`'s `retrain_recommended` flag doesn't trigger anything.
- `src/agent/human_review_timeout.py` (SLA-overdue → safe-fallback logic) is
  exercised only by `tests/chaos/test_chaos_scenarios.py` — nothing in the
  running API/orchestrator periodically scans `store.list_pending_actions()`
  for overdue reviews and applies it. **A pending action left un-decided
  today sits in `pending_review` forever**, which is exactly the
  indefinite-stall failure mode `docs/design_goal.md` says must not happen.

Safety-critical defaults:

- **`RuleContext`'s fleet-topology fields default to the *permissive* side
  when not wired**, not the conservative one:
  `is_last_healthy_node_in_domain` defaults to `False` (so `HARD_NO_LAST_NODE`
  can never fire) and `quorum_ok_after_action` defaults to `True` (so
  `HARD_QUORUM` can never fire) — see `src/agent/nodes.py`'s `plan()` and
  `src/guardrails/adapter.py`'s `evaluate()`. `src/agent/demo.py`'s fixture
  never populates either field, so **both of the two hardest safety
  guardrails are silently disabled** in the default demo/API wiring today,
  with no warning or log line. Any real deployment MUST wire real
  fleet-topology data (e.g. from `FleetSimulator.is_last_healthy_node_in_domain`/
  `quorum_ok_after_drain`, §8) into every drive dict the `predictor` returns
  before enabling DRAIN in anything but a demo.

Operational hardening still needed for production:

- **No authentication on the FastAPI service at all** — every endpoint,
  including `/api/v1/actions/{id}/approve|reject`, is open to any caller who
  can reach the port. `operator_id`/`reason_code` are unverified free-text
  fields, not derived from an authenticated identity.
- **No locking around concurrent requests.** `InMemoryAuditStore.decide_action`
  does a check-then-act on `action.status` with no lock; two concurrent
  approve/reject calls (or two concurrent `/agent/run-cycle` calls with the
  same `thread_id`) can both pass the check before either writes, and both
  end up calling `self.app.invoke(...)` against the same LangGraph thread
  concurrently. FastAPI's sync routes run in a thread pool, so this is a real
  exposure, not theoretical.
- **No FastAPI exception handlers.** `run_cycle()` has no try/except at all;
  any exception inside `predictor`/`executor`/`validator`/`guardrail_evaluator`
  produces a raw unhandled-exception 500. `_decide()` only catches
  `KeyError`/`ValueError`; anything else also leaks a raw 500.
- `InMemoryAuditStore` doesn't persist across an API restart (only the
  LangGraph checkpoint and action ledger do) — the entire audit trail an
  incident review would need is lost on restart.
- No containerization (no Dockerfile/compose), no `.env`/secrets-management
  pattern, no health-check endpoint (`GET /health`), no CORS/rate-limiting.
- No coverage threshold is enforced anywhere (`pytest-cov` is installed but
  only invoked via the local-only `make coverage`, not in CI).
