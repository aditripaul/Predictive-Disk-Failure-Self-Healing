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
| `make test` | Full pytest suite (unit + integration + chaos + golden + smoke) |
| `make test-unit` | `tests/unit/` only |
| `make test-integration` | `tests/integration/` only |
| `make test-chaos` | `tests/chaos/` only |
| `make test-golden` / `make data-check` | `tests/golden/` only (same thing, two names) |
| `make smoke` | `tests/smoke/` only — see §12 |
| `make coverage` | Full suite under `pytest-cov`; writes `htmlcov/index.html` |
| `make ci` | `lint` + `test` — exactly what CI runs |
| `make clean` | Removes caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `htmlcov`, `.coverage`) — never touches data or runtime state |
| `make clean-data` | Removes regenerated pipeline outputs (`data/bronze`, `data/silver`, `data/gold`, `data/audit/*/*`) — never touches `data/raw/` source data |
| `make ingest-backblaze` / `ingest-smartz` / `ingest-synthetic-stub` | Bronze ingestion (§5) |
| `make build-silver` / `build-features` / `build-labels` | Silver/Gold pipeline stages (§5) |
| `make train` | Trains the model, tunes threshold, computes SHAP importance (§5.1) |
| `make final-report` | `pipelines/generate_final_report.py` — aggregates chaos/latency/model reports |
| `make agent-demo` | One MAPE-K cycle against the hardcoded demo fleet (§6) |
| `make dashboard` | Streamlit UI (§10) |
| `make api` | FastAPI service (§10) |

`clean`/`clean-data` are intentionally split: `clean` is always safe to run
without thinking; `clean-data` deletes real (if synthetic-derived) pipeline
output, so it's separate and never invoked by anything else automatically.

---

## 4. The configuration system

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

```text
make ingest-backblaze / ingest-smartz / ingest-synthetic-stub   → data/bronze/
make build-silver                                               → data/silver/
make build-features                                              → data/gold/features/
make build-labels                                                 → data/gold/labels/
make train                                                        → MLflow run + data/audit/.../model_evaluation_report.json
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
frame has zero non-censored rows for the primary horizon — which is exactly
what happens against the synthetic stub, since 10 days of history can never
observe a 14-day horizon.

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

### 5.1 Explainability (SHAP)

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
is the Phase 7 stand-in for the Redis-backed hot counters the design doc
describes; the interface is small enough to swap for a real Redis client
without touching `GuardrailEngine`.

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
tests/smoke/         boots the real default wiring (build_demo_dependencies) end-to-end and
                      walks every documented API endpoint once — "does the shipped system work at all"
```

Run everything: `make test`. Run one layer: `make test-unit` /
`make test-integration` / `make test-chaos` / `make test-golden` (alias:
`make data-check`) / `make smoke`. `make ci` runs exactly what
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
- `InMemoryOperationalState` (guardrail rate-limit/drain counters) is
  per-process, not shared across multiple API instances — a real deployment
  needs the Redis-backed version the design doc describes. Note: `redis` is
  already a `pyproject.toml` dependency but is never imported anywhere in
  `src/`/`pipelines/` — it's a declared-but-unused placeholder, not partial
  wiring.
- No containerization (no Dockerfile/compose), no `.env`/secrets-management
  pattern, no health-check endpoint (`GET /health`), no CORS/rate-limiting.
- No coverage threshold is enforced anywhere (`pytest-cov` is installed but
  only invoked via the local-only `make coverage`, not in CI).
