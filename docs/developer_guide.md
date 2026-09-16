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

## 1. Mental model in five sentences

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
uv sync --extra dev      # installs runtime + dev deps into .venv
make lint                 # ruff + mypy (both must be clean; mypy runs in CI too)
make test                 # full pytest suite
```

Python 3.11+ is declared in `pyproject.toml`; mypy is configured for 3.12
syntax (`[tool.mypy] python_version`) because some dependency stub files
(numpy) require it — this doesn't affect the actual supported runtime
version, only what mypy parses in stub files.

Never invoke `python`, `ruff`, `mypy`, `pytest`, `streamlit`, or `uvicorn`
directly in this repo — always through `uv run` (or a `make` target, which
already does this). The Makefile learned this the hard way: every target
used to call these tools bare and silently failed outside an activated venv.

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
| `GET /api/v1/reliability/trust-trend` | Provisional/final trust scores over time |
| `GET /api/v1/guardrails/violations` | Flattened violation log |

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
```

Run everything: `make test`. Run one layer: `uv run pytest tests/chaos/`.
Every new module should get a same-named test file; every new cross-module
behavior (agent talking to guardrails, guardrails talking to the simulator,
API talking to the orchestrator) should get an integration test that
exercises the *real* collaborators, not mocks of them — the codebase leans
heavily on real `GuardrailEngine`/`FleetSimulator`/`compiled_agent` instances
in tests specifically to catch integration bugs that mocks would hide (this
is how the "human review was a dead end" and "action ledger wasn't
persistent" bugs were actually caught).

---

## 13. Known gaps (don't be surprised by these)

- No real Backblaze/SMART-Z data has been ingested in this repo — the data
  pipeline is real and tested against synthetic fixtures only.
- `predictor` is never backed by the trained Phase 5 model in `demo.py`/the
  API's default wiring — it's a hardcoded two-drive fixture. Wiring a real
  MLflow-registered model in is the natural next step (see §11).
- `InMemoryAuditStore` doesn't persist across an API restart (only the
  LangGraph checkpoint and action ledger do).
- `src/reliability/batch.py` and `drift.py` aren't wired to any scheduled
  job — they're ready to call, but nothing calls them periodically yet.
- `InMemoryOperationalState` (guardrail rate-limit/drain counters) is
  per-process, not shared across multiple API instances — a real deployment
  needs the Redis-backed version the design doc describes.
