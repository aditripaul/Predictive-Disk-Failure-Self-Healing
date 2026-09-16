# User Guide

## Predictive Disk-Failure Self-Healing Agent for Server Fleets

**Audience:** people running, operating, or reviewing decisions from this
system — SREs, on-call operators, or anyone approving/rejecting a proposed
fleet action. No Python or code-reading required for this guide. If you're
modifying the code itself, see `docs/developer_guide.md` instead.

---

## 1. What this system does

Disks fail. This system watches SMART telemetry from a server fleet,
predicts which drives are at risk of failing soon, and proposes what to do
about it — from "just keep an eye on it" up to "take this drive out of
service before it fails." It never acts alone on a risky judgment call: if
the evidence is thin, stale, or would violate a safety rule (like breaking
data redundancy), it stops and asks a human instead of guessing.

Every single decision — acted on, blocked, or handed to a human — gets a
**trust score** and a plain-English explanation, so you can audit *why* the
system did what it did, not just *what* it did.

### The five action levels, from least to most disruptive

| Level | What happens | When |
|---|---|---|
| **Monitor** | Nothing — just keep watching | Risk is low |
| **Warn** | A dashboard alert, no fleet change | Risk is moderate |
| **Cordon** | Stop placing new workload on the drive | Risk is high, non-destructive |
| **Migrate** | Proactively move data off the drive | Risk is high and evidence is strong |
| **Drain** | Take the drive out of service entirely | Risk is very high and evidence is very strong |

Migrate and Drain are "destructive" in the sense that they change what's
running where — the system holds these two to a much higher evidence bar
than the other three (see §7).

---

## 2. Quickstart

You need [uv](https://docs.astral.sh/uv/) installed. Everything below runs
from the repository root.

```bash
make install       # installs dependencies
make lint          # sanity check: code style + type checks
make test          # sanity check: full test suite
make smoke         # fastest sanity check: does the whole system boot and respond?
```

For every other operational command (running one layer of tests, a
coverage report, cleaning generated files, etc.), see
`docs/developer_guide.md` §3.1 — that table lists every `make` target this
project has.

### See the agent make one decision

```bash
make agent-demo
```

This runs one decision cycle against a small built-in two-drive scenario
(no real fleet or database needed) and prints exactly what it decided, why,
whether guardrails passed, and the resulting trust score.

### Run the full service (API + dashboard)

In one terminal:

```bash
make api
```

In another:

```bash
make dashboard
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 3. Using the dashboard

The dashboard has four tabs:

- **Fleet Health** — how many drives are in each state (healthy, cordoned,
  drained, etc.) right now.
- **Trust & Reliability** — a chart of trust scores over time, plus a table
  of every guardrail violation that's been recorded.
- **Approval Queue** — every action currently waiting on a human decision.
  Each one is expandable; fill in your operator ID and a reason code, then
  click **Approve** or **Reject**.
- **Audit Trail** — every decision ever made, in full.

The dashboard reads from the API, so `make api` must be running alongside
it. If a tab shows "Could not reach API," check that the API process is
still up.

---

## 4. Using the API directly

If you're scripting against this instead of using the dashboard, the API is
plain JSON over HTTP. With `make api` running:

**Trigger a decision cycle:**

```bash
curl -X POST http://localhost:8000/api/v1/agent/run-cycle \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "fleet-1"}'
```

`thread_id` identifies one ongoing "conversation" with the agent about one
fleet/scenario — reuse the same `thread_id` across cycles for the same
fleet. The response tells you whether the cycle **completed** (something was
decided, possibly executed) or is **pending_review** (blocked, waiting on a
human).

**Check what's waiting for your approval:**

```bash
curl http://localhost:8000/api/v1/actions/pending
```

**Approve or reject one:**

```bash
curl -X POST http://localhost:8000/api/v1/actions/<action_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"operator_id": "your-name", "reason_code": "CONFIRMED_TRAJECTORY"}'
```

`operator_id` and `reason_code` are **required** — the system will not
accept an approval or rejection without them (this is deliberate: every
override needs a name and a reason attached, permanently, in the audit
trail). `comment` is optional free text.

Rejecting works the same way with `/reject` instead of `/approve` — the
proposed action is never carried out, but the decision (including your
reason) is still recorded.

**See the full history:**

```bash
curl http://localhost:8000/api/v1/audit/decisions
curl http://localhost:8000/api/v1/reliability/trust-trend
curl http://localhost:8000/api/v1/guardrails/violations
```

| Endpoint | What it returns |
|---|---|
| `POST /api/v1/agent/run-cycle` | Runs one decision cycle |
| `GET /api/v1/fleet/state` | The last reported fleet snapshot |
| `GET /api/v1/predictions/latest` | The last cycle's risk predictions per drive |
| `GET /api/v1/actions/pending` | Actions waiting for your approval |
| `POST /api/v1/actions/{id}/approve` | Approve a pending action |
| `POST /api/v1/actions/{id}/reject` | Reject a pending action |
| `GET /api/v1/audit/decisions` | Every decision ever made |
| `GET /api/v1/reliability/trust-trend` | Trust scores over time |
| `GET /api/v1/guardrails/violations` | Every guardrail rule that ever fired |

---

## 5. Why did it ask me for approval?

A decision gets routed to you instead of being carried out automatically
whenever one of these is true:

- **A hard guardrail fired.** The system refuses to, for example, drain the
  last healthy copy of your data, break a required replication quorum,
  exceed the concurrent-drain limit, act on stale telemetry, or act on
  low-confidence evidence. Any one of these blocks the action outright,
  every time, no exceptions — this is a hard rule, not a suggestion the
  system can talk itself out of.
- **The proposed action itself required review** by policy, regardless of
  guardrail outcome.

When you open a pending action, you'll see exactly which guardrail rule(s)
fired and why, in the same explanation text that ends up in the permanent
audit trail. Common ones you'll see:

| Rule | Meaning |
|---|---|
| `HARD_NO_LAST_NODE` | This is the only healthy node in its failure domain — draining it would leave that domain with zero coverage |
| `HARD_QUORUM` | Draining this drive would drop the replication group below its minimum healthy-copy requirement |
| `HARD_MAX_DRAINS` | Too many drains are already in progress fleet-wide |
| `TELEMETRY_FRESHNESS` | The drive's telemetry is stale, or it hasn't been observed long enough to trust |
| `FEATURE_CONFIDENCE` | The evidence behind this prediction isn't strong enough for a destructive action |
| `PRED_THRESHOLD` | The failure probability isn't high enough to justify this action tier |

Soft rules (`SOFT_HIGH_IO`, `OPS_MAINTENANCE`, `OPS_RATE_LIMIT`) don't block
an action outright, but they do reduce its trust score and show up in the
audit trail — a reasonable thing to glance at even for actions that went
through automatically.

**Approving overrides the block for that one action.** It does not change
the underlying guardrail rule or lower the bar for the next decision — the
next cycle re-evaluates from scratch.

**A note on trust scores and overrides:** if you approve an action that a
hard guardrail had blocked, its trust score still reflects that the
*automated* decision was untrustworthy (score `0.0`) — approving it doesn't
retroactively make the automated judgment sound, it just means a human
chose to proceed anyway, with your name and reason attached. That's by
design: the trust score is a record of decision quality, not an approval
stamp.

---

## 6. Understanding trust scores

Every decision gets a **provisional** trust score immediately, and (once
enough time has passed to know whether the prediction was actually right) a
**final** trust score later. Scores run from `0.0` (untrustworthy) to `1.0`
(fully trustworthy) and are built from four ingredients:

- **Correctness** — was the prediction actually right, once the outcome is known?
- **Necessity** — was the action actually needed (no acting on drives that were fine)?
- **Timeliness** — if something did fail, was the action taken early enough to matter?
- **Safety & guardrail compliance** — did anything actually go wrong (data
  loss, broken redundancy), or was a hard rule violated? Either one forces
  the score straight to `0.0`, regardless of how good the other three
  ingredients look. A soft guardrail violation instead halves the score.

The provisional score, computed the moment a decision is made, uses the
model's own confidence as a stand-in for correctness/necessity/timeliness
(which genuinely can't be known yet) — treat a provisional score as "how
confident was the system," and a final score as "was it actually right."

### Which SMART readings drove the prediction?

Every time a model is trained (`make train`), the system also computes a
**global feature-importance ranking** using SHAP (a standard model-
explainability technique): which SMART attributes and derived signals
(rolling averages, slopes, spike counts, etc.) most influenced the model's
predictions overall. That ranking is written to
`data/audit/data_quality_reports/shap_feature_importance.json` after
training, and logged alongside the model in MLflow. It answers "what does
this model generally pay attention to," which is a useful sanity check —
e.g. confirming the model is actually keying off reallocated-sector-count
trends and not something spurious. This is a training-time, whole-model
view; per-decision explanations in the audit trail (§5) come from the
guardrail/trust-score system, not from SHAP directly.

---

## 7. Why destructive actions get extra scrutiny

Migrate and Drain require **all** of the following, not just a high risk
score:

1. The predicted failure probability clears the tier's threshold.
2. The evidence behind that prediction is fresh and complete enough to trust
   (not stale telemetry, not a drive too new to have a reliable history).
3. No hard guardrail fires (quorum, last-node, concurrent-drain limit,
   confidence, freshness — see §5).

If evidence is thin but risk still looks elevated, the system will not
guess — it downgrades to Cordon (stop new workload, nothing destructive)
instead of Migrate/Drain. You'll see this in the audit trail as an action
that looks less aggressive than the raw risk score alone might suggest;
that's the safety-over-aggressiveness design working as intended, not a bug.

---

## 8. Troubleshooting

**"Could not reach API" in the dashboard.** Make sure `make api` is running
in another terminal, and that `API_BASE_URL` (if you've set it) points at
the right host/port.

**An approval/rejection returns an error about missing fields.** Both
`operator_id` and `reason_code` are required on every approve/reject call —
this is enforced deliberately (see §4).

**Approving an action twice returns an error.** Once a pending action has
been decided (approved or rejected), it's final — you'll get an error
rather than a silent no-op or a duplicate execution.

**The training pipeline (`make train`) fails immediately.** This almost
always means no real drive-failure data has been loaded yet — see §9.

---

## 9. If you're setting this up with real fleet data

The demo/dashboard defaults ship with a hardcoded two-drive scenario so the
system is runnable with zero setup. To run it against real Backblaze
SMART-telemetry data instead:

1. Download Backblaze's quarterly hard-drive-stats CSVs into `data/raw/backblaze/`.
2. Run the pipeline in order:
   ```bash
   make ingest-backblaze
   make build-silver
   make build-features
   make build-labels
   make train
   ```
3. Each step prints where it wrote its output and a data-quality report.

Without real data, every one of these steps still runs correctly against
the small built-in synthetic dataset (`make ingest-synthetic-stub`), which
is useful for confirming the pipeline itself works, but `make train` will
tell you plainly that there isn't enough real history to train a meaningful
model — that message is expected, not an error in the system.
