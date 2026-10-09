# ADR 0002: Attribute selection by measured separation, and vendor-aware feature confidence

Status: accepted, 2026-10-09. Implemented except where marked "not done".

## Context

Two questions were open after the precision work stalled:

1. The pipeline ingested 9 of the 186 SMART columns Backblaze publishes. Which
   of the rest, if any, are worth the cost (~31 gold columns per priority
   attribute)? An earlier expansion of four *context* attributes (seek error
   rate, power-on hours, temperature, UDMA CRC) moved nothing, so the question
   needed evidence rather than literature priors.
2. Was there a source of information the model had never seen, as opposed to
   another re-slicing of the same nine counters? Backblaze's 2023+ releases
   carry `datacenter`, `cluster_id`, `vault_id`, `pod_id`, `pod_slot_num`,
   which nothing in the project used or documented.

`scripts/screen_raw_attributes.py` was written to answer both from the raw
CSVs: single-attribute drive-day AUC with coverage for every SMART column, and
a per-grain comparison of failure counts against a binomial no-clustering null.

## Findings

Screen over 40 days spread across Q1+Q2 2026 (72,477 rows, 3,239 pre-failure,
30-day horizon). `sep` = |AUC - 0.5|.

### The priority set was well chosen, with one gap

| Attribute | sep | Coverage | Ingested before |
|---|---|---|---|
| `smart_187` reported uncorrectable | 0.326 | **0.32** | yes |
| `smart_196` reallocation event count | **0.306** | **0.66** | **no** |
| `smart_5` reallocated sectors | 0.290 | 0.99 | yes |
| `smart_197` current pending sectors | 0.287 | 0.97 | yes |
| `smart_198` offline uncorrectable | 0.211 | 0.99 | yes |
| `smart_188` command timeout | 0.182 | 0.32 | yes |

`smart_196` out-separates four of the five priority attributes and was not
ingested. The coverage column matters more than the ranking: `smart_187` and
`smart_188` are **Seagate** attributes at ~32% coverage, so the model's single
strongest feature is absent on about two-thirds of the fleet. `smart_196` sits
at double that coverage, reaching the drives where those two are null —
drives that until now had no defect-*event* counter at all. `0.32 + 0.66 =
0.98` suggests the two sets are close to complementary by vendor.

Expect the marginal gain to be well under the standalone AUC: reallocation
events and reallocated sectors are near-duplicates physically, so `smart_196`
will correlate with `smart_5`.

### Hypotheses the screen refuted

- **`smart_1` (raw read error rate) and `smart_10` (spin retry count).** Both
  were already in `STANDARD_SMART_ID_TO_CANONICAL` but never ingested, and
  README §2 names read-error rates and spin retries as the warning signs the
  project set out to detect. Measured: `smart_1` sep 0.062, `smart_10` below
  the reporting cutoff (< 0.032) — weaker than `smart_7`, already known weak.
  Textbook SMART lore, not this fleet. Not added.
- **Vendor-normalized values.** Every `_normalized` column matched or
  underperformed its raw twin; `smart_5` loses most of its signal (0.290 raw
  vs 0.098 normalized) because vendor scaling compresses it. Not added as
  features. They are still required in the ingest for SMART-Z harmonization,
  which publishes only normalized values — a comparability mechanism, not a
  model input.
- **`smart_183` (0.2% coverage) and `smart_184` (7.2%).** Too sparse to carry
  a fleet-wide feature, whatever the literature says about end-to-end errors.
- **Workload normalization** (`smart_240/241/242`, defects per TB written):
  weak univariately (0.063 / <0.032 / 0.082) at 44–65% coverage. Note this is
  *unsupported*, not refuted: a ratio is an interaction, and a univariate
  screen structurally cannot see one. Deferred, not ruled out.

### Spatial clustering: cohort effect, not local correlation

| Grain | Drives/group | Groups | Variance ratio |
|---|---|---|---|
| `vault_pod` (chassis) | 61 | 5,176 | **1.33** |
| `vault_id` | 1,228 | 260 | **3.35** |
| `pod_id` | 18,273 | 20 | 2.54 |
| `cluster_id` | 33,167 | 7 | 247 |
| `datacenter` | 66,381 | 6 | 52 |

`pod_id` turned out to be the pod *index within a vault* (~20 values), not a
unique chassis; `vault_id` matches a Backblaze vault (20 pods x ~60 drives).
The pair identifies a physical chassis of ~61 drives, which is the grain at
which "the drives beside this one" means anything.

**Clustering weakens as the grain gets finer** — 1.33x at chassis against
3.35x at vault. A local physical mechanism (shared vibration, a failing PSU, a
hot pocket, a bad controller) would concentrate failures most tightly at
chassis level. It is lowest there. Over-dispersion that *grows* with group
size is the signature of cohort effects: Backblaze fills a vault with
identical drives bought as one batch, so "vaults differ" is mostly "vaults
hold different models and age cohorts". Drive model and age are already
featurized (`drive_model`, model-family z-scores, `drive_age_days`,
`power_on_hours`), so neighbour features would largely re-encode what the
model has. The 1.33x at chassis level is statistically real (~17 sigma at
5,176 groups) but tiny, and partly the same cohort effect leaking down, since
chassis within a vault share a model.

The `datacenter` and `cluster_id` ratios are not evidence of clustering: with
6-7 enormous groups they mostly reflect differing fleet composition. The
screen now marks grains with fewer than 30 groups as uninterpretable.

**Decision: no neighbour-failure features.** This was the only remaining idea
that would have added information rather than rearranged it.

### A live bug in the safety layer

Found while checking whether adding a 6th priority attribute would disturb
`feature_confidence`. Verified chain:

1. `smart_187`/`smart_188` coverage 0.32 (measured, above).
2. Nulls survive to gold: silver wrote 563,089,383 canonical rows from
   62,565,487 drive-days — exactly 9.000 per drive-day, one per ingested
   attribute — so a null reading is kept as a row, and the pivot's `.first()`
   preserves it as an all-null gold column.
3. `pipelines/build_gold_features.py` passed `priority_smart_attributes` — all
   five — to `add_feature_confidence`.
4. `attribute_coverage_factor = sum(is_not_null) / 5` = **3/5 = 0.60** for
   every non-Seagate drive-day.
5. `feature_confidence = telemetry_coverage_30d x recency_factor x
   attribute_coverage_factor <= 0.60`, however complete and fresh the
   telemetry.
6. `configs/agent.yaml` and `context.py` set
   `min_confidence_for_destructive_action = 0.80`, rule `FEATURE_CONFIDENCE`,
   severity **hard**; `rules.py` blocks destructive actions below it.

**Cordon, migrate and drain were therefore structurally impossible on roughly
two-thirds of the fleet** — not because those drives had degraded telemetry,
but because they do not report another vendor's attributes. No test caught it:
every guardrail and chaos fixture builds synthetic drives with all attributes
populated. In operation it is silent — the guardrail fires, the action routes
to human review, the audit trail looks healthy — while the system is unable to
act autonomously on most of the fleet, which is the project's whole purpose.

## Decisions

1. **`attribute_coverage_factor` is scored per drive model.**
   `src/features/cross_vendor.py::model_expected_attributes` computes which
   attributes each model actually reports (non-null in >= 50% of that model's
   drive-days, with a 1,000 drive-day minimum before a model's own set is
   trusted). `add_feature_confidence` counts only expected attributes, so a
   Seagate drive that stops reporting `smart_187` is still marked down (4/5)
   while an HGST drive reporting everything it has scores 3/3 = 1.0.
   - Computed fleet-wide, matching the `model_family_zscore_stats` precedent.
     An attribute set is a property of hardware and firmware, independent of
     any label, so it carries no outcome leakage.
   - A model below the drive-day minimum, or absent from the table entirely,
     has every attribute marked expected. That is the conservative direction:
     it can only lower confidence, never raise it above what the drive earned.
   - The `_expected_*` flags are dropped before returning, because `pl.Boolean`
     counts as a feature dtype (`src/models/features.py::NUMERIC_DTYPES`) and
     they would otherwise become silent model inputs.
   - **The 0.80 floor is unchanged.** The bug was the denominator, not the
     threshold.
2. **`smart_196` is ingested as a priority attribute** (`reallocation_event_count`),
   on the measured separation above. `smart_22` (helium, sep 0.169 over the
   34% helium fleet) is **not done**: it is the first attribute whose raw value
   is *inverted* (lower helium is worse), and
   `src/preprocess/smart_mapping.py` currently documents direction
   normalization as identity for every onboarded counter. Onboarding it needs
   that path built and tested, which belongs in its own change.
3. **Per-family operating points: not done.** The guardrail's instinct was
   sound even though its mechanism was wrong — prediction quality genuinely
   differs by family (pooled drive-level AUPRC 0.303 Seagate, 0.217 HGST,
   0.207 Toshiba), consistent with `smart_187`/`188` availability though not
   proof of it. The right way to express "require more confidence before
   draining a drive we predict less well" is a per-family threshold on the
   pooled model, not a blanket block. Note this is distinct from per-family
   *models*, which were tested and lost to pooled (0.128 vs 0.217 on HGST).
   Falsifiable check for the next run: if attribute availability causes the
   per-family gap, adding `smart_196` should narrow it.

## Consequences

- Destructive actions become possible on the non-Seagate fleet. This is a
  behavioural change in a safety system: the `feature_confidence` distribution
  before and after, specifically how many drive-days cross 0.80, should be
  inspected on a real build before this goes near a real fleet.
- `tests/unit/test_vendor_attribute_confidence.py` pins the whole chain,
  including a case asserting the pre-fix 0.60 value is still blocked, so a
  regression to the global denominator fails the suite.
- The gold table grows by ~31 columns (one priority attribute). The guard
  tests in `tests/unit/test_feature_tiers.py` were updated deliberately, and
  still bound the total.
- Requires a full re-ingest -> silver -> features -> labels -> train, because
  the bronze layer must pick up `smart_196_raw`. The frozen run recorded in
  `docs/model_status_and_runbook.md` is invalidated and must be re-frozen.
  Doing this before Q3 is ingested costs nothing; afterwards it would cost the
  sealed evaluation.
- The precision ceiling is unaffected by any of this. The screen found no
  hidden strong attribute — the best un-ingested candidate merely matches what
  is already there — which is further evidence for the standing conclusion
  that the ceiling is in the data rather than in the model or the features.
