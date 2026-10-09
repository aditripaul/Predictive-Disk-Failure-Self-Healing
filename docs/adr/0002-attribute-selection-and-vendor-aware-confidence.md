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
at double that coverage, reaching the drives where those two are null -
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
  the reporting cutoff (< 0.032) - weaker than `smart_7`, already known weak.
  Textbook SMART lore, not this fleet. Not added.
- **Vendor-normalized values.** Every `_normalized` column matched or
  underperformed its raw twin; `smart_5` loses most of its signal (0.290 raw
  vs 0.098 normalized) because vendor scaling compresses it. Not added as
  features. They are still required in the ingest for SMART-Z harmonization,
  which publishes only normalized values - a comparability mechanism, not a
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

**Clustering weakens as the grain gets finer** - 1.33x at chassis against
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
   62,565,487 drive-days - exactly 9.000 per drive-day, one per ingested
   attribute - so a null reading is kept as a row, and the pivot's `.first()`
   preserves it as an all-null gold column.
3. `pipelines/build_gold_features.py` passed `priority_smart_attributes` - all
   five - to `add_feature_confidence`.
4. `attribute_coverage_factor = sum(is_not_null) / 5` = **3/5 = 0.60** for
   every non-Seagate drive-day.
5. `feature_confidence = telemetry_coverage_30d x recency_factor x
   attribute_coverage_factor <= 0.60`, however complete and fresh the
   telemetry.
6. `configs/agent.yaml` and `context.py` set
   `min_confidence_for_destructive_action = 0.80`, rule `FEATURE_CONFIDENCE`,
   severity **hard**; `rules.py` blocks destructive actions below it.

**Cordon, migrate and drain were therefore structurally impossible on roughly
two-thirds of the fleet** - not because those drives had degraded telemetry,
but because they do not report another vendor's attributes. No test caught it:
every guardrail and chaos fixture builds synthetic drives with all attributes
populated. In operation it is silent - the guardrail fires, the action routes
to human review, the audit trail looks healthy - while the system is unable to
act autonomously on most of the fleet, which is the project's whole purpose.

### A second, larger cap in the same formula

Found by running `scripts/check_confidence_distribution.py` on the rebuilt
gold table after the fix above. With `attribute_coverage_factor` now 1.00 for
nearly every model, **0.0% of 62,565,487 drive-days still reached the 0.80
floor** - so the first fix was real but not sufficient, and the earlier claim
that the agent could not act on "about two thirds" of the fleet was wrong. It
could not act on **any** of it.

`src/preprocess/telemetry_gaps.py` defines
`days_since_last_telemetry = date - date.shift(1).over("drive_id")`: the
interval BETWEEN consecutive readings. That is the right quantity for
`telemetry_gap_flag` and `stale_telemetry_flag`, which use it. But
`add_feature_confidence` consumed it as the AGE of the current reading:

    hours_since_last = days_since_last_telemetry * 24.0   # = 24 for daily data
    recency_factor   = exp(-24 / 48) = 0.6065

Every gold row is a reading taken on its own date, so its age is zero. The
formula charged a full day of staleness for perfectly fresh daily telemetry.

The two surviving factors are also anti-correlated, which makes the 0% result
structural rather than marginal:

| recency_factor | when | telemetry_coverage_30d | product |
|---|---|---|---|
| 0.6065 | every day after a drive's first | up to 1.0 | <= 0.6065 |
| 1.0 | a drive's first day only (`fill_null(0)`) | 1/30 = 0.033 | 0.033 |

Clearing 0.80 at recency 0.6065 would need coverage >= 1.32, and coverage is
clipped at 1.0. No drive-day could satisfy both terms.

Measured on the real build: `recency_factor` median 0.6065, p95 0.6065, max
1.0 (the 363,548 first-day rows), `telemetry_coverage_30d` median 1.0,
`attribute_coverage_factor` median 1.0.

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
2. **Recency charges only the time beyond the expected cadence.**
   `expected_cadence_days: 1` (`configs/features.yaml`, daily in both
   Backblaze and SMART-Z), and
   `recency_factor = exp(-max(0, days_since_last - cadence) * 24 / tau)`. A
   reading one day after the previous one is fresh and scores 1.0; a three-day
   gap is charged two days and scores exp(-48/48) = 0.37, so it stays blocked.
   The 0.80 floor becomes meaningful instead of unreachable and needs no
   change. `hours_since_last_telemetry` keeps reporting the true interval,
   because `src/models/serving.py` surfaces it.
   - Not done, and a genuinely different quantity: at serving time the age of
     the latest reading relative to `as_of` is what matters, and `serving.py`
     currently reads the feature-table column rather than computing it against
     the decision time. For a live agent that distinction matters; for an
     offline daily-snapshot build it does not.
3. **The gold build refuses a fleet that can never act.**
   `_check_confidence_is_attainable` fails `build_gold_features` when NO
   drive-day reaches the floor, and logs the share and maximum otherwise. Two
   bugs of this exact shape - a multiplicative factor silently pinning the
   whole fleet below the floor - got through because every unit and chaos
   fixture builds drives with ideal values. Only real data exposes it. Zero is
   the sole failing value, since a genuinely stale fleet may legitimately have
   a low share.
4. **`smart_196` is ingested as a priority attribute** (`reallocation_event_count`),
   on the measured separation above. `smart_22` (helium, sep 0.169 over the
   34% helium fleet) is **not done**: it is the first attribute whose raw value
   is *inverted* (lower helium is worse), and
   `src/preprocess/smart_mapping.py` currently documents direction
   normalization as identity for every onboarded counter. Onboarding it needs
   that path built and tested, which belongs in its own change.
5. **Per-family operating points: implemented and refuted on test, 2026-10-09.**
   `src/models/threshold.py::tune_action_tiers_by_family` gives each drive
   model its own action-tier thresholds, tuned on that model's own validation
   drives from the same lift targets, converted at that model's own failure
   rate - so a tier keeps one meaning across a heterogeneous fleet instead of
   averaging over it. A model needs at least
   `threshold.min_family_failing_drives` (50) failing validation drives to
   earn its own thresholds; below that it keeps the fleet-wide ones, because
   recall moves in steps of 1/n on a family's curve and a threshold fitted to
   a handful of drives is fitted to noise.
   `resolve_family_thresholds` is the single lookup, so scoring, serving and
   evaluation cannot drift apart on the fallback rule.

   **Measured on test and refuted.** 7 of 81 drive models cleared the gate,
   74 kept the fleet-wide thresholds. Per-family against fleet-wide, same test
   drives and same scores:

   | Tier | Recall delta | Precision delta |
   |---|---|---|
   | warn | -0.069 | +0.005 |
   | cordon | -0.081 | +0.010 |
   | migrate | -0.077 | +0.007 |
   | drain | -0.019 | **-0.099** |

   Seven to eight points of recall traded for under one point of precision at
   the lenient tiers, and at **drain** it is strictly worse on both axes: 64
   caught from 173 alerts against 76 from 162. Worse recall AND more alerts,
   at the tier with the most consequential action.

   Two causes:

   - **Overfitting, worst where it matters most.** At 50 to 100 failing
     validation drives a family's curve has 1 to 2 percent recall resolution,
     and the drain target (226x lift) sits at the top of that curve where only
     a handful of drives clear the threshold. The tuned cut-off is fitted to
     noise precisely at the drain tier. More data per family, not a better
     tuner, is what this would need.
   - **Per-family *lift* was the wrong target.** Equalising lift means a
     family with a higher base rate needs higher precision for the same lift,
     so it alerts *less* aggressively while a low-risk family alerts more -
     the inverse of the sensible ordering, and wrong for action cost, since a
     false drain costs the same whatever the family's base rate. A per-family
     *precision* target would be better posed, but it would still meet the
     first cause, which the drain row shows is severe. Not attempted.

   **The original inference was unsound, and that is the durable lesson.**
   Families sitting at different operating points under one threshold is the
   expected behaviour of a calibrated score meeting populations of differing
   risk - not evidence that the threshold is mis-set. Thresholding a
   well-calibrated P(fail) globally is already optimal for recall at a fixed
   precision; per-family thresholds can only help where the score is
   *differentially miscalibrated by family*, and this test says it is not,
   enough to matter. The 0.508 precision at 0.492 recall seen on
   ST12000NM0008 was that family's position on the pooled frontier, not a
   gain waiting to be unlocked.

   The code is kept because it is the measurement, not because the thresholds
   are used: `train_model` reports the comparison and nothing consumes
   `action_tiers_by_family` thresholds. Serving and the agent still apply
   fleet-wide cut-offs and **should not** be rewired to these.

6. **Why per-family thresholds, and the evidence for them.** The guardrail's
   instinct was sound even though its mechanism was wrong - prediction quality genuinely
   differs by family (pooled drive-level AUPRC 0.303 Seagate, 0.217 HGST,
   0.207 Toshiba), consistent with `smart_187`/`188` availability though not
   proof of it. The right way to express "require more confidence before
   draining a drive we predict less well" is a per-family threshold on the
   pooled model, not a blanket block. Note this is distinct from per-family
   *models*, which were tested and lost to pooled (0.128 vs 0.217 on HGST).
   Falsifiable check for the next run: if attribute availability causes the
   per-family gap, adding `smart_196` should narrow it.
   - **Tested 2026-10-09, not supported.** On run `29e338248d8948ccb41ee58db5321e76`
     (246 features) the pooled model's drive AUPRC is 0.473 on
     ST12000NM0008 against 0.244 on HGST HUH721212ALN604; the gap did not
     narrow. The headline also did not move: test drive precision 0.486
     against the previous 0.496, recall 0.087 against 0.105, row AUPRC 0.1108
     against 0.1115 - all inside the bootstrap interval.
   - **The experiment was confounded, by this ADR's own changes.** Three things
     moved at once: `smart_196` was added, `attribute_coverage_factor` went
     from a 0.6/1.0 vendor split to about 1.0, and `recency_factor` went from
     about 0.6065 to 1.0. The second matters more than it looked when shipped:
     `attribute_coverage_factor` was an accidental vendor-identifying feature,
     and the fix removed that signal from the model's inputs. Nothing here can
     be attributed to `smart_196` alone. Bundling a feature experiment with
     two bug fixes was a mistake; the lesson is to land fixes and measure
     features in separate runs.
   - **What the run did show, strongly.** At a single global threshold the
     pooled model sits at completely different operating points per family:
     fleet-wide 0.394 precision at 0.177 recall, but 0.733/0.129 on TOSHIBA
     MG07ACA14TA, 0.508/**0.492** on ST12000NM0008 and 0.360/0.158 on HGST
     HUH721212ALN604 - a 3.8x spread in recall and 2x in precision at an
     identical cut-off (`per_model_experiment` evaluates the pooled model at
     the pooled threshold, not a per-family one). ST12000NM0008 reaching 49%
     recall at 51% precision, against 8.7% recall fleet-wide, is the strongest
     evidence yet for per-family operating points, and the first lever in this
     work that looks capable of moving the frontier rather than nibbling at
     it. It raises recall by not averaging a single threshold over
     heterogeneous populations, without improving the model at all.

## Consequences

- Destructive actions become possible on the non-Seagate fleet. This is a
  behavioural change in a safety system, so the distribution was inspected on a
  real build before accepting it (`scripts/check_confidence_distribution.py`,
  62,565,487 drive-days, 2026-10-09):

  | | Before both fixes | After |
  |---|---|---|
  | `recency_factor` median | 0.6065 | 1.0000 |
  | max `feature_confidence` | 0.6065 | 1.0000 |
  | drive-days at or above 0.80 | 0.0% | **86.1%** |
  | newly blocked | - | **0.00%** |

  Holding the corrected recency fixed, the per-model denominator alone moves
  27.2% -> 86.1%, so it unblocks 58.9% of drive-days. Nothing is newly
  blocked, as expected: a per-model denominator is never larger than the
  global one.

  The ~13% that remains blocked is correct, not residual breakage. A drive
  needs at least 24 of the trailing 30 days of telemetry to reach
  `telemetry_coverage_30d >= 0.80`, so its first 23 days are below the floor;
  over the 181-day window that is 23/181 = 12.7%, against an observed 13%. A
  drive observed for two weeks should not be drainable.

  `WDC WUH722222ALE6L4`, the largest model in the fleet at 8.1M drive-days,
  went from 0% to 87% actionable.
- **Known remaining gap, ~0.3% of drive-days.** SSDs and laptop drives
  (`Seagate SSD`, `WD Blue SA510`, `TOSHIBA MQ01ABF050`, `MTFDDAV240TCB`,
  `HGST HMS5C4040BLE640`) sit at median attribute coverage 0.00-0.67 and stay
  below the floor. Part of this is real - they do not report HDD defect
  counters, and an HDD defect model arguably should not act on them - but part
  is the 50% expectation threshold being too permissive: an attribute present
  on just over half a model's drive-days counts as "expected" and is then
  scored as missing on the other half. Raising the threshold (an attribute
  counts as expected only if nearly always present) would fix that, and is
  deferred because it affects 0.3% of drive-days and no HDD model.
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
  hidden strong attribute - the best un-ingested candidate merely matches what
  is already there - which is further evidence for the standing conclusion
  that the ceiling is in the data rather than in the model or the features.
