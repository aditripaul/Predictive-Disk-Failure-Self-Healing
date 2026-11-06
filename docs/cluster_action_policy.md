# Cluster-based action policy for MAPE-K

Status: **run completed 2026-11-06; the rule was not passed, so the agent is
unchanged.** Sections 1-6 were written and committed (`8dcca9b`) before the Q3 step
of `scripts/cluster_action_policy.py` ran; sections 7-10 are filled from the saved
outputs, and every number names the file it came from
(`data/audit/cluster_policy_h30/`).

## 1. Question

The two-stage model gives each drive a score, and the agent turns the score into
an action tier (warn, cordon, migrate, drain). The score says **how likely** a drive
is to fail, not **what kind of problem** it has. The deep dive (runbook 1.1.5)
showed that false alarms are drives with real SMART damage (98-99.8% of them at
warn and above have a non-zero core counter), older than the fleet, and that the
stricter the tier the more of its false alarms were removed from service anyway.

Does a failure-mode cluster, crossed with the score tier, give the agent a better
action than the tier alone: fewer unnecessary cordons and drains, no lost catches,
and a recommendation of the right *kind* of action?

## 2. Where it sits in the system

```
Stage 1 (LightGBM, ~80% of failing drives in ~6% of the fleet)
  -> Stage 2 (LightGBM, the best Stage 2 kind; runbook 1.1.4)
  -> combined score -> tier          (runbook 1.9: how the tiers are set)
  -> SMART signature -> cluster       (this document, section 3)
  -> MAPE-K plan: (tier, cluster) -> effective tier and action kind
  -> hysteresis, cooldown, control arm, guardrails (unchanged)
```

The MAPE-K loop, step by step, with what this policy changes:

| Step | Today (`src/agent/`) | With the cluster policy |
|---|---|---|
| Monitor | `fleet.py::load_fleet_snapshot`: each drive's latest gold row | unchanged |
| Analyze | score per drive (`fleet.py::score_snapshot`); `to_candidate_drives` keeps `drive_id, p_fail, feature_confidence, feature_maturity, stale_telemetry` | adds `cluster` from the frozen cluster model (`ClusterModel.assign` on the same gold row) |
| Plan | `nodes.py::make_plan_node`: the top drive's tier from `models/action_tiers.py::determine_action_tier`, then hysteresis, cooldown, control arm, guardrails | the tier is looked up in the action map for its `(cluster, tier)` cell **before** the confidence / maturity / staleness gates, which still bind; the proposal carries `cluster` and `action_kind` |
| Execute | simulated (`fleet.py`) | unchanged |
| Knowledge | trained thresholds (`action_tiers.json`) | plus `cluster_model.json`, `action_map.json` and the validation matrix |

The integration (Phase B) is built only if the offline comparison passes the rule
of section 6. Until then the policy exists as a measured proposal and the agent is
unchanged.

## 3. The clusters

**Who is clustered.** The Stage 1 suspects of the frozen 30-day run
(`frozen_runs/c459046e.../`): drives whose drive score (`drive_level_table`'s
convention) reaches the candidate threshold, 0.235. Clusters are fitted on the
**validation** suspects only.

**On which row.** A suspect is clustered on its gold row of the day it first passed
Stage 1 (fitting), and placed in the matrix by its row on its *decision day*: the
first day its score reached its highest tier's threshold, the row the agent would
act on.

**Signature** (`src/models/failure_clusters.py::signature_frame`), all known on the
decision day: the six core counters (log-scaled), their 30-day deltas and slopes,
days since each last rose, whether each just became non-zero; active defects and
their velocity and acceleration; CRC errors and their change; seek-error change;
temperature and temperature spikes; power-on age; telemetry coverage; pending vs
reallocated ratio. **Excluded on purpose:** the score (the clusters would just
repeat the score bands), any label, and the drive vendor or model (vendor would
dominate; family is reported per cluster instead). Columns are scaled by median
and interquartile range, clipped to +-10.

**Algorithm ablation** (validation only). k-means, Gaussian mixture (full
covariance), agglomerative (Ward), HDBSCAN (noise as its own cluster) and
spectral clustering (15-nearest-neighbour graph), over 3 to 10 clusters (HDBSCAN:
minimum cluster size 50, 100, 200). Spectral clustering was first specified with an
RBF kernel; on the real validation suspects that version did not converge in two
minutes (a local-scale kernel width splits the graph into near-disconnected pieces
and the eigensolver stalls), so the neighbour-graph version replaced it before any
result was read. The first full run was stopped for the same reason after 39
minutes, and the ablation now saves each finished candidate and resumes from them. For each: failure-rate
separation, silhouette, stability, smallest cluster, noise share, time.
Kernel SVMs are not in it: supervised, a re-ranker on the same features cannot add
information, and a one-class SVM is the anomaly-score idea that gave nothing
(runbook 4.3).

**Selection rule, fixed in advance:** the largest **failure-rate separation within
band**, subject to stability (adjusted Rand index between models fitted on two
disjoint halves, mean of 5 draws) of at least 0.7 and no cluster below 2% of
suspects; ties to k-means. Separation is the between-cluster variance of the
failure indicator within each score band, measured *across halves* of the drives
(cell rates estimated on one half, deviations multiplied across halves), so it does
not grow with the number of clusters by chance. Labels are used here only to choose
the algorithm on validation, never on test. If no candidate is stable enough, the
most stable one is taken and the result says so.

**Assignment of a new drive** is nearest centroid (or the mixture's highest
posterior), plain numpy, so the agent can run it.

**Names and action kinds** come from each cluster's largest standardised features:
"inspect connection" when they are interface errors (CRC, command timeouts),
"migrate data" when they are pending or uncorrectable sectors, "replace" for other
damage, "watch" when nothing stands out. The kind is a recommendation; it does not
change any metric.

## 4. The action map

**Matrix.** Rows are clusters, columns are bands: monitor (passed Stage 1, below
warn), warn, cordon, migrate, drain - a drive in its highest band. The 30-day run's
migrate and drain thresholds are both 0.926 (runbook 1.9), so the migrate band is
empty and drain holds the top drives; both columns are kept, as decided.

**Rule** (`src/agent/cluster_policy.py::derive_action_map`), fixed before any test
row is read. For each cell with at least 30 validation drives, the effective band is
the highest band whose own validation failure rate the cell's failure rate matches
or beats, moved at most one band from the score band; a cell that beats no band
rate goes to monitor; cells below 30 drives keep their band.

**Why band rates, not the lift targets.** The tier thresholds were tuned so that the
*cumulative* set at or above a tier meets a lift-based precision target (warn 33.3%
on validation). A single band fails less often than that set, because the set also
holds every stricter drive; comparing a warn-band cell with 33.3% would demote
almost every warn cell. Comparing a cell with the bands' own rates asks the right
question: does this cell's drives behave like warn drives, or like cordon drives?
(The plan first named the lift targets; this was corrected before any test row was
read.)

**What the map cannot claim.** Backblaze records failures and removals, not repairs,
so whether "inspect connection" was the right kind of action is not measurable
here. The matrix reports each cluster's failure and removal rates beside its kind.

## 5. How actions are judged

The executor is simulated, so an action cannot change a drive's fate: the label is
the counterfactual. Per drive, after its decision day
(`src/agent/cluster_policy.py::add_outcomes`, follow-up to 2026-09-30):

| Outcome | Meaning |
|---|---|
| correct | a failing drive (failed within 30 days of a scored row) |
| early | a healthy-labelled drive that failed within 60 days of the decision day |
| later | failed after that |
| removed | left service without a recorded failure |
| active | still in service |

Per tier, counted cumulatively, for the tier-only policy and the cluster policy on
the same drives (`compare_policies`): flagged, caught, false alarms, precision,
recall (over **all** failing drives, so it matches runbook 1.1.4), recall with at
least 7 days' lead, median lead, an illustrative cost (misses x 20 + false alarms),
drives moved down (and how many of them were failing: lost) and up (gained), the
outcome mix, and a paired drive bootstrap (1,000 resamples) of the precision and
recall gaps.

Figures (`data/audit/cluster_policy_h30/figures/`, redrawn from the JSON without
reading data): clustering ablation, cluster signatures, family mix per cluster,
the tier x cluster matrix on validation and on Q3 (moved cells outlined), action
outcomes per tier, policy comparison with intervals, where each policy operates on
the precision-recall curve, lead time per cluster, cluster stability, and how the
tiers are set (validation curve with the lift-derived targets; lift x base rate).

## 6. Rule fixed in advance

**Building the agent integration (Phase B)** goes ahead only if, on Q3:
1. at **warn** and at **cordon**, the cluster policy's precision is above the
   tier-only policy's with the paired interval above 0, and its recall is not lower
   (the paired recall interval includes 0 or lies above it);
2. the clusters are stable: every cluster's share of suspects on Q3 is within a
   factor of 2 of its validation share, and a refit on Q3 agrees with the assignment
   (adjusted Rand index >= 0.5);
3. no cell's action changed on the strength of fewer than 30 validation drives.

Otherwise the result is recorded as "clusters describe failure modes but do not
improve the policy" and the agent is not changed. On a future untouched period, the
same three checks would confirm a policy adopted here.

## 7. Results

### 7.1 Clustering ablation (validation suspects, no test row read)

4,729 suspects (335 failing) of 70,977 validation drives, 42 signature columns
(`tables.md`, `validation.json` `ablation`; figure `figures/clustering_ablation.png`).
35 candidates: k-means, Gaussian mixture, agglomerative and spectral at 3-10
clusters, HDBSCAN at minimum sizes 50, 100, 200. Separation is small for every
candidate (about 1e-4 to 1e-3), so differences between neighbouring candidates are
not shown to be meaningful.

| Candidate | Separation | Stability (ARI) | Smallest cluster | Qualifies? |
|---|---|---|---|---|
| **GMM, 4 clusters (chosen)** | 9.5e-4 | 0.83 | 9.6% | yes |
| GMM, 8 clusters (highest separation) | 1.06e-3 | 0.68 | 1.4% | no: stability, size |
| GMM, 10 clusters | 1.01e-3 | 0.75 | 1.5% | no: size |
| Agglomerative, 10 clusters | 9.1e-4 | 0.84 | 1.1% | no: size |
| k-means, best qualifying (5) | 4.8e-4 | 0.85 | 2.8% | yes |
| Spectral (neighbour graph), best stability (4) | 2.1e-5 | 0.80 | 6.4% | yes, far lower separation |
| HDBSCAN | 1.8e-4 - 3.6e-4 | 0.79 - 1.00 | 4.1% - 6.7% | noise 53-57% of suspects |

Spectral clustering with an RBF kernel did not converge in two minutes on the real
data and was replaced by the neighbour-graph version before any result was read
(section 3). The first full run was stopped after 39 minutes for that reason; the
ablation was then made resumable.

### 7.2 The clusters and the action map (validation)

| Cluster | Name | Share of suspects | Failure rate | Action kind |
|---|---|---|---|---|
| C0 | no clear damage | 53.5% | 1.1% | watch |
| C1 | reallocated sectors rising, defect acceleration | 14.9% | 4.7% | replace |
| C2 | pending sectors, pending vs reallocated | 22.1% | 11.7% | migrate data |
| C3 | reallocated sectors rising, defect velocity | 9.6% | 33.4% | replace |

Clusters alone separate risk from 1% to 33%. The map moved three of 13 cells, all
**down**, none up: C1 at warn (38 drives, 5.3% failing), C2 at warn (247, 14.2%) and
C2 at cordon (80, 48.8%) (`action_map.json`; figures `cluster_signatures.png`,
`tier_cluster_matrix_validation.png`). In sample on validation the map raises warn
precision by 11.3 points and lowers warn recall by 8.9.

### 7.3 Q3 2026 test (one look: test-access log line 133, commit `8dcca9b`)

`q3_evaluation.json`, `tables.md`. Cumulative tiers, per drive, 1,105 failing and
367,583 healthy drives; paired drive bootstrap, 1,000 resamples.

| Tier | Tier-only P / R / flagged | Cluster P / R / flagged | Precision gap [95%] | Recall gap [95%] | Moved down (failing) | Cost 20:1 |
|---|---|---|---|---|---|---|
| warn | 17.3% / 60.9% / 3,881 | 25.3% / 52.1% / 2,275 | +8.0% [+7.1, +8.9] | **-8.8% [-10.5, -7.1]** | 1,606 (97) | 11,848 -> 12,279 |
| cordon | 34.8% / 42.4% / 1,346 | 39.4% / 36.2% / 1,015 | +4.6% [+3.1, +5.9] | **-6.2% [-7.7, -4.9]** | 331 (69) | 13,597 -> 14,715 |
| migrate = drain | 33.3% / 0.2% / 6 | same | 0 | 0 | 0 | unchanged |

- **Stability:** cluster shares on Q3 are 49.6 / 14.3 / 23.6 / 12.4% against 53.5 / 14.9
  / 22.1 / 9.6% on validation, all within a factor of 2; a refit on Q3 agrees with
  the assignment (ARI 0.87). Failure rates fell by about half (C3 33.4% -> 16.1%,
  C2 11.7% -> 4.5%), as the fleet's did (0.59% -> 0.30%), with the order C3 > C2 >
  C1 > C0 intact (`figures/cluster_stability.png`).
- **The validation trade held out of sample:** precision +8.0 points (validation
  +11.3) and recall -8.8 (validation -8.9) at warn.
- **It does not beat the score's own curve.** At the recall each policy reaches, a
  plain stricter threshold on the combined score gives 26.0% precision at 52.1%
  recall (warn; the cluster policy gets 25.3%) and 42.5% at 36.2% recall (cordon;
  the cluster policy gets 39.4%): the cluster policy's operating points lie on or
  below the precision-recall curve (read from the curve saved in
  `q3_evaluation.json`, thinned to about 2,000 points; `figures/policy_precision_recall.png`).
- **Break-even:** demoting at warn avoided 1,509 false alarms for 97 lost failing
  drives, so it pays only if one missed failure costs less than about 16 false alarms
  (the illustrative 20 loses); at cordon, 262 for 69, break-even 3.8.
- Most drives the policy dropped were still running (`figures/action_outcomes_by_cell.png`).

Figures: `figures/policy_comparison.png`, `tier_cluster_matrix_test.png`,
`lead_time_by_cell.png`, `cluster_families.png`, shown in runbook 1.1.6.

## 8. Decision

**Rule not passed** (`q3_evaluation.json` `rule`): precision is up at warn and
cordon, but recall is lower at both (the recall clause fails), the clusters are
stable (shares within 2x, ARI 0.87) and no cell moved on fewer than 30 drives.
**The agent integration (Phase B) is not built** and the agent is unchanged.

The validation result already pointed this way, and the recall clause could not
be met by a policy that only demotes cells containing failing drives; the rule was
run as written and not changed. What survives: the four clusters are stable,
readable failure modes with very different risk (C3: 12% of Q3 suspects, 16%
failing; C0: half of the suspects, 0.7%), usable as a description and a watch list.
They do not improve the tiers: the precision they buy at warn and cordon is the
precision a stricter score threshold buys, at a better price.

## 9. Provenance

| Item | Value |
|---|---|
| Method and rule commit (before the Q3 look) | `8dcca9b` |
| Run | frozen 30-day run `c459046e4f1645169151e7be25ff064d` |
| Outputs | `data/audit/cluster_policy_h30/` (`validation.json`, `cluster_model.json`, `action_map.json`, `q3_evaluation.json`, `ablation_partial.json`, `tables.md`, `figures/`) |
| Test accesses | `data/audit/test_access_log.jsonl` line 133 (the step before it, `--validation-only`, records none; lines 130-132 are another session's stacked-Stage-2 runs) |
| Commands | runbook 1.1.6 |
| Wall times | validation about 12 min (35 candidates; Gaussian mixtures 46-109 s each, the rest seconds), after a first run stopped at 39 min; Q3 step about 5 min |

## 10. Threats to validity

- **One test quarter**, and the fleet's failure rate halved against validation, so
  per-cell failure rates are not comparable between the two splits.
- **The clusters are fitted on the same holdout the thresholds came from,** and labels
  choose the algorithm on validation; separations between candidates are tiny and
  their differences were not tested.
- **The map's rule compares cells with band rates,** not with a cost break-even; a
  cost-based rule would demote fewer cells (none above about 5% failing at 20:1).
  It was not tried: the rule was fixed before the data was seen.
- **Action kinds are unverifiable:** Backblaze records failures and removals, not
  repairs.
- **Cells near the 30-drive floor** (C0 at warn and cordon, drain) kept their band
  and carry no result.
- **The executor is simulated;** outcomes are what happened to the drive afterwards.
- **The curve comparison is read from a thinned curve** (about 2,000 points); gaps
  under a point are within its resolution.
