# Cluster-based action policy for MAPE-K

Status: **method fixed, Q3 not yet read** (sections 1-6 written 2026-11-06, before
the Q3 step of `scripts/cluster_action_policy.py` ran). Sections 7-10 are filled
from the saved outputs afterwards, never typed by hand: every number names the file
it came from.

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

To be filled from `data/audit/cluster_policy_h30/` after the Q3 step.

## 8. Decision

To be filled.

## 9. Provenance

To be filled: commits, output files, test-access line, wall times.

## 10. Threats to validity

To be filled after the run. Known in advance: one test quarter; the clusters are
fitted on the same holdout the thresholds came from; labels choose the algorithm on
validation; action kinds are unverifiable; cells near the 30-drive floor; the
executor is simulated.
