# Wide-net cascade: method and experiment record

Status: run completed 2026-11-06 (MLflow `c05a109a5af04e749e3a9ea715ad2823`).
Decision under the pre-registered rule: **keep the 0.5 net** (Section 9.3). This document is written to be cited: every number names the run,
the code and the data it came from, and the decision rule (Section 6) was written
down before any test result was seen.

## 1. Question

Can a two-stage cascade raise precision at the **high-recall** operating points
(50-80% of failing drives caught), where an operator who prioritises catching
failing drives wants to work?

- **Stage 1** is a broad, high-recall filter.
- **Stage 2** is a validator trained only on Stage 1's suspects.

The repository already runs such a cascade. Its Stage 1 net, however, keeps only
50% of failing drives (`model.two_stage.candidate_recall: 0.5`). Every operating
point above 50% recall is therefore Stage 1 alone. This experiment widens the net
and measures whether Stage 2 then helps where it previously could not reach.

**Hypotheses**

- H1: for a net wider than 60%, the cascade's test precision at 60% recall is
  higher than Stage 1's alone.
- H0: there is no gain.

The prior expectation, stated before the run, was a small or null gain, for the
reason given in Section 3.

## 2. The cascade as implemented

Code: `src/models/two_stage.py`. Notation:
- `s1(x)` is the Stage 1 score of a drive-day row.
- `s2(x, s1)` is the Stage 2 score.
- `T1` is the candidate threshold.

1. **Candidate threshold.** `T1` is chosen on the **validation** split. It is the
   most precise drive-level threshold whose recall is at least `r_net` (the net).
2. **Out-of-fold Stage 1 scores for the train rows.** Train drives are hashed into
   3 folds. Each fold is scored by a Stage 1 model fitted on the other two folds,
   using Stage 1's tree count. In-sample scores would be far more confident than
   the scores Stage 2 meets in use, so Stage 2 would learn from a different suspect
   pool.
3. **Stage 2 training set.** Train rows whose out-of-fold `s1 >= T1`.
4. **Stage 2 inputs.** The same 246 features plus `s1` as one extra feature.
5. **Stage 2 early stopping.** It stops on the validation candidate rows: all
   positive rows plus at most 1.5M negatives (`early_stopping_max_negatives`).
6. **Combined score:**
   - `s = s1` if `s1 < T1`;
   - `s = T1 + (1 - T1) * s2` otherwise.

   Candidates always rank above non-candidates and are ordered among themselves
   by Stage 2. One score therefore serves every alert tier. For any threshold
   `t >= T1`, flagging `s >= t` is the textbook rule "`s1 >= T1` **and**
   `s2 >= (t - T1)/(1 - T1)`".

Two consequences follow directly from this construction:

- **Recall ceiling (the "leaky bucket").** No threshold on `s` can catch a failing
  drive whose rows all score below `T1` on Stage 1. Stage 1's recall at `T1` is
  the most the cascade can reach. The experiment reports it on test.
- **Reach.** Only operating points whose threshold is at or above `T1`, that is
  recall levels `<= r_net`, can change. At recall level `= r_net` the threshold
  *is* `T1`, every candidate is flagged, and the cascade equals Stage 1 by
  construction. That row is kept only as a check.

## 3. Pre-experiment measurement: what a wide Stage 1 net costs

**Source.** Validation split of the frozen merged run `fb58d957` (score table
`merged_scores_h30_h30_seed0.parquet`), column `lightgbm`. Below `T1` this column
equals Stage 1's score, so the table describes Stage 1 for every net of 50% or
more.

**Population.** 1,124 failing and 347,091 healthy validation drives.
15,359,094 rows, of which 0.105% are positive. Test was not used.

| Stage 1 recall | Healthy drives passed | Drive precision | Candidate rows (share) | Positive share of candidate rows (enrichment over 0.105%) |
|---|---|---|---|---|
| 50% | 1,245 (0.36%) | 31.1% | 19,830 (0.13%) | 22.3% (212x) |
| 60% | 2,461 (0.71%) | 21.5% | 39,109 (0.25%) | 15.5% (147x) |
| 70% | 4,578 (1.32%) | 14.7% | 84,842 (0.55%) | 9.5% (90x) |
| 80% | 14,956 (4.31%) | 5.7% | 359,524 (2.34%) | 3.0% (29x) |
| 90% | 103,451 (29.8%) | 0.97% | 2,177,109 (14.2%) | 0.62% (6x) |
| 95% | 193,416 (55.7%) | 0.55% | 5,655,620 (36.8%) | 0.26% (2.5x) |
| 98% | 294,879 (85.0%) | 0.37% | 10,798,499 (70.3%) | 0.14% (1.3x) |

The LightGBM + AFT + TCN rank blend has the same shape: at 95% recall it passes
58.3% of healthy drives.

**Reading.**

The cascade's premise is **enrichment**: Stage 2 learns on a pool far richer in
failures than the raw data. Enrichment collapses past about 80% recall.

The last ~20% of failing drives score like healthy drives, because they show no
SMART signal before failure. A Stage 1 net of 95% or 98%, as in textbook
descriptions, would pass most of the fleet and leave Stage 2 with nearly the
original problem. The nets compared are therefore:

- **0.5:** the current setting and the baseline.
- **0.7 and 0.8:** before and at the knee.
- **0.9:** past the knee, as a "too wide" control.

95% was dropped on this evidence, which also keeps Stage 2 within the run's 20 GB
memory cap.

## 4. Data and protocol

| Item | Value |
|---|---|
| Source | Backblaze drive stats (SMART), quarter-aligned window (runbook 2.2) |
| Splits (time-ordered) | train to 2026-01-15; validation 2026-02-15 to 2026-03-31; test 2026-04-01 to 2026-05-31 |
| Label | row positive if the drive fails within 30 days (`horizon_days: 30`) |
| Rows | train 5,002,092 (capped from 32,342,482; 24,560 positive); validation 15,359,094 (16,179); test 21,297,371 (26,523) |
| Features | 246 (`frame_meta.json`), float32 |
| Feature arrays | kept work dir `data/tmp/train_model_frame_bjdxfhj1/`, written by the `h30_seed0` training run (MLflow `839d92bc`) |
| Stage 1 | LightGBM, pipeline params + `REGULARIZED` (`pipelines/experiment_model.py`), positive weight (neg/pos)^0.5, early stopping 50 rounds on a validation subsample (all positives + 1.5M negatives) |
| Stage 2 | LightGBM, same params with `min_child_samples: 50` (`model.two_stage.stage2_params`), Stage 1 score as extra feature |
| Out-of-fold folds | 3, by drive hash, seeded |
| Nets `r_net` | 0.5, 0.7, 0.8, 0.9 |
| Recall levels | 5, 10, 20, 35, 50, 60, 70, 80% (each net uses the levels `<= r_net`) |
| Seeds | 0, 1, 2. A seed sets the fold assignment, both models' sampling and the bootstrap resamples. Test drives are the same in every seed. |

**Stage 1 is refitted per seed.** It is not loaded from the frozen run, so its
numbers differ slightly from run `839d92bc`. Every comparison is **paired within
a seed**: both models share one Stage 1, one set of out-of-fold scores, and the
same test drives and bootstrap resamples. Within a seed, all four nets share that
Stage 1 as well.

**Threshold selection.** At each recall level, the threshold is the most precise
one on the validation drive-level curve that still has at least that recall. It
is chosen separately for Stage 1 and for the cascade, then applied unchanged to
test.

## 5. Metrics

All metrics are per drive, not per drive-day (`drive_level_table`):
- A **failing** drive has a positive row. Its score is the maximum over its
  positive-window rows.
- A **healthy** drive's score is the maximum over all its rows.

| Metric | Definition |
|---|---|
| Precision | caught failing drives / (caught + healthy drives flagged) |
| Recall | caught / failing drives |
| Recall with lead | as recall, but a failing drive counts only if a row at least `lead_min_days` = 1 day before the failure clears the threshold. A flag on the failure day gives no time to act. **At 1 day this equals recall by construction**: the positive label covers days 1-30 before a failure (validation: 16,179 positive rows, `days_to_event` 1 to 30, none at 0), so drive-level recall already means warned at least a day ahead. Found after the run; a 7-day lead would be informative and is not in this run. |
| Cost (illustration) | missed failures x 20 + false alarms. The 20:1 ratio is an assumption, not a measured cost. |
| Ceiling | Stage 1's test recall (and recall with lead) at `T1` |
| Drives passing Stage 1 | share of validation and test drives whose Stage 1 drive score is at least `T1`. A deployed cascade would monitor this share for drift. |
| Paired gap | test precision of the cascade minus Stage 1, with a 95% percentile interval from 1,000 drive-level bootstrap resamples. The resamples are identical for both models, every net and every level within a seed (`paired_flag_difference_ci`). The recall gap is reported likewise. |

## 6. Decision rule (fixed before results)

A wider net replaces 0.5 in `configs/model.yaml` only if **all** of the following
hold for that net:

1. At the 60% recall level, the cascade's test precision is higher than Stage 1's
   in **every** seed.
2. The paired-bootstrap interval of that gap lies above 0 in **at least two of
   three** seeds.
3. Its recall with lead at 60% is not lower than Stage 1's.
4. At the 10% and 20% levels (the strict tiers), its mean precision is not below
   the 0.5 net's. A wider net trains Stage 2 on a thinner pool and could lose the
   strict-tier gain the drain tier depends on.

Otherwise 0.5 stays, and the null result is recorded here and in runbook 4.12.
Adopting a new net means retraining the pipeline model under a new tag. That is a
separate, approved step.

## 7. How this differs from the textbook cascade

The design follows the usual two-stage pattern. Four common shortcuts give
optimistic or wrong numbers on this kind of data, and are avoided:

| Shortcut | Why it misleads here | Used instead |
|---|---|---|
| Choose `T1` and `T2` on the test set | Test becomes tuning data; results are optimistic | Thresholds from validation only; test is scored once per seed and logged (`data/audit/test_access_log.jsonl`) |
| Select Stage 2's train suspects with in-sample Stage 1 scores | In-sample scores are overconfident, so the train suspect pool differs from the live one | Out-of-fold scores by drive |
| Row-level metrics | A failing drive contributes ~14 near-identical positive rows (validation: 16,179 rows over 1,124 drives), inflating precision and recall | Drive-level metrics |
| Random row split | The same drive appears in train and test | Time-ordered splits |

Two further choices were considered and not adopted:
- **Different model families per stage.** Runbook 5.6: model family did not
  move the ceiling on this data.
- **Costlier features for Stage 2 only.** The 246 features already include
  rolling-window trends. The costly signals available here are the survival and
  sequence scores (AFT, Cox, LSTM, TCN). Feeding them to Stage 2 is a stacked
  cascade, the natural follow-up if a wide net helps.

## 8. Reproducibility

**Command:**

    make experiment-model EXPERIMENT_ARGS="--work-dir data/tmp/train_model_frame_bjdxfhj1 --variants --skip-baseline --two-stage"

Settings come from `model.two_stage` and `model.two_stage.experiment` in
`configs/model.yaml`, and the command line can override them
(`--candidate-recalls`, `--recall-levels`, `--two-stage-seeds`,
`--lead-min-days`, `--cost-ratio`, `--n-boot`, `--two-stage-out`).

**Code.**
- The run started from commit `f4772f3` with the cascade changes uncommitted.
- `settings.git_dirty` is `true`, and the exact diff is saved as `code.patch`
  beside the results.
- Applying `code.patch` to `f4772f3` rebuilds the code that ran.
- The commit that later records these changes is noted in Section 9.

**Outputs** (`data/audit/`, tracked):
- `data/audit/cascade_wide_net_h30/experiment_two_stage.json`: settings, every
  per-seed and per-net report, and the summary.
- `data/audit/cascade_wide_net_h30/tables.md`: the tables of Section 9.
- `data/audit/cascade_wide_net_h30/code.patch`
- `data/audit/cascade_logs/cascade_wide_net_h30.log`: full console log.

**Test accesses.**
- Line 110 of `test_access_log.jsonl` (2026-11-06 02:39:03 UTC, seed 0) is a
  launch stopped within seconds, before Stage 1 finished. No test score was
  computed. It was restarted so that the run would record the uncommitted code.
- The completed run adds one line per seed (lines 111-113).

**MLflow.**
- Logging was added while this run was in progress, so the run itself did not
  log.
- It was logged afterwards from its output directory with
  `scripts/log_two_stage_experiment.py`, which reads the same results file and
  attaches the console log. The params, metrics and artifacts are the ones a
  self-logged run would record. Only the MLflow start time is later than the
  experiment.
- The run id is in `mlflow_run.json` and in Section 9.
- The run is tagged `run_kind=experiment` and holds no model. The
  latest-model lookup (`latest_run_id_for_horizon`) now considers only runs that
  trained a model.

**Overwrite guard.** The output directory refuses to be overwritten without
`--overwrite`. A rerun should use a new `--two-stage-out`.

## 9. Results

Test split, per drive, means over 3 seeds. Thresholds were chosen on validation.
The full tables, every level and the cost columns are in `tables.md` and the
MLflow run.

### 9.1 Candidate pools

| Net | Train candidate rows | Failing share | Test candidate rows | Ceiling (test recall at T1) | Test drives passing Stage 1 |
|---|---|---|---|---|---|
| 50% | 9,427 | 57.2% | 32,339 | 55.0% | 0.7% |
| 70% | 30,303 | 33.1% | 124,941 | 71.4% | 1.9% |
| 80% | 100,923 | 14.5% | 592,342 | 80.5% | 5.2% |
| 90% | 565,842 | 3.4% | 3,272,886 | 90.9% | 34.8% |

The validation measurement in Section 3 holds on test. The failing share falls
fast, and the 90% net passes a third of the test fleet.

### 9.2 Cascade against Stage 1 alone

The cell shows test precision, then (in brackets) the mean paired gap in
precision points and the number of seeds whose 95% interval lies above 0.

| Recall level | Stage 1 alone | Net 50% | Net 70% | Net 80% | Net 90% |
|---|---|---|---|---|---|
| 10% | 59.4% | 61.7% (+2.3, 1/3) | 64.5% (+5.1, 3/3) | 64.3% (+4.8, 2/3) | 64.0% (+4.6, 2/3) |
| 20% | 55.5% | 56.0% (+0.5, 0/3) | 58.0% (+2.4, 1/3) | 58.2% (+2.7, 1/3) | 57.5% (+2.0, 1/3) |
| 35% | 46.2% | 47.9% (+1.7, 1/3) | 50.9% (+4.7, 3/3) | 49.7% (+3.5, 3/3) | 48.0% (+1.8, 2/3) |
| 50% | 33.4% | (= Stage 1) | 35.4% (+2.0, 3/3) | 35.1% (+1.7, 3/3) | 34.2% (+0.9, 1/3) |
| 60% | 22.8%, R 64.5% | - | 23.8%, R 64.2% (+1.0, 2/3) | 24.8%, R 64.3% (+1.9, 3/3) | 24.7%, R 63.8% (+1.9, 3/3) |
| 70% | 15.8% | - | (= Stage 1) | 14.9% (-0.8, 0/3) | 14.7% (-1.1, 0/3) |
| 80% | 6.4% | - | - | (= Stage 1) | 5.8% (-0.7, 0/3) |

Per seed at 60% recall (caught failing drives and false alarms, Stage 1 alone
then cascade). There are 1,479 failing test drives.

| Net | Seed 0 | Seed 1 | Seed 2 | Recall gap, 95% interval (every seed) |
|---|---|---|---|---|
| 70% | 956 / 3,388 -> 952 / 3,222 | 949 / 3,236 -> 945 / 2,877 | 955 / 3,060 -> 953 / 3,016 | includes 0 |
| 80% | 956 / 3,388 -> 948 / 2,760 | 949 / 3,236 -> 946 / 2,983 | 955 / 3,060 -> 960 / 2,940 | includes 0 |
| 90% | 956 / 3,388 -> 940 / 2,980 | 949 / 3,236 -> 943 / 2,912 | 955 / 3,060 -> 947 / 2,728 | includes 0 (seed 0: -2.2 to +0.1 points) |

What the results show:

1. **A wider net extends the cascade's reach to 60% recall.** At 60% recall
   every wider net raised precision in every seed, by about 1-2 points: 44 to
   630 fewer false alarms per test window, out of about 3,200.
2. **The gain rises at the strict levels too.** At 10% recall the 70% net gains
   +5.1 points with all three intervals above 0, against +2.3 (1/3) for today's
   50% net. A Stage 2 trained on a broader pool of look-alikes separates the top
   of the ranking better.
3. **Past about 70% recall the cascade hurts.** At the 80% and 90% nets,
   precision falls 0.7-1.1 points at the 70-80% levels, because Stage 2 is
   re-ranking a pool that is mostly healthy.
4. **Recall at 60% dips slightly.** In 8 of 9 net-and-seed pairs the cascade
   caught 2-16 fewer failing drives than Stage 1 at the same validation-chosen
   recall level. Every recall interval includes 0, so the dip is not
   established, but it is consistent.
5. **The goal is still unmet.** No operating point approaches 90% precision.
   The best figure at 60% recall is 24.8%.

### 9.3 Decision (Section 6 rule)

| Criterion | Net 70% | Net 80% | Net 90% |
|---|---|---|---|
| 1. Precision higher at 60% in every seed | yes (+0.2 to +2.0) | yes (+0.8 to +3.6) | yes (+1.8 to +2.0) |
| 2. Interval above 0 in at least 2 of 3 seeds | yes (2/3) | yes (3/3) | yes (3/3) |
| 3. Recall at 60% not lower | **no** (64.2% vs 64.5%) | **no** (64.3% vs 64.5%) | **no** (63.8% vs 64.5%) |
| 4. Strict levels not below the 50% net | yes | yes | yes |

**No net passes, so `candidate_recall` stays 0.5.** Every net fails on
criterion 3 alone: a recall dip of 3-10 drives out of 1,479 that is not
statistically established. The rule was fixed before the results and is applied
as written. Relaxing it now, for example to "recall not significantly lower",
would be a post-hoc change. It would make the **80% net** the candidate:
- +1.9 precision points at 60% recall, interval above 0 in 3/3 seeds;
- about 330 fewer false alarms;
- +4.8 points at 10% recall;
- a mean of 2 fewer failing drives caught.

Adopting it would therefore need a **confirmatory run under the relaxed rule,
fixed in advance**, on data this experiment has not used, for example the
sealed Q3 data (runbook 1.4).

### 9.4 Provenance of these numbers

| Item | Value |
|---|---|
| MLflow run | `c05a109a5af04e749e3a9ea715ad2823` (experiment `disk-failure-prediction`, tag `run_kind=experiment`), logged after the run |
| Code | `f4772f3` + `code.patch` (the commit that records it is in the git history of `docs/cascade_wide_net.md`) |
| Data | kept work dir `train_model_frame_bjdxfhj1` from training run `839d92bc` |
| Test accesses | `test_access_log.jsonl` lines 111-113 (line 110: aborted launch, no scores) |
| Stage 1 (seed 0) | 184 trees; Stage 2 at the 50% net: 40 trees |
| Wall time | about 8 minutes per seed for all four nets (each net 30-90 s after the shared Stage 1) |

## 10. Threats to validity

- **One test period.** The seeds vary the training randomness, not the test
  drives (1,479 failing). A gain could still depend on this period.
- **Refitted Stage 1.** It matches the pipeline's settings but is not the frozen
  model, and seed-to-seed spread at a fixed threshold is a few precision points
  (runbook 4.5).
- **Validation optimism.** Stage 2 early-stops on validation candidates, and
  thresholds are chosen on validation. Validation precision is therefore
  optimistic by design, and only test numbers are compared.
- **Fixed Stage 2 regularisation.** It stays at `min_child_samples: 50` for every
  net, so the comparison changes one thing. A wide net gives Stage 2 a much
  larger pool, and a negative result does not rule out a re-regularised Stage 2.
- **Assumed cost ratio.** The 20:1 ratio is illustrative. The counts of caught
  drives and false alarms are reported, so any other ratio can be applied.

## 11. Addendum (2026-11-06): the 0.8 net adopted post hoc and run on Q3 2026

The 0.8 net was adopted on 2026-11-06 by decision, without the confirmatory run
section 9.3 asked for: a post-hoc choice, recorded as such. It was then run on
the Q1+Q2 -> Q3 window at three horizons with LightGBM, Cox and AFT Stage 2
models (`docs/q3_horizon_and_cascade_run.md` section 7.3; runbook 1.1.4). On Q3
the strict-end gain is weaker than here (+5.1 points at 10% recall for 30 days,
1 of 3 seeds above 0; none at 44 or 60 days), the 60% gap is -0.2, and the 0.5
net loses 2.5 points at 35% recall. The 0.8 net stays, not confirmed; these
numbers are descriptive.

On Q3 the 0.8 net's Stage 1 kept 79-82% of failing test drives while passing
5-6% of all drives, as on validation, and 98.6% of the failing drives it does
not reach show no SMART counter rise before failure (runbook 1.1.5), so the net
is about as wide as the telemetry allows. A stack of the net's LightGBM and Cox
Stage 2 then caught a few more failing drives in the top 500 than the LightGBM
Stage 2 alone, not confirmed and not adopted (runbook 1.1.4a;
`docs/q3_horizon_and_cascade_run.md` section 7.7).
