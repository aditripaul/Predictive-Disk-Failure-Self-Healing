# Q1+Q2 2026 to Q3 2026: horizon comparison and survival Stage 2

Status: **runs completed 2026-11-06.** Sections 1-6 were written before any
run read Q3 2026 (commit `edeb1b3`); sections 7-9 were filled afterwards from
the saved outputs, and every number names the file and the MLflow run it came
from. Outcome: **30 days stays the primary horizon, the LightGBM Stage 2 stays,
and the 0.8 net is kept but not confirmed** (section 7.4).

This document is written to be cited. It records what was decided before the
test data was read, and says which choices were made after earlier results and
are therefore descriptive here, not confirmed.

## 1. Questions

1. **Prospective hold-up.** Does a LightGBM two-stage model trained on Q1+Q2
   2026 keep its drive-level precision and recall on Q3 2026, a quarter it never
   saw in any form (features, thresholds, early stopping, feature statistics)?
2. **Horizon.** Does a longer label (44 or 60 days) find failing drives better
   than the 30-day label, **once the base-rate effect is removed**? A longer
   label marks more drive-days positive, so precision and AUPRC rise even when
   the model ranks drives no better (runbook 2.4). The comparison is therefore
   made in ROC terms: drive AUROC and recall at a matched false-alarm rate, on
   the same drives, under each label definition.
3. **Stage 2 kind.** On the candidates of a 0.8 Stage 1 net, does a survival
   Stage 2 (XGBoost Cox or AFT) rank suspects better than the LightGBM
   classifier Stage 2? Alone, Cox (62.9% test precision) and AFT (68.5%) beat
   the LightGBM two-stage (57.1%) at similar recall on the 2026-04/05 test
   window (runbook 1.1.2), but neither has been tried as a Stage 2.

## 2. Data and window

Backblaze drive stats, gold layer rebuilt 2026-11-06 with Q3 2026 ingested
(data from 2025-10-01 to 2026-09-30). Fleet-wide feature statistics (model-family
z-score mean and standard deviation, each model's expected SMART attributes)
are computed only on rows before 2026-07-01
(`pipelines/build_gold_features.py::sealed_start`), so no Q3 value enters a
feature of any row.

| Split | Rows | Why |
|---|---|---|
| train | 80% of drives, 2026-01-01 .. 2026-05-01 | Every 60-day label is observed by 06-30, so no training label depends on Q3 |
| validation | the other 20% of drives, same dates | Thresholds, early stopping, Stage 2 candidates |
| unused | 2026-05-02 .. 2026-06-30 | Their 60-day labels depend on Q3 outcomes |
| test, 30-day | all drives, 2026-07-01 .. 2026-08-31 | Last row whose 30-day label is observed (data ends 09-30) |
| test, 44-day | all drives, 2026-07-01 .. 2026-08-17 | 09-30 minus 44 |
| test, 60-day | all drives, 2026-07-01 .. 2026-08-01 | 09-30 minus 60 |

- **Same training rows for every horizon.** Train and validation stop at
  2026-05-01 for all three horizons, so a difference between them comes from the
  label, not from more or fresher training data. The cost is two months of
  training data the 30-day model could have used.
- **Validation is a drive holdout over the training dates**
  (`splits.holdout_fraction: 0.2`), not a later period. A later period would
  either peek into Q3 (June's labels) or shorten training to Q1. 20% rather
  than the earlier 10% gives about twice as many failing validation drives.
  The holdout is drawn only from drives first seen before 2026-07-01.
- **Training sample.** Every positive row plus a hash-seeded sample of
  negatives up to 5,000,000 rows (`model.max_train_rows`), as in every earlier
  run.
- **Q4 2025** is used only as feature history: the 7-30 day rolling windows of
  early January read it.
- **Test windows differ by horizon.** Each model's headline uses all the Q3 its
  labels allow. The paired horizon comparison uses the common window,
  2026-07-01 .. 2026-08-01.

Configuration: `configs/model.yaml` `splits` and `horizons_days: [30, 44, 60]`;
split logic in `pipelines/train_model.py::_split_period_predicate`,
`resolved_test_end` and `test_source_split`.

## 3. Protocol, and what is post hoc

- **Q3 was the sealed quarter** for frozen run `839d92bc`. Its single-shot
  evaluation was **skipped by decision on 2026-11-06**, and Q3 is used here as
  an ordinary test set (`splits.test_split: sealed`). `make evaluate-frozen`
  now refuses a sealed evaluation while that is so. The frozen run is never
  scored prospectively; this is recorded in runbook 1.4.
- **Every look at Q3 is logged** in `data/audit/test_access_log.jsonl`. Planned:
  3 training runs, 1 horizon comparison, 9 cascade runs (3 seeds x 3 horizons).
  Any extra look is reported in section 8.
- **Post-hoc choices.** Three choices follow from earlier results on the
  2026-04/05 test window rather than from a pre-registered rule:
  - the 0.8 Stage 1 net (`candidate_recall: 0.8`). Under the rule of
    `docs/cascade_wide_net.md` section 6 it failed only criterion 3, a recall
    dip of 3-10 of 1,479 failing drives that was not statistically established;
    it gained +1.9 precision points at 60% recall (interval above 0 in 3/3
    seeds) and +4.8 at 10%;
  - the survival Stage 2 kinds, prompted by the merged-model results;
  - the 44- and 60-day horizons, prompted by "false alarms" that failed 31-60
    days later (runbook 4.3, 4.11).

  Q3 results for these are **descriptive**. Section 6 fixes the rules later data
  must confirm them by.
- **Thresholds are never re-chosen on test.** The primary threshold, the action
  tiers, the Stage 1 candidate threshold and Stage 2's early stopping all come
  from the validation holdout.

## 4. The cascade

Code: `src/models/two_stage.py` (method in `docs/cascade_wide_net.md` section 2).

- **Stage 1**: LightGBM on all 246 features, the repository's settings
  (`configs/model.yaml` `model.params`, `positive_weight_power: 0.5`).
- **Candidate threshold**: the most precise validation threshold that still
  catches 80% of failing validation drives. On the 2026-04/05 test window this
  net passed 5.2% of test drives and its pool was 14.5% failing rows.
- **Stage 2**, trained only on candidates, with the out-of-fold Stage 1 score as
  an extra feature:
  - `lightgbm`: the classifier Stage 2 the pipeline ships;
  - `cox`: XGBoost `survival:cox` on censored time to failure;
  - `aft`: XGBoost `survival:aft`.

  The survival kinds use `src/models/survival.py::survival_bounds`: an outcome is
  seen only up to the training cutoff plus the horizon, the same information the
  classifier's label uses.
- **Combined score**: Stage 1's score below the candidate threshold, and
  `T1 + (1 - T1) * s2` above it, so the cascade can only re-order candidates and
  its recall can never exceed Stage 1's at `T1` (the ceiling).
- `make train` ships the LightGBM Stage 2. The survival kinds are compared by
  `make experiment-model ... --stage2-kinds lightgbm,cox,aft`.

**A proposal this was checked against.** A "two-stage XGBoost cascade" was
proposed: an XGBoost Stage 1 on the latest daily snapshot at at least 95%
recall, and an XGBoost Cox Stage 2 ranked by top K. Taken up here: a survival
Stage 2, recall at top K, the C-index, and recording the share of drives Stage 1
passes (a drift signal). Not taken up, on measured evidence:

| Proposed | Measured here |
|---|---|
| Stage 1 recall at least 95% | On validation, a 95% net flags 56% of healthy drives and leaves a pool only 2.5x richer in failures than the raw data; a 90% net passed 35% of test drives and gained less than the 0.7 and 0.8 nets (`docs/cascade_wide_net.md` sections 3 and 9) |
| Snapshot-only Stage 1 | Stage 1 already uses the temporal features (deltas, slopes, rolling statistics, days since last increase); dropping them can only need a wider net for the same recall |
| XGBoost Stage 1 | XGBoost at depth 6 and 8 did no better than LightGBM (runbook 5.6) |
| "Train Stage 2 on suspects to prevent leakage" | Training on suspects concentrates failures; leakage is prevented by giving Stage 2 out-of-fold Stage 1 scores, which the code does |
| No missed failures and no alert fatigue | The best test result so far is about 70% precision at 14% recall; about 20% of failures show no SMART signal before they fail |

## 5. Metrics

All per **drive** (`src/models/evaluation.py::drive_level_table`): a failing
drive's score is its highest score inside its label window, a healthy drive's
its highest score over the test window.

- Precision and recall at the primary threshold and each action tier, with a
  1000-draw drive bootstrap interval.
- **Recall at a matched false-alarm rate** (0.1% and 0.5% of healthy drives):
  the base-rate-free comparison of horizons.
- **Drive AUROC**.
- **Precision and recall at top K** drives flagged over the window,
  K in {100, 250, 500, 1000, 2500, 5000} (`evaluation.top_k`). K counts drives
  over the whole window, so values are compared only between windows of the
  same length.
- **C-index** (Harrell) of Stage 2's risk, per drive: each drive's latest
  test-window candidate row, time to failure from that date, censored at
  2026-09-30.
- **Warning lead time**: days from a failing drive's first alert to its
  failure.
- **Paired drive bootstrap** of every model-to-model difference
  (`src/models/uncertainty.py::paired_flag_difference_ci`), the same resampled
  drives for both models.
- Precision is reported **with its base rate** beside it.

Figures: drive and drive-day precision-recall curves with the operating points
marked, and drive confusion matrices at each operating point, for train
(in-sample, optimistic), validation and test of every run
(`src/reporting/evaluation_figures.py`; `data/audit/plots/train/<tag>/` and the
MLflow run's `figures/`).

## 6. What is not confirmed

No later quarter exists to test on, so the Q3 results are the last data these
choices were examined on, and nothing here is confirmed:

- the 0.8 Stage 1 net,
- a survival Stage 2 (Cox or AFT) in place of the LightGBM one,
- a horizon longer than 30 days.

Each was chosen or examined on Q3 or on the 2026-04/05 test window, so Q3
numbers for them are descriptive. A claim about any of them needs data none of
these runs has seen. When such data exists, the rules below are the ones to
apply, fixed here before it does:

1. **0.8 net.** The 0.8 cascade's precision at 60% recall is above Stage 1
   alone in every seed, the paired interval excludes 0 in at least 2 of 3
   seeds, and the paired recall interval at 60% includes 0 or lies above it.
2. **Survival Stage 2.** Cox or AFT has higher precision than the LightGBM
   Stage 2 at the 10% and 20% recall levels in every seed, with the paired
   interval above 0 in at least 2 of 3 seeds, and is not lower at 35%.
3. **Horizon.** A horizon longer than 30 days becomes primary only if its
   drive AUROC and its recall at a 0.1% false-alarm rate, both scored against
   the **30-day** labels, are not lower than the 30-day model's (paired
   interval includes 0 or lies above it), and its median warning lead time is
   longer by more than the paired interval. A precision gain alone does not
   qualify, since it follows from the base rate.
4. A 90-day horizon is not testable on this data: a 90-day label is observed
   only 90 days after the row.
5. `pipelines/evaluate_frozen.py` reads the sealed split without a date cap;
   it needs one (the last day of data minus the horizon) before any quarter is
   sealed again.

## 7. Results

All figures are per drive on the Q3 2026 test set, thresholds chosen on the
validation holdout. The three training runs are frozen in `frozen_runs/<run id>/`.

### 7.1 Prospective hold-up (question 1)

| | 30-day (`c459046e`) | 44-day (`a2cd06d5`) | 60-day (`804a20ce`) |
|---|---|---|---|
| Test window (Q3 2026) | 07-01 .. 08-31 (62 d) | 07-01 .. 08-17 (48 d) | 07-01 .. 08-01 (32 d) |
| Failing / healthy test drives | 1,105 / 367,583 | 1,106 / 363,685 | 1,117 / 359,895 |
| **Primary threshold: precision** [95% CI] | **59.0%** [52.3, 65.5] | **62.6%** [55.3, 69.4] | **63.3%** [55.7, 70.6] |
| **Primary threshold: recall** [95% CI] | **12.5%** [10.5, 14.3] | **9.9%** [8.2, 11.7] | **9.0%** [7.4, 10.7] |
| Caught / missed / false alarms | 138 / 967 / 96 | 109 / 997 / 65 | 100 / 1,017 / 58 |
| Validation precision / recall (same point) | 75.0% / 10.0% | 81.3% / 11.5% | 83.6% / 10.6% |
| Drive AUROC / AUPRC | 0.926 / 0.288 | 0.925 / 0.298 | 0.919 / 0.300 |
| Recall at 0.1% false-alarm rate | 30.4% | 31.0% | 29.5% |
| Warn tier: precision / recall | 17.3% / 60.9% | 20.2% / 58.0% | 23.7% / 53.4% |
| Cordon tier: precision / recall | 34.8% / 42.4% | 38.8% / 40.1% | 45.0% / 33.6% |
| Migrate = drain tier: precision / recall (drives) | 33.3% / 0.2% (6) | 70.3% / 2.4% (37) | 80.0% / 0.7% (10) |
| Stage 1 (0.8 net) recall on test / drives passed | 79.9% / 6.3% | 81.7% / 6.2% | 78.5% / 5.2% |
| Stage 2 gain at the primary point [95% CI] | +4.2 [-0.1, +8.5] | +1.1 [-5.0, +6.9] | -3.8 [-10.5, +3.0] |
| Median warning lead time | 20 days | 22.5 days | 20 days |
| Trees: Stage 1 / Stage 2 | 310 / 74 | 225 / 201 | 281 / 54 |

Source: `data/audit/data_quality_reports/model_evaluation_report_h<H>_q1q2_q3.json`
and the runs' MLflow metrics. The 30-day model's 59.0% at 12.5% recall on a
quarter it never saw sits beside the 57.1% at 12.0% of run `839d92bc` on the
2026-04/05 window (runbook 1.1.1); the two are not paired, so this is "held
up", not "improved". The goal of 90% precision at >= 10% recall is not met.
Validation overstates test precision by 15-20 points in every run (section 2's
same-period caveat, realised).

### 7.2 Horizon (question 2)

`scripts/compare_horizons.py`, 11,392,351 drive-days of 361,012 drives in
2026-07-01 .. 08-01 (the window every horizon observes), each model against
each label definition; `data/audit/horizon_q3/horizon_comparison.json`,
`tables.md` and `figures/`. Against the 30-day labels (777 failing drives):

| Model | AUROC [gap vs 30-day, 95%] | Recall at 0.1% false alarms [gap] | Recall at 0.5% false alarms [gap] | Median lead at 0.1% |
|---|---|---|---|---|
| 30-day | 0.916 | 30.6% | 51.7% | 14 days |
| 44-day | 0.918 [-0.001, +0.006] | 30.9% [-1.4, +1.9] | 52.1% [-0.9, +1.7] | 15 days |
| 60-day | 0.919 [-0.001, +0.008] | 29.6% [-2.9, +0.6] | 52.9% [-0.3, +2.8] | 15 days |

Against the 44- and 60-day labels the gaps are the same size (AUROC +0.002 to
+0.003, recall within 1.3 points, every interval including 0). The precision
that rises with the horizon in section 7.1 is the base rate:

| 30-day model scored against | Base rate | Precision at 0.1% false alarms | Recall |
|---|---|---|---|
| 30-day labels | 0.215% | 40.1% | 30.6% |
| 44-day labels | 0.264% | 44.8% | 30.1% |
| 60-day labels | 0.309% | 49.0% | 30.9% |

Figures: `docs/figures/q3_2026/horizon_labels_<H>d_*.png` (shown in runbook
1.1.4); the three precision-recall curves overlap.

**Answer:** no. Matched on the same drives and false-alarm rate, the three
models catch the same share of failing drives and warn at the same time
(14-15 days median lead). Under section 6 rule 3 no longer horizon qualifies.
Top-K precision and recall agree (within a few points at every K).

### 7.3 Stage 2 kind and the 0.8 net (question 3)

`make experiment-model EXPERIMENT_ARGS="--work-dir <kept dir> --variants
--skip-baseline --two-stage --two-stage-out data/audit/cascade_q3_h<H>"`
(nets 0.5 and 0.8, Stage 2 kinds lightgbm / cox / aft at the 0.8 net, 3 seeds,
1,000-resample paired drive bootstrap). Cascade minus Stage 1 alone, test
precision points, mean over seeds, with seeds whose interval lies above / below
0:

| Horizon | Stage 2 variant (net) | 10% recall | 35% | 60% | 70% |
|---|---|---|---|---|---|
| 30 | LightGBM (0.5, the old net) | +3.8 (1 / 0) | -2.5 (0 / 3) | not reached | not reached |
| 30 | **LightGBM (0.8)** | **+5.1** (1 / 0) | +0.6 (0 / 0) | -0.2 (1 / 2) | +0.2 (2 / 1) |
| 30 | Cox (0.8) | +4.5 (0 / 0) | -0.9 (0 / 1) | +0.2 (1 / 1) | +0.5 (2 / 0) |
| 30 | AFT (0.8) | -6.5 (0 / 2) | -6.6 (0 / 3) | -2.8 (0 / 3) | -1.7 (0 / 3) |
| 44 | LightGBM (0.8) | -0.5 (0 / 0) | +0.6 (1 / 0) | -0.1 (1 / 1) | -0.8 (0 / 2) |
| 44 | Cox (0.8) | +4.2 (1 / 0) | -0.1 (0 / 0) | -0.3 (1 / 1) | -0.1 (0 / 1) |
| 44 | AFT (0.8) | -2.3 (0 / 0) | -7.6 (0 / 3) | -3.2 (0 / 3) | -1.7 (0 / 3) |
| 60 | LightGBM (0.8) | -1.3 (0 / 1) | +1.8 (2 / 0) | +1.3 (3 / 0) | -0.8 (0 / 2) |
| 60 | Cox (0.8) | +0.9 (0 / 0) | +0.6 (1 / 1) | -0.1 (0 / 1) | -0.3 (0 / 2) |
| 60 | AFT (0.8) | -5.8 (0 / 1) | -6.7 (0 / 3) | -3.3 (0 / 3) | -1.8 (0 / 3) |

Means over seeds per horizon: Stage 1 at the 0.8 cutoff keeps 79.7% / 80.4% /
79.2% of failing test drives and passes 6.3% / 5.9% / 5.6% of drives; the
C-index on the test candidates (per drive) is 0.901 / 0.880 / 0.864 for Stage
1 and within 0.01 of it for every Stage 2 kind.

- **Above 20% recall, no Stage 2 variant beats Stage 1 alone by more than
  about 1 point** in any horizon; the curves coincide past 30% recall
  (`data/audit/cascade_q3_h<H>/figures/`). The cascade's value is at the
  strict end only.
- **0.8 net:** +5.1 points at 10% recall for 30 days (1 of 3 seeds above 0),
  nothing for 44 and 60 days; the 0.5 net loses 2.5 points at 35% recall in 3
  of 3 seeds. Not confirmed under section 6 rule 1 (the 60% gap is -0.2); not
  worse than 0.5.
- **Cox** matches the LightGBM Stage 2 within about 1 point with no consistent
  sign; rule 2 is not met.
- **AFT** is worse at every level in every horizon (2-8 points, 3 of 3
  seeds). Its validation thresholds land at the wrong recall on test (the
  "10%" row catches 16-21%), so it is not a candidate.

The runbook's generated cascade tables for all three horizons, and the seed-0
figures (`docs/figures/q3_2026/cascade_h<H>_*.png`), are in section 1.1.4 of
`docs/model_status_and_runbook.md`.

### 7.4 Decisions

| Choice | Outcome on Q3 | Decision |
|---|---|---|
| Horizon | No longer horizon ranks better, catches more at matched false-alarm rate, or warns earlier | **30 days stays primary** |
| Stage 2 kind | Cox = LightGBM; AFT worse | **LightGBM Stage 2 stays** |
| Stage 1 net | 0.8 not worse than 0.5; strict-end gain weaker than on 2026-04/05 | **0.8 kept, not confirmed** |
| Model in use | 30-day run `c459046e` is the latest run for the primary horizon | `score-fleet` and the agent use it |

### 7.5 What the cascade gives, end to end (30-day, Q3)

| Operating point | Precision | Recall | Caught / missed / false alarms |
|---|---|---|---|
| Stage 1 cutoff (filter) | 3.8% | 79.9% | 883 / 222 / 22,463 |
| Warn tier | 17.3% | 60.9% | 673 / 432 / 3,208 |
| Cordon tier | 34.8% | 42.4% | 469 / 636 / 877 |
| Primary threshold | 59.0% | 12.5% | 138 / 967 / 96 |

Above 60% recall false alarms outnumber catches about 4 to 1 (25 to 1 at
80%), while under 1% of healthy drives are flagged: with 0.3% of drives
failing, a small false-alarm rate is many false alarms. The limit is the
information in SMART (about 20% of failing drives show no change before they
fail), not the Stage 2 learner: three learners rank the same suspects the same
way.

## 8. Provenance

| Item | Value |
|---|---|
| Method commit (before any run read Q3) | `edeb1b3` |
| Code for the runs | `edeb1b3` (training), `6c61e35` (30-day cascade and horizon comparison), `3bfe609` (44- and 60-day cascade); each cascade directory's `code.patch` holds the only uncommitted change, a `.gitignore` edit |
| Training runs (MLflow, experiment `disk-failure-prediction`) | `c459046e4f1645169151e7be25ff064d` (h30_q1q2_q3), `a2cd06d53e624c09b0d12f81076def8a` (h44_q1q2_q3), `804a20cec10048abb1bc3ccf510f96e1` (h60_q1q2_q3) |
| Frozen bundles | `frozen_runs/<run id>/` (spec, tiers, two-stage settings, model card, figures, both stages, SHA256SUMS) |
| Cascade experiments (MLflow, `run_kind=experiment`) | `f760b06de7e3405ab1ee6723109090ef` (30), `a4f74561227b46e483863bb1d7baf892` (44), `cabc3023e9944a31b6633f59ec3f39bc` (60) |
| Figures (committed copies) | `docs/figures/q3_2026/` (README lists the source of each) |
| Saved outputs (local, not committed) | `data/audit/cascade_q3_h<H>/`, `data/audit/horizon_q3/`, `data/audit/plots/train/h<H>_q1q2_q3/`, score tables `data/tmp/lightgbm_scores_h<H>_h<H>_q1q2_q3.parquet` (SHA-256 in each bundle's README) |
| Label table | `data/audit/dataset_versions/20261106T091944670108Z.json` |
| Test accesses | `data/audit/test_access_log.jsonl` lines 114-116 (training), 117 (horizon comparison), 118-126 (cascade, 3 seeds x 3 horizons) |
| Wall times | labels 20 min; each training run about 25-30 min; horizon comparison 3 min; each cascade experiment about 25 min (Stage 1 and out-of-fold scores about 4 min per seed, each net 1-2 min) |

## 9. Threats to validity

- **One test quarter.** Q3 2026 only; a gain or loss of a point or two could
  belong to this period. The seeds vary the training randomness, not the test
  drives.
- **Same-period validation.** The holdout shares the training dates and
  overstated test precision by 15-20 points; every threshold, the Stage 1 net
  and Stage 2's early stopping come from it.
- **Post-hoc choices.** The 0.8 net, the survival kinds and the longer horizons
  were chosen on earlier results; their Q3 numbers are descriptive (section 3),
  and the fixed rules of section 6 remain untested on untouched data.
- **Shorter test windows for longer horizons** (32 days at 60 days against 62
  at 30), so their single-model numbers are noisier; the paired comparison
  uses the common window only.
- **"Caught" depends on the label definition**, which is why every model was
  scored against every horizon's labels.
- **Survival thresholds.** Cox and AFT thresholds chosen on validation land at
  a different recall on test, so their rows compare at their realised recall,
  not the nominal level.
- **Figures are seed 0** of each experiment; the tables summarise all seeds.
