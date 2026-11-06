# Q1+Q2 2026 to Q3 2026: horizon comparison and survival Stage 2

Status: **method fixed, runs not yet started** (sections 1-6 written
2026-11-06, before any run read Q3 2026). Sections 7-9 are filled from the
saved outputs after the runs, never typed by hand: every number names the file
and the MLflow run it came from.

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

  Q3 results for these are **descriptive**. Section 6 fixes the rules a later
  sealed quarter must confirm them by.
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

## 6. Pre-registration for Q4 2026

Q4 2026 (2026-10-01 to 12-31) is the next sealed quarter. Before it is
ingested, the model to be frozen for it is chosen from the Q3 results below and
named in runbook 1.4, and it is then scored on Q4 once. These rules are fixed
now, before the Q3 results exist.

1. **0.8 net.** Confirmed if, on Q4 and over 3 seeds, the 0.8 cascade's
   precision at 60% recall is above Stage 1 alone in every seed, the paired
   interval excludes 0 in at least 2 of 3 seeds, and the paired recall interval
   at 60% includes 0 or lies above it.
2. **Survival Stage 2.** Cox or AFT replaces the LightGBM Stage 2 only if it
   has higher precision than the LightGBM Stage 2 at the 10% and 20% recall
   levels in every seed, with the paired interval above 0 in at least 2 of 3
   seeds, and is not lower at 35%.
3. **Horizon.** A horizon longer than 30 days becomes primary only if its drive
   AUROC and its recall at a 0.1% false-alarm rate, both scored against the
   **30-day** labels, are not lower than the 30-day model's (paired interval
   includes 0 or lies above it), and its median warning lead time is longer.
   A precision gain alone does not qualify, since it follows from the base
   rate.
4. **90-day horizon.** Not testable on Q3 or on Q4 alone: a 90-day label is
   observed only 90 days after the row, so a 90-day test window on Q4 rows needs
   Q1 2027 data. It is added as an exploratory arm, reported in ROC terms, once
   that data exists.
5. Before Q4 is scored, `pipelines/evaluate_frozen.py` gets the date cap it
   lacks (the last day of data minus the horizon).

## 7. Results

To be filled from the saved outputs after the runs.

## 8. Provenance

To be filled: MLflow run ids, commits, `code.patch`, test-access log lines,
wall times.

## 9. Threats to validity

To be filled after the runs. Known in advance:
- one test quarter, and Q3 includes summer;
- validation shares the training dates (earlier same-period holdouts carried
  over to test worse than a later validation period);
- the post-hoc choices of section 3;
- the 60-day test window is half as long as the 30-day one;
- which failures count as "caught" depends on the label definition, which is
  why horizons are also scored against each other's labels.
