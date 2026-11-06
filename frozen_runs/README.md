# `frozen_runs/`

The files of the training runs that results are quoted from, copied out of the
local MLflow store (`mlflow/`, untracked) so a run can be inspected and checked
from a clone, without the machine that trained it. One folder per run, named by
the MLflow run id; each has its own `README.md`.

**Frozen means the run is not retrained or overwritten.** A model that was
retrained and then scored on the same data would quietly become another
validation set. A new experiment gets a new run tag and a new folder.

## The runs

| Run id | Tag | What | Window and horizon | Test result (per drive) | Size |
|---|---|---|---|---|---|
| `c459046e4f1645169151e7be25ff064d` | `h30_q1q2_q3` | LightGBM two-stage with a 0.8 Stage 1 net (details below). **The current model**: the latest run for the primary horizon, so `score-fleet` and the agent use it | Train / validation holdout (20% of drives) 2026-01-01..05-01, test Q3 2026 (07-01..08-31), 30-day | 59.0% precision at 12.5% recall | 2.6 MB |
| `a2cd06d53e624c09b0d12f81076def8a` | `h44_q1q2_q3` | Same, 44-day label | Same rows, test 07-01..08-17 | 62.6% at 9.9% | 2.7 MB |
| `804a20cec10048abb1bc3ccf510f96e1` | `h60_q1q2_q3` | Same, 60-day label | Same rows, test 07-01..08-01 | 63.3% at 9.0% | 2.4 MB |
| `839d92bcd305494a8930d0dfbbb3f2a6` | `h30_seed0` | LightGBM two-stage, 0.5 net | Train to 2026-01-15, validation 02-15..03-31, test 04-01..05-31 (61 days), 30-day | 57.1% at 12.0% | 1.3 MB |
| `fb58d9576c6146ba94c13389005fa15b` | `h30_seed0` (merged) | The merged approach: XGBoost AFT and Cox, LSTM, TCN and rank blends, beside the LightGBM two-stage member above, on identical rows. No LightGBM model of its own | Same window as `839d92bc` | Best blend (LightGBM + AFT + TCN) 70.1% at 13.7% | 426 MB (the score table) |

The three Q3 runs are the 30-, 44- and 60-day models of one experiment, trained
on identical rows so the horizon is the only difference. Compared on the same
drives and matched on false-alarm rate they are indistinguishable (30 days
stays primary); results, the cascade and survival Stage 2 comparison and the
figures are in `docs/model_status_and_runbook.md` Section 1.1.4 and
`docs/q3_horizon_and_cascade_run.md`. The April-May runs are Sections 1.1.1
(`839d92bc`) and 1.1.2 (`fb58d957`). The two windows differ, so their
precision figures are not comparable with each other (runbook 1.7).

No run here has been scored on an untouched period: Q3 2026 was the test set of
the first three, and the blend of the last has not been read on Q3 at all.

## The current model: `c459046e` (`h30_q1q2_q3`)

It answers one question per drive: will this drive fail within 30 days? A
drive-day row is positive if the drive fails within 30 days of it. Per drive, a
failing drive is scored by its highest score inside that window and a healthy
drive by its highest score over the whole test period, so precision and recall
count drives, not rows.

| Part | What it is |
|---|---|
| Features | The 246 columns listed in `frozen_spec.json` (rolling statistics, deltas, slopes, acceleration, defect velocity, days since last increase, lifetime and ratio features, per-model-family z-scores, feature confidence) |
| Training rows | 80% of drives, 2026-01-01 to 2026-05-01. Every positive row plus a seeded sample of negatives, capped at 4,997,704 rows: 37,063 positive rows from 1,754 failing drives, among 283,181 drives |
| Validation | The other 20% of drives over the same dates: 70,977 drives, 418 failing. Used for early stopping, every threshold and the Stage 1 net |
| Stage 1 | LightGBM, 31 leaves, learning rate 0.05, at least 200 rows per leaf, L2 10, positive rows weighted by (healthy / failing)^0.5, early stopping on validation average precision: **310 trees** |
| Candidate threshold | **0.235**: the most precise validation cutoff that still catches 80% of failing validation drives. Rows at or above it are Stage 2's candidates; on Q3 about 6.3% of drives pass |
| Stage 2 | LightGBM on the candidate training rows only (121,612 rows, 20.4% positive), with Stage 1's out-of-fold score (3 folds split by drive) as a 247th feature; 50 rows per leaf; early-stopped on the 176,616 validation candidate rows: **74 trees** |
| Combined score | Stage 1's score below the candidate threshold; `T1 + (1 - T1) * stage2` above it. So the cascade can re-order the candidates but never catch a failing drive Stage 1 missed |
| Thresholds (on the combined score, all from validation) | Primary **0.919** (most precise point with at least 10% recall); warn 0.653; cordon 0.836; migrate = drain 0.926 (their separate precision targets are unreachable, so they share one cutoff). A monotone isotonic calibrator is stored beside them |

**Q3 2026 test** (2026-07-01 to 08-31: 1,105 failing and 367,583 healthy drives,
368,688 in all):

| Operating point | Precision | Recall | Caught | Missed | False alarms |
|---|---|---|---|---|---|
| Stage 1 cutoff (the filter) | 3.8% | 79.9% | 883 | 222 | 22,463 |
| Warn tier | 17.3% | 60.9% | 673 | 432 | 3,208 |
| Cordon tier | 34.8% | 42.4% | 469 | 636 | 877 |
| **Primary threshold** | **59.0%** | **12.5%** | 138 | 967 | 96 |
| Migrate = drain tier | 33.3% | 0.2% | 2 | 1,103 | 4 |

Read it with these in mind (runbook 1.1.4):
- Stage 2's gain over Stage 1 alone at the primary point is +4.2 precision points
  (95% interval -0.1 to +8.5): not established. Above 20% recall it adds about a
  point or less.
- The same thresholds on the validation holdout gave 75% precision. The gap
  to test is the base rate (0.59% of validation drives fail against 0.30% on
  test), not overfitting: restated at the validation base rate, test precision
  is 73.9% (runbook 1.1.5).
- The migrate and drain tiers flag only 6 drives, so they are not usable here.
- The 90%-precision goal is not met.

## What a LightGBM run folder holds

| Path | What |
|---|---|
| `frozen_spec.json` | Everything a scoring run needs and nothing re-chosen: model type, horizon, the 246 feature columns, the drive threshold, the warn / cordon / migrate / drain thresholds, the calibrator and (Q3 runs) the Stage 1 flag rate |
| `action_tiers.json` | The four tier thresholds, as `score-fleet` and the agent read them |
| `two_stage.json` | Stage 2 settings: candidate threshold, candidate counts, trees, and the Stage 1 precision-at-recall table |
| `model_card.json`, `model_card.md` | The model card `make train` wrote (data, features, metrics, limits) |
| `stage1/`, `stage2/` | The two LightGBM classifiers in MLflow `lightgbm` flavor, skops format (`MLmodel`, `model.skops`, `requirements.txt`, `python_env.yaml`). Stage 2 takes 247 features: Stage 1's score is the extra input. `MLmodel` files use a relative `artifact_path` |
| `figures/` | Drive precision-recall curves and confusion matrices for train (in-sample, optimistic), validation and test (Q3 runs only) |
| `SHA256SUMS` | Checksums of every other file in the folder |
| `README.md` | That run's tag, window, trees, how it was made and what is not included |

The merged run's folder is different (boosters, weights, a packed score table,
a blend definition); its own README explains it and how to rebuild the table.

## Check and load a run

From the repository root, with the libraries named in the run's README
(mlflow 3.16.1, lightgbm 4.7.0 for the Q3 runs):

```bash
cd frozen_runs/<run id> && sha256sum -c SHA256SUMS && cd ../..
```

```python
import mlflow
stage1 = mlflow.lightgbm.load_model("frozen_runs/<run id>/stage1")
stage2 = mlflow.lightgbm.load_model("frozen_runs/<run id>/stage2")
```

Scoring a drive means Stage 1's score, then, if it clears the candidate
threshold in `two_stage.json`, Stage 2 on the features plus that score
(`src/models/two_stage.py::TwoStageModel`).

## Limits

- **Nothing in the repository loads from this folder.** `score-fleet`,
  `evaluate-frozen` and the agent look a run up by id in the local MLflow store
  (`sqlite:///mlflow/mlflow.db`); the artifact locations recorded there are
  absolute paths on the machine that trained it, so a run is not loadable from
  a clone by id. The folder is for inspection and verification.
- **Not included:** each run's score table (about 230 MB for a Q3 run; its
  SHA-256 is in the run's README, its `not_committed.sha256`, or the runbook) and each model's
  `conda.yaml`. A regenerated file that does not match its checksum is a
  different run.
- **Two older freezes are void** and are not here: `0ee06c01ff2a427ca76011fb1afb8ce3`
  and `c9845215896e47a397317cadd3983b6d` were trained before two
  `feature_confidence` bugs were fixed, so their feature columns are far outside
  what the rebuilt data produces; scoring them would run silently
  (runbook 1.1.1).

## Adding a run

1. Train with its own tag: `make train TRAIN_TAG=<tag> TRAIN_ARGS="... --save-scores"`.
2. Copy the run's artifacts (`frozen_spec.json`, `action_tiers.json`,
   `two_stage.json`, the model card, `figures/`) and both stages' model folders
   from `mlruns/`; rewrite each `MLmodel` `artifact_path` to `stage1` / `stage2`.
3. Write the folder's `README.md` and `SHA256SUMS`, verify with
   `sha256sum -c`, and scan the files for absolute paths before committing.
4. Add a row to the table above and a section to the runbook.
