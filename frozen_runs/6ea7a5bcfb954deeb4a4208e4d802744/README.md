# Frozen run 6ea7a5bcfb954deeb4a4208e4d802744 (`chrono_h30`)

The files MLflow holds for the 30-day run of the **chronological split with gaps**
(`make chrono-train`, which is `make train TRAIN_TAG=chrono_h30 TRAIN_ARGS="--horizon-days 30 --keep-work-dir --save-scores"`,
2026-11-10), copied here so the run can be inspected and checked without the machine that trained it.
Frozen 2026-11-10 after its validation and test results were recorded. Why it was made, its results and
the rule not to retrain it: `docs/model_status_and_runbook.md` Section 1.1.9.

Window (`configs/model.yaml` `splits`; label table `data/audit/dataset_versions/20261110T082656102066Z.json`):
train rows 2026-01-01..04-20, validation 05-21..06-16, test 07-17..08-31, with 30-day gaps, no drive
holdout, feature statistics from rows before 04-21.

| Path | What |
|---|---|
| `frozen_spec.json` | Drive threshold, the 246 feature columns, the action-tier cut-offs, the calibrator, the Stage 1 flag rate |
| `action_tiers.json` | warn / cordon / migrate / drain cut-offs |
| `two_stage.json` | Second-stage settings (0.8 net), candidate counts and the stage-1 precision-at-recall table |
| `model_card.json`, `model_card.md` | The model card written by `make train` |
| `figures/` | Drive precision-recall curves and confusion matrices for train (in-sample), validation and test |
| `stage1/` | Stage-1 LightGBM classifier (339 trees, 246 features), MLflow `lightgbm` flavor, skops format |
| `stage2/` | Stage-2 LightGBM classifier (103 trees, 247 features: stage 1's score is the extra input) |
| `SHA256SUMS` | Checksums of every other file here; `sha256sum -c SHA256SUMS` |

Not included: the score table `lightgbm_scores_h30_chrono_h30.parquet` (validation and test scores,
197,723,045 bytes, SHA-256 `a78bb20804c006c02d05fed66ffaab27271ece2c2c2107c8b4d12e2217e66b2c`), and each
model's `conda.yaml` (`MLmodel` still names it; the pinned `requirements.txt` and `python_env.yaml` are
here). The stack of Stage 2 variants (LightGBM, Cox, stack) is **not** part of this folder: its results are
in `data/audit/stack_chrono_h30/` and it is not saved as a scorer.

Trained with mlflow 3.16.1, lightgbm 4.7.0, skops 0.16.0, **Python 3.12.16** (the earlier frozen runs used
3.14.8; the repository pins 3.12 in `.python-version` since 2026-11-10).

## What it is

It answers one question per drive: will this drive fail within 30 days? Per drive, a failing drive is
scored by its highest score inside that window and a healthy drive by its highest score over the whole test
period, so precision and recall count drives, not rows.

| Part | What it is |
|---|---|
| Training rows | 4,998,343 (every positive row plus a seeded sample of negatives, cap 5,000,000) from rows 2026-01-01..04-20 |
| Validation | Rows 05-21..06-16: 356,811 drives, 684 failing. Used for early stopping, every threshold and the Stage 1 net |
| Stage 1 | LightGBM, 31 leaves, learning rate 0.05, at least 200 rows per leaf, L2 10: **339 trees**, 246 features |
| Candidate threshold | **0.1431**: the validation cutoff that keeps 80% of failing validation drives; 7.2% of validation and 9.0% of test drives pass |
| Stage 2 | LightGBM on the candidate training rows only (201,052 rows, 29,667 positive, 14.8%), with Stage 1's out-of-fold score as a 247th feature: **103 trees** |
| Combined score | Stage 1's score below the candidate threshold; `T1 + (1 - T1) * stage2` above it |
| Thresholds (on the combined score, all from validation) | Primary **0.9431** (the goal's 68% at >= 10% recall was unreachable, so the highest-precision point with recall >= 10%; `target_met: false`); warn 0.5298; cordon 0.7336; migrate 0.8572; drain 0.9347 |

**Test, 2026-07-17 to 08-31** (864 failing and 367,260 healthy drives, 368,124 in all):

| Operating point | Precision | Recall | Caught | False alarms |
|---|---|---|---|---|
| Stage 1 net (0.1431; computed from the saved scores) | 2.2% | 85.0% | 734 | 32,579 |
| Warn tier | 10.8% | 65.2% | 563 | 4,661 |
| Cordon tier | 18.1% | 56.1% | 485 | 2,192 |
| Migrate tier | 28.0% | 44.4% | 384 | 987 |
| Drain tier | 52.4% | 20.4% | 176 | 160 |
| **Primary threshold** | **59.9%** [52.5, 66.3] | **13.3%** [11.1, 15.6] | 115 | 77 |

Read it with these in mind (runbook 1.1.9):
- **The 68% goal is not met**: 51.5% at 10.2% recall on validation, 59.9% at 13.3% on test.
- Stage 2's gain over Stage 1 alone at the primary point is +5.1 precision points [+0.9, +9.4].
- Test AUROC is 0.923 (validation 0.906). Validation and test base rates are 0.19% and 0.23%, so
  validation no longer overstates test.
- Unlike the drive-holdout runs, all four tiers have distinct cut-offs and flag drives.
- Q3 is not a fresh test set: 143 logged test looks by the end of the run.

## Which run the code loads

`score-fleet` and the agent take the **latest training run for the horizon** by MLflow start time
(`src/models/trained_thresholds.py::latest_run_id_for_horizon`). Since this run was trained, that is
`6ea7a5bc`, not `c459046e`: freezing does not change it, and neither did anyone choose it. `evaluate-frozen`
and the other readers still look a run up by id in the local MLflow store, and nothing reads this folder.

Check and load, with the library versions above:

```bash
cd frozen_runs/6ea7a5bcfb954deeb4a4208e4d802744 && sha256sum -c SHA256SUMS && cd ../..
```

```python
import mlflow
stage1 = mlflow.lightgbm.load_model("frozen_runs/6ea7a5bcfb954deeb4a4208e4d802744/stage1")
stage2 = mlflow.lightgbm.load_model("frozen_runs/6ea7a5bcfb954deeb4a4208e4d802744/stage2")
```
