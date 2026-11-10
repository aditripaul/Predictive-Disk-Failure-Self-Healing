# `stack_chrono_h30_seed1`: the 30-day stacked Stage 2 on the chronological split, seed 1, frozen as a scorer

The seed-1 fit of the Stage 2 variants of runbook Section 1.1.9 and their stack (LightGBM and Cox Stage 2
models blended by validation-ranked scores on the Stage 1 candidates), at the 30-day horizon, net 0.8, trained on
the chronological split with gaps (train rows 2026-01-01..04-20, validation 05-21..06-16, test 07-17..08-31).
The three seeds are `frozen_runs/stack_chrono_h30_seed0`, `_seed1` and `_seed2`; the seed changes the fold
assignment, both models' sampling and the blend, so they are three fits, not three copies.

**Not an MLflow run** (`log_two_stage` accepts a LightGBM Stage 2 only). Load it with
`src/models/stacking.py::load_stacked_two_stage(Path("frozen_runs/stack_chrono_h30_seed1"))`.
**Not the models of the frozen run `6ea7a5bcfb954deeb4a4208e4d802744`** (`make train`): this is the experiment's
seed-1 fit (401 trees, LightGBM Stage 2 354, candidate threshold 0.1467), a separate fit on the same
rows. Frozen: never retrained or overwritten (`save_stacked_two_stage` refuses an existing folder).

## How it was made

`make chrono-freeze CHRONO_WORK_DIR=data/tmp/train_model_frame_4pf36rc4 CHRONO_SEED=1`, which runs
`scripts/freeze_stack_scorer.py` on the kept work directory of the frozen run `6ea7a5bc` (`chrono_h30`),
2026-11-10 11:00-11:05 UTC (5 minutes; log `data/audit/chrono_logs/freeze_stack_seed1.log`). Code commit
`7715c56` plus uncommitted changes (the `CHRONO_SEED` parameter of the Makefile target; the freeze script is unchanged). It refits what `make experiment-model ... --two-stage --stage2-kinds lightgbm cox stack` fits for this
seed and net, with the same functions and settings, and **never opens the test matrix**: no test access was
recorded (the access log stayed at 146 lines).

**Reproduction check: passed, 14 of 14**, against `data/audit/stack_chrono_h30/experiment_two_stage.json` (seed 1,
net 0.8): candidate threshold 0.146683, Stage 1 401 trees, 193,438 train and
487,586 validation candidate rows (29,033 and 6,212 positive), LightGBM Stage 2
354 trees, Cox 355, stack weights LightGBM 0.4 / Cox 0.6, validation drive-level average
precision 0.29063 (stack), 0.28371 (LightGBM), 0.27833 (Cox). Every value equals the experiment's
(`freeze_report.json`, `reproduction`).

The stack saved by the re-run of the experiment (`data/audit/stack_chrono_h30_rerun/models/seed1_net0.80/`)
scores identically to this folder (largest difference 2.15e-15 on 300,000 validation rows), so the two fits agree.

## Files

| File | What |
|---|---|
| `stage1/` | Stage 1 LightGBM (MLflow LightGBM format) |
| `stage2_lightgbm/` | LightGBM Stage 2 (MLflow LightGBM format) |
| `stage2_cox/` | Cox Stage 2 (`cox.json` booster, `cox_meta.json` tree count) |
| `references.npz` | Each member's sorted validation candidate scores (the rank reference of the blend) |
| `stack.json` | Weights, members, candidate threshold |
| `agent_thresholds_stack_chrono_h30_seed1.json` | Tiers for the stack's score: `action_tiers` (lift), `action_tiers_cost` (cost), primary `threshold`, and their validation precision and recall; `score_column: score` |
| `agent_thresholds_lightgbm_chrono_h30_seed1.json`, `agent_thresholds_cox_chrono_h30_seed1.json` | The same for the LightGBM and Cox Stage 2 of this refit (reference arms; `score_lightgbm`, `score_cox`) |
| `cost_tiers_validation.md` | Cost-optimal operating points on validation for ratios 2-100, the three arms and the `chrono_h30` LightGBM cascade |
| `freeze_report.json` | Settings, git commit, the reproduction check, fit reports, tiers, sensitivity |
| `CHECKSUMS.sha256` | SHA-256 of every file here; `sha256sum -c CHECKSUMS.sha256` from this folder |

The validation scores of the three arms (9,541,741 rows, `split, drive_id, date, label, score, score_lightgbm,
score_cox`) are in `data/tmp/stack_scores/scores_chrono_h30_seed1.parquet` (194 MB, untracked like the other
score tables; SHA-256 `adb230411180582cb118bd9e9258cc048f2b688bdf40b855d9cdca67f074e918`).

## Validation tiers (drive level; failure rate 0.192%, 684 failing of 356,811 drives)

Precision / recall (drives flagged) at each tier; the lift tiers use `threshold.action_tier_lift`, converted to
precision at this validation rate (10.9, 18.1, 28.9 and 43.4%).

| Arm | Lift warn | Lift cordon | Lift migrate | Lift drain | Primary |
|---|---|---|---|---|---|
| Stack | 10.9% / 60.4% (3,806) | 18.1% / 49.9% (1,886) | 29.0% / 38.9% (918) | 43.5% / 23.4% (368) | 52.8% / 11.1% (144), threshold 0.9982 |
| LightGBM Stage 2 | 10.9% / 59.1% (3,723) | 18.1% / 48.7% (1,842) | 29.0% / 37.3% (880) | 43.5% / 20.2% (317) | 54.8% / 10.1% (126), threshold 0.9645 |
| Cox Stage 2 | 10.9% / 59.6% (3,760) | 18.1% / 50.0% (1,891) | 29.0% / 39.0% (922) | 43.4% / 20.8% (327) | 50.6% / 13.3% (180), threshold 0.9604 |

- **The 68% goal is not met on validation**: the primary threshold gives 52.8% at 11.1% recall for
  the stack (runbook 1.1.9). All four tiers fire for every arm.
- At ratio 20 the cost value `20 x caught - false alarms` is 5,370 (stack), 5,245 (LightGBM), 5,346 (Cox) and
  5,346 (the `chrono_h30` cascade).
- Cost tiers use the placeholder ratios of `configs/model.yaml threshold.action_cost_ratio`, an operations
  decision not yet made.
- These are validation numbers: the members stopped early on validation candidates and the blend weights were
  fitted there, so they are optimistic. The same seed's test results, from the experiment (thresholds set on
  validation, the 10% recall level; precision / recall): stack 50.6% / 14.1%, LightGBM 53.1% / 12.0%,
  Cox 50.3% / 17.6% (runbook 1.1.9).
