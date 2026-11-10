# `stack_chrono_h30_seed2`: the 30-day stacked Stage 2 on the chronological split, seed 2, frozen as a scorer

The seed-2 fit of the Stage 2 variants of runbook Section 1.1.9 and their stack (LightGBM and Cox Stage 2
models blended by validation-ranked scores on the Stage 1 candidates), at the 30-day horizon, net 0.8, trained on
the chronological split with gaps (train rows 2026-01-01..04-20, validation 05-21..06-16, test 07-17..08-31).
The three seeds are `frozen_runs/stack_chrono_h30_seed0`, `_seed1` and `_seed2`; the seed changes the fold
assignment, both models' sampling and the blend, so they are three fits, not three copies.

**Not an MLflow run** (`log_two_stage` accepts a LightGBM Stage 2 only). Load it with
`src/models/stacking.py::load_stacked_two_stage(Path("frozen_runs/stack_chrono_h30_seed2"))`.
**Not the models of the frozen run `6ea7a5bcfb954deeb4a4208e4d802744`** (`make train`): this is the experiment's
seed-2 fit (400 trees, LightGBM Stage 2 199, candidate threshold 0.1358), a separate fit on the same
rows. Frozen: never retrained or overwritten (`save_stacked_two_stage` refuses an existing folder).

## How it was made

`make chrono-freeze CHRONO_WORK_DIR=data/tmp/train_model_frame_4pf36rc4 CHRONO_SEED=2`, which runs
`scripts/freeze_stack_scorer.py` on the kept work directory of the frozen run `6ea7a5bc` (`chrono_h30`),
2026-11-10 11:05-11:10 UTC (5 minutes; log `data/audit/chrono_logs/freeze_stack_seed2.log`). Code commit
`7715c56` plus uncommitted changes (the `CHRONO_SEED` parameter of the Makefile target; the freeze script is unchanged). It refits what `make experiment-model ... --two-stage --stage2-kinds lightgbm cox stack` fits for this
seed and net, with the same functions and settings, and **never opens the test matrix**: no test access was
recorded (the access log stayed at 146 lines).

**Reproduction check: passed, 14 of 14**, against `data/audit/stack_chrono_h30/experiment_two_stage.json` (seed 2,
net 0.8): candidate threshold 0.135783, Stage 1 400 trees, 207,712 train and
509,010 validation candidate rows (29,457 and 6,210 positive), LightGBM Stage 2
199 trees, Cox 268, stack weights LightGBM 0.3 / Cox 0.7, validation drive-level average
precision 0.29032 (stack), 0.28192 (LightGBM), 0.28310 (Cox). Every value equals the experiment's
(`freeze_report.json`, `reproduction`).

The stack saved by the re-run of the experiment (`data/audit/stack_chrono_h30_rerun/models/seed2_net0.80/`)
scores identically to this folder (largest difference 3.87e-15 on 300,000 validation rows), so the two fits agree.

## Files

| File | What |
|---|---|
| `stage1/` | Stage 1 LightGBM (MLflow LightGBM format) |
| `stage2_lightgbm/` | LightGBM Stage 2 (MLflow LightGBM format) |
| `stage2_cox/` | Cox Stage 2 (`cox.json` booster, `cox_meta.json` tree count) |
| `references.npz` | Each member's sorted validation candidate scores (the rank reference of the blend) |
| `stack.json` | Weights, members, candidate threshold |
| `agent_thresholds_stack_chrono_h30_seed2.json` | Tiers for the stack's score: `action_tiers` (lift), `action_tiers_cost` (cost), primary `threshold`, and their validation precision and recall; `score_column: score` |
| `agent_thresholds_lightgbm_chrono_h30_seed2.json`, `agent_thresholds_cox_chrono_h30_seed2.json` | The same for the LightGBM and Cox Stage 2 of this refit (reference arms; `score_lightgbm`, `score_cox`) |
| `cost_tiers_validation.md` | Cost-optimal operating points on validation for ratios 2-100, the three arms and the `chrono_h30` LightGBM cascade |
| `freeze_report.json` | Settings, git commit, the reproduction check, fit reports, tiers, sensitivity |
| `CHECKSUMS.sha256` | SHA-256 of every file here; `sha256sum -c CHECKSUMS.sha256` from this folder |

The validation scores of the three arms (9,541,741 rows, `split, drive_id, date, label, score, score_lightgbm,
score_cox`) are in `data/tmp/stack_scores/scores_chrono_h30_seed2.parquet` (183 MB, untracked like the other
score tables; SHA-256 `3f991113b268496a231d07731e7b2c5eed8bfb8feed0ea7b1ff6d09117579a2a`).

## Validation tiers (drive level; failure rate 0.192%, 684 failing of 356,811 drives)

Precision / recall (drives flagged) at each tier; the lift tiers use `threshold.action_tier_lift`, converted to
precision at this validation rate (10.9, 18.1, 28.9 and 43.4%).

| Arm | Lift warn | Lift cordon | Lift migrate | Lift drain | Primary |
|---|---|---|---|---|---|
| Stack | 10.9% / 59.8% (3,769) | 18.1% / 49.9% (1,886) | 29.0% / 39.2% (925) | 43.5% / 17.5% (276) | 51.8% / 12.7% (168), threshold 0.9980 |
| LightGBM Stage 2 | 10.9% / 59.9% (3,778) | 18.1% / 50.0% (1,891) | 29.0% / 37.7% (891) | 43.4% / 18.3% (288) | 50.4% / 10.1% (137), threshold 0.9590 |
| Cox Stage 2 | 10.9% / 60.4% (3,806) | 18.1% / 50.7% (1,919) | 29.0% / 39.3% (929) | 43.5% / 19.4% (306) | 50.4% / 10.1% (137), threshold 0.9615 |

- **The 68% goal is not met on validation**: the primary threshold gives 51.8% at 12.7% recall for
  the stack (runbook 1.1.9). All four tiers fire for every arm.
- At ratio 20 the cost value `20 x caught - false alarms` is 5,405 (stack), 5,419 (LightGBM), 5,413 (Cox) and
  5,346 (the `chrono_h30` cascade).
- Cost tiers use the placeholder ratios of `configs/model.yaml threshold.action_cost_ratio`, an operations
  decision not yet made.
- These are validation numbers: the members stopped early on validation candidates and the blend weights were
  fitted there, so they are optimistic. The same seed's test results, from the experiment (thresholds set on
  validation, the 10% recall level; precision / recall): stack 57.0% / 18.4%, LightGBM 56.3% / 13.9%,
  Cox 54.4% / 15.2% (runbook 1.1.9).
