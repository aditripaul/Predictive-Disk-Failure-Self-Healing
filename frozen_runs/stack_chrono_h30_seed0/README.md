# `stack_chrono_h30_seed0`: the 30-day stacked Stage 2 on the chronological split, frozen as a scorer

The Stage 2 variants of runbook Section 1.1.9 and their stack (LightGBM and Cox Stage 2 models blended by
validation-ranked scores on the Stage 1 candidates), at the 30-day horizon, seed 0, net 0.8, trained on the
chronological split with gaps (train rows 2026-01-01..04-20, validation 05-21..06-16, test 07-17..08-31).
The experiment that measured them (`data/audit/stack_chrono_h30/`) saved no models; this folder holds the
seed-0 models, so the agent and the point-in-time scripts can use the Cox Stage 2 and the stack.

**Not an MLflow run** (`log_two_stage` accepts a LightGBM Stage 2 only). Load it with
`src/models/stacking.py::load_stacked_two_stage(Path("frozen_runs/stack_chrono_h30_seed0"))`.

**These are not the models of the frozen run `6ea7a5bcfb954deeb4a4208e4d802744`** (`make train`, 339-tree
Stage 1, 103-tree LightGBM Stage 2, candidate threshold 0.1431). They are the experiment's seed-0 fit, a
separate fit on the same rows (396 trees, 142 trees, candidate threshold 0.1340), so their thresholds and
scores differ from that run's. The stack and the Cox Stage 2 exist only here.
Frozen: never retrained or overwritten (`save_stacked_two_stage` refuses an existing folder).

## How it was made

`make chrono-freeze CHRONO_WORK_DIR=data/tmp/train_model_frame_4pf36rc4`, which runs
`scripts/freeze_stack_scorer.py` on the kept work directory of the frozen run `6ea7a5bc`
(`chrono_h30`), 2026-11-10 09:56-10:02 UTC (6 minutes; log `data/audit/chrono_logs/freeze_stack.log`). The code
commit is `7228894` plus uncommitted changes made just before (the `chrono-freeze` Makefile target and the
experiment's model-saving option); the freeze script itself is unchanged. It refits exactly what
`make experiment-model ... --two-stage --stage2-kinds lightgbm cox stack` fits for this seed and net, with
the same functions and settings, and **never opens the test matrix**: no test access was recorded (the
access log stayed at 143 lines).

**Reproduction check: passed, 14 of 14**, against `data/audit/stack_chrono_h30/experiment_two_stage.json`
(seed 0, net 0.8): candidate threshold 0.133968, Stage 1 396 trees, 207,935 train and 511,170 validation
candidate rows (29,758 and 6,182 positive), LightGBM Stage 2 142 trees, Cox 272, stack weights LightGBM
0.6 / Cox 0.4, validation drive-level average precision 0.28492 (stack), 0.27363 (LightGBM), 0.27533
(Cox). Every value equals the experiment's (`freeze_report.json`, `reproduction`).

## Files

| File | What |
|---|---|
| `stage1/` | Stage 1 LightGBM (MLflow LightGBM format) |
| `stage2_lightgbm/` | LightGBM Stage 2 (MLflow LightGBM format) |
| `stage2_cox/` | Cox Stage 2 (`cox.json` booster, `cox_meta.json` tree count) |
| `references.npz` | Each member's sorted validation candidate scores (the rank reference of the blend) |
| `stack.json` | Weights, members, candidate threshold |
| `agent_thresholds_stack_chrono_h30_seed0.json` | Tiers for the stack's score: `action_tiers` (lift), `action_tiers_cost` (cost), primary `threshold`, and their validation precision and recall; `score_column: score` |
| `agent_thresholds_lightgbm_chrono_h30_seed0.json`, `agent_thresholds_cox_chrono_h30_seed0.json` | The same for the LightGBM and Cox Stage 2 of this refit (reference arms; `score_lightgbm`, `score_cox`) |
| `cost_tiers_validation.md` | Cost-optimal operating points on validation for ratios 2-100, the three arms and the `chrono_h30` LightGBM cascade |
| `freeze_report.json` | Settings, git commit, the reproduction check, fit reports, tiers, sensitivity |
| `CHECKSUMS.sha256` | SHA-256 of every file here; `sha256sum -c CHECKSUMS.sha256` from this folder |

The validation scores of the three arms (9,541,741 rows, `split, drive_id, date, label, score,
score_lightgbm, score_cox`) are in `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet` (189 MB,
untracked like the other score tables; SHA-256
`b5fb5e41de6a031d1f4f8bc24140500da9e315f8fb062e9a240f43072e9d0c82`).

## Validation tiers (drive level; failure rate 0.192%, 684 failing of 356,811 drives)

Precision / recall (drives flagged) at each tier; the lift tiers use `threshold.action_tier_lift`, converted
to precision at this validation rate (10.9, 18.1, 29.0 and 43.4%).

| Arm | Lift warn | Lift cordon | Lift migrate | Lift drain | Primary |
|---|---|---|---|---|---|
| Stack | 10.9% / 59.4% (3,741) | 18.1% / 49.0% (1,853) | 29.0% / 39.0% (922) | 43.5% / 22.5% (354) | 52.8% / 12.6% (163), threshold 0.9979 |
| LightGBM Stage 2 | 10.9% / 59.4% (3,741) | 18.1% / 48.7% (1,842) | 29.0% / 37.6% (887) | 43.5% / 19.4% (306) | 48.6% / 10.2% (144), threshold 0.9506 |
| Cox Stage 2 | 10.9% / 59.9% (3,778) | 18.1% / 49.0% (1,853) | 29.0% / 37.9% (894) | 43.4% / 23.2% (366) | 49.8% / 15.2% (209), threshold 0.9555 |

- **The 68% goal is not met on validation**: the primary threshold gives 52.8% at 12.6% recall for the
  stack (runbook 1.1.9). All four tiers fire for every arm, unlike on the drive-holdout split.
- At ratio 20 the cost value `20 x caught - false alarms` is 5,380 (stack), 5,351 (LightGBM), 5,322 (Cox)
  and 5,346 (the `chrono_h30` cascade): the arms are worth about the same.
- Cost tiers use the placeholder ratios of `configs/model.yaml threshold.action_cost_ratio` (warn 100, cordon
  20, migrate 5, drain 2), an operations decision not yet made.
- These are validation numbers: the members stopped early on validation candidates and the blend weights were
  fitted there, so they are optimistic. Test results of the same fits are in runbook 1.1.9 (stack, the 10%
  level: 55.4% precision at 16.2% recall, mean of 3 seeds; seed 0 alone 58.6% at 16.2%).
- Seeds 1 and 2 are saved in `data/audit/stack_chrono_h30_rerun/models/seed<k>_net0.80/`, from a re-run of the
  experiment with its model-saving option (default on; `--no-save-models` turns it off). That re-run's seed 0
  scores identically to this folder (largest difference 2e-15), so the two fits agree.
