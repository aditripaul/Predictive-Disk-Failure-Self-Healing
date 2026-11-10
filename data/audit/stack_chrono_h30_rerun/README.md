# `stack_chrono_h30_rerun`: the Stage 2 experiment again, with the models saved

The same command as `data/audit/stack_chrono_h30/` (runbook Section 1.1.9), run again on 2026-11-10
(15:55 to 16:25 IST) with the code that saves what the experiment fits:

```bash
make chrono-stack CHRONO_WORK_DIR=data/tmp/train_model_frame_4pf36rc4 CHRONO_OUT=data/audit/stack_chrono_h30_rerun
```

It reads the kept work directory of the frozen run `6ea7a5bcfb954deeb4a4208e4d802744` (`chrono_h30`), runs
Stage 1, the LightGBM, Cox and stack Stage 2 at the 0.5 and 0.8 nets for seeds 0, 1 and 2, and, because the
run includes the `stack` kind, saves each seed's stack to `models/` (`pipelines/experiment_model.py`;
`--no-save-models` turns it off). MLflow run `mlflow_run.json`; code commit `b1a8056` (no `code.patch` was written, so the tracked files were unchanged at launch).

## It reproduces the first run

Compared value by value against `data/audit/stack_chrono_h30/experiment_two_stage.json` (12 runs: 3 seeds,
the 0.5 net's LightGBM, the 0.8 net's LightGBM, Cox and stack; timings and the model path left out): 3,450
numbers and labels, **one difference**, the stack's cascade C-index for seed 1 (0.87765156 against
0.87765154, 2e-8). Every precision, recall, count, threshold, tree count and blend weight is equal, so the
`tables.md` of the two folders differs in one line, the recorded git commit (`4e37458` against `b1a8056`);
the first run's tables are the ones the runbook quotes.

## Files

| Path | What |
|---|---|
| `models/seed<k>_net0.80/` | The stack of seed k (0, 1, 2), 6.7, 8.4 and 7.2 MB: `stage1/` and `stage2_lightgbm/` (MLflow LightGBM format), `stage2_cox/` (`cox.json`, `cox_meta.json`), `references.npz` (each member's sorted validation candidate scores), `stack.json` (weights, members, candidate threshold) |
| `experiment_two_stage.json`, `tables.md`, `figures/` | The results, as in `stack_chrono_h30/`; each stack report has `saved_model_dir` |
| `mlflow_run.json` | The MLflow run of this experiment |

Load a stack: `src.models.stacking.load_stacked_two_stage(Path("data/audit/stack_chrono_h30_rerun/models/seed0_net0.80"))`.
Verified: all three load; the candidate thresholds are 0.133968, 0.146683 and 0.135783 and the blend weights
LightGBM 0.6 / Cox 0.4, 0.4 / 0.6 and 0.3 / 0.7. Seed 0 scores identically (largest difference 2e-15 on
300,000 validation rows) to `frozen_runs/stack_chrono_h30_seed0/`, which `make chrono-freeze` refitted
separately, so the two fits agree.

These models are tracked, not frozen runs: `frozen_runs/` holds the seed-0 scorer with its tiers.

## Test looks

The experiment records one test look per seed in `data/audit/test_access_log.jsonl`: 3 for this run, taking
the log from 143 to 146 entries. The models themselves read no test rows after fitting; the look is the
experiment's test scoring, as in the first run.
