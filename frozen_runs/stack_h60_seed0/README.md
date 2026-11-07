# `stack_h60_seed0`: the 60-day stacked Stage 2, frozen as a scorer

Phase S of `docs/agent_knowledge_base.md`. The stacked Stage 2 (LightGBM and Cox Stage 2
models blended by validation-ranked scores on the Stage 1 candidates; runbook 1.1.4a) at
the 60-day horizon, seed 0, net 0.8. The stack runs of 1.1.4a saved no models and no
per-drive scores; this folder holds the models, so the knowledge base and the agent can
use the stack.

**Not an MLflow run** (`log_two_stage` accepts a LightGBM Stage 2 only). Load it with
`src/models/stacking.py::load_stacked_two_stage(Path("frozen_runs/stack_h60_seed0"))`.
Frozen: never retrained or overwritten (`save_stacked_two_stage` refuses an existing folder).

## How it was made

`scripts/freeze_stack_scorer.py --work-dir data/tmp/train_model_frame_xjxdieoe` (launcher
`data/audit/stack_scorer_h60_logs/run_freeze.sh`), code commit `b8da3eb` (clean tree),
2026-11-07 13:40-13:44 UTC. The work dir is that of the 60-day frozen run
`804a20cec10048abb1bc3ccf510f96e1` (`h60_q1q2_q3`). The script refits exactly what
`make experiment-model ... --two-stage --stage2-kinds lightgbm cox stack` fits for this
seed and net, with the same functions and settings, and **never opens the test matrix**:
no test access was recorded (the access log stayed at 139 lines).

**Reproduction check: passed, 14 of 14**, against
`data/audit/stack_q3_h60/experiment_two_stage.json` (seed 0, net 0.8): candidate threshold
0.2716041601, Stage 1 355 trees, 176,461 train / 234,275 validation candidate rows,
LightGBM Stage 2 43 trees, Cox 176, stack weights LightGBM 0.5 / Cox 0.5, validation
drive-level average precision 0.59589 (stack), 0.57567 (LightGBM), 0.59325 (Cox). Every
value is equal to the frozen run's (`freeze_report.json`, `reproduction`).

## Files

| File | What |
|---|---|
| `stage1/` | Stage 1 LightGBM (MLflow LightGBM format) |
| `stage2_lightgbm/` | LightGBM Stage 2 (MLflow LightGBM format) |
| `stage2_cox/` | Cox Stage 2 (`cox.json` booster, `cox_meta.json` tree count) |
| `references.npz` | Each member's sorted validation candidate scores (the rank reference of the blend) |
| `stack.json` | Weights, members, candidate threshold |
| `agent_thresholds_stack_h60_seed0.json` | Tiers for the stack's score: `action_tiers` (lift), `action_tiers_cost` (cost), primary `threshold`, and their validation precision/recall; `score_column: score` |
| `agent_thresholds_lightgbm_h60_seed0.json`, `agent_thresholds_cox_h60_seed0.json` | The same for the LightGBM and Cox Stage 2 of this refit (reference arms; `score_lightgbm`, `score_cox`) |
| `cost_tiers_validation.md` | Cost-optimal operating points on validation for ratios 2-100, the three arms and the frozen LightGBM cascade |
| `freeze_report.json` | Settings, git commit, the reproduction check, fit reports, tiers, sensitivity |
| `CHECKSUMS.sha256` | SHA-256 of every file here; `sha256sum -c CHECKSUMS.sha256` from this folder |

The validation scores of the three arms (8,242,015 rows, `split, drive_id, date, label,
score, score_lightgbm, score_cox`) are in `data/tmp/stack_scores/scores_h60_seed0.parquet`
(156 MB, untracked like the other score tables; SHA-256 starts `06684d5f158d59cd`).

## Validation tiers (drive level; failure rate 0.679%, 482 failing of 70,969 drives)

| Arm | Lift warn P / R | Lift cordon P / R | Primary P / R |
|---|---|---|---|
| Stack | 38.5% / 63.1% | 64.0% / 44.0% | 82.7% / 18.9% |
| LightGBM Stage 2 | 38.4% / 62.4% | 64.2% / 42.1% | 79.4% / 11.2% |
| Cox Stage 2 | 38.5% / 62.7% | 64.2% / 43.6% | 81.9% / 17.8% |

- Lift migrate and drain fire on 4 drives (stack), 2 (Cox) or none (LightGBM): not usable
  tiers, as for the cascade (runbook 1.9).
- At equal cost the arms are worth the same on validation: at ratio 20, `20 x caught - false
  alarms` is 5,724 (stack), 5,725 (LightGBM), 5,753 (Cox), 5,729 (frozen cascade); at ratio 2,
  310, 311, 315, 305.
- Cost tiers use the placeholder ratios of `configs/model.yaml threshold.action_cost_ratio`
  (warn 100, cordon 20, migrate 5, drain 2), an operations decision not yet made. At warn
  (100) the stack's threshold, 0.216, is below the Stage 1 net (0.272), so it flags 8.1% of
  drives on Stage 1's score alone.
- These are validation numbers: the members stopped early on validation candidates and the
  blend weights were fitted there, so they are optimistic, and precision is at the
  validation failure rate (Q3's is about half).
