# Point-in-time baseline on validation drives, 60-day horizon (frozen)

Phase 0 of `docs/agent_knowledge_base.md`. Written by `make agent-pit-validation`
(launcher `data/audit/agent_pit_validation_h60_logs/run_pit.sh`) at commit `787b8b8`,
2026-11-07 13:53-13:54 UTC.

Nine weekly snapshots, 2026-01-05..2026-03-02, 60-day horizon (every horizon ends by
2026-05-01, the end of validation), **validation holdout drives only**: no training drive
is scored and no test row is read (the test-access log stayed at 139 lines). Each arm is
judged on the same 606,194 drive-snapshots of 68,428 drives (1,191 failing; base rate
0.196%), with its own validation-tuned lift tiers.

| File | Arm | Scores | Tiers |
|---|---|---|---|
| `stack.json`, `stack_observations.parquet` | Stacked Stage 2 (`frozen_runs/stack_h60_seed0`) | `data/tmp/stack_scores/scores_h60_seed0.parquet`, column `score`; cross-checked against the frozen model on 2026-02-23 | `agent_thresholds_stack_h60_seed0.json` |
| `lightgbm.json`, `lightgbm_observations.parquet` | LightGBM Stage 2 of the same refit | same table, `score_lightgbm` | `agent_thresholds_lightgbm_h60_seed0.json` |
| `cox.json`, `cox_observations.parquet` | Cox Stage 2 of the same refit | same table, `score_cox` | `agent_thresholds_cox_h60_seed0.json` |
| `cascade.json`, `cascade_observations.parquet` | Frozen 60-day LightGBM cascade `804a20ce` | `data/tmp/lightgbm_scores_h60_h60_q1q2_q3.parquet`, `score` | `frozen_runs/804a20ce.../action_tiers.json` |

**Cross-check (stack):** on 2026-02-23 the frozen model and the score table score the same
67,477 drives with a maximum difference of 0.0.

**Results** (pooled over the nine snapshots; 95% intervals resample whole drives):

| Arm | >= warn precision / recall | >= cordon precision [CI] / recall |
|---|---|---|
| Stack | 28.3% / 31.2% | 53.7% [44, 64] / 13.4% |
| LightGBM Stage 2 | 28.2% / 31.0% | 49.5% [40, 60] / 12.3% |
| Cox Stage 2 | 28.6% / 30.8% | 56.2% [46, 67] / 12.9% |
| Cascade `804a20ce` | 28.6% / 29.8% | 49.6% [40, 60] / 11.6% |

The arms are indistinguishable within their intervals. Migrate and drain do not fire for
the stack, LightGBM or Cox on their lift tiers (the cascade fires on 2 snapshots). Recall
is per snapshot (a failing drive is judged on every snapshot inside its window), so it is
lower than the per-drive recall of the frozen scorer's README.

The per-date partitions (`<arm>/date=*.parquet`) duplicate the merged observation files and
stay local (`.gitignore`); `make agent-pit-validation` reuses them on a rerun. The run log
(`../agent_pit_validation_h60_logs/pit.log`) is tracked, its paths made repo-relative on
2026-11-08 with `scripts/scrub_paths.py` (this README and its checksum updated to say so).
`CHECKSUMS.sha256` covers every tracked file here; verify with `sha256sum -c CHECKSUMS.sha256` from this folder.
