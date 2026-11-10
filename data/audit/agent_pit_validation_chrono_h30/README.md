# Point-in-time baseline on the chronological validation window, 30-day horizon

Runbook Section 1.1.9, "The agent's evidence on this split". Written by `make chrono-pit-validation` (launcher
`data/audit/chrono_logs/run_agent_evidence.sh`, log `data/audit/chrono_logs/pit.log`), 2026-11-10 18:59 to 19:02 IST,
code commit `553770b` plus the uncommitted Makefile targets of that day.

Nine snapshots three days apart, 2026-05-23..06-16, 30-day horizon (the last horizon ends 07-16, the day before the
test starts), every drive that reports: **validation rows only**, no test row is read (the test-access log stayed at
146 entries). 3,182,259 drive-snapshots of 356,768 drives, 2,819 failing, base rate 0.089%.

| File | Arm | Scores | Tiers |
|---|---|---|---|
| `stack.json` | Stacked Stage 2, seed 0 (`frozen_runs/stack_chrono_h30_seed0`) | `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`, `score`; cross-checked against the frozen model on 2026-06-04 (353,133 drives, maximum difference 0.0) | `agent_thresholds_stack_chrono_h30_seed0.json` |
| `lightgbm.json` | LightGBM Stage 2 of the same refit | same table, `score_lightgbm` | `agent_thresholds_lightgbm_chrono_h30_seed0.json` |
| `cox.json` | Cox Stage 2 of the same refit | same table, `score_cox` | `agent_thresholds_cox_chrono_h30_seed0.json` |
| `cascade.json` | LightGBM two-stage run `6ea7a5bc` | `data/tmp/lightgbm_scores_h30_chrono_h30.parquet`, `score` | `frozen_runs/6ea7a5bc.../action_tiers.json` |

Precision and recall are per drive-snapshot, so they are lower than the per-drive figures of Section 1.1.9. The
merged observation files (`<arm>_observations.parquet`, about 53 MB each) and the per-date partitions
(`<arm>/date=*.parquet`) stay local (`.gitignore`; the 60-day folder's are 10 MB each and are tracked); a rerun
reuses the partitions and rewrites the merged files.

Reproduce (needs the frozen scorer and the score tables of `make chrono-freeze`):

```bash
make chrono-pit-validation          # about 3 minutes; CHRONO_SEED=<k> picks another seed's scorer
```

`CHECKSUMS.sha256` covers every tracked file here; verify with `sha256sum -c CHECKSUMS.sha256` from this folder.
