# Lift tiers against cost tiers on the chronological validation window, 30-day horizon

Runbook Section 1.9, "What the lift tiers assume, on the chronological run". Written by `make chrono-cost-tiers`
(`scripts/cost_tier_report.py`) on 2026-11-10 from **validation rows only**: the score tables are filtered to
`split == "validation"` before anything is materialised, and the point-in-time observations are the nine validation
snapshots of `make chrono-pit-validation`. No test row is read, and the test-access log is unchanged. Late hits are
not credited, because days 31-60 after a validation snapshot fall in the test period.

| File | Arm | Scores | Lift tiers cross-checked against |
|---|---|---|---|
| `stack.md` / `.json` | Stacked Stage 2, seed 0 | `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`, `score` | `frozen_runs/stack_chrono_h30_seed0/agent_thresholds_stack_chrono_h30_seed0.json` (exact) |
| `lightgbm.md` / `.json` | LightGBM Stage 2 of the same refit | same table, `score_lightgbm` | `.../agent_thresholds_lightgbm_chrono_h30_seed0.json` (exact) |
| `cox.md` / `.json` | Cox Stage 2 of the same refit | same table, `score_cox` | `.../agent_thresholds_cox_chrono_h30_seed0.json` (exact) |
| `cascade.md` / `.json` | LightGBM two-stage run `6ea7a5bc` | `data/tmp/lightgbm_scores_h30_chrono_h30.parquet`, `score` | `frozen_runs/6ea7a5bc.../action_tiers.json` (exact) |

Each report contains:

- the ratios in force: the placeholder `threshold.action_cost_ratio` (warn 100, cordon 20, migrate 5, drain 2),
  because `threshold.cost_model` is not filled in;
- the ratio each lift tier implies;
- both tier sets on the validation drive table and per snapshot.

Both tier sets are tuned on the same validation drives, so the cost tiers' net value is in-sample.

**Result.** The lift tiers are cost-optimal for about r = 35 (warn), 16-20 (cordon), 5-7 (migrate) and 2-4 (drain) on
every arm. At the placeholder ratios the cost tiers move warn to 68-69% drive recall at 6.5-7.3% precision, against
59-60% at 10.9%, and cordon to 51-55% recall, against 49%. Migrate and drain barely move. The net value gained is
8-9% at warn and 2-4% at cordon.

Reproduce (needs the score tables of `make chrono-freeze` and the observations of `make chrono-pit-validation`):

```bash
make chrono-cost-tiers                                   # about 2 minutes
make chrono-cost-tiers COST_ARGS="--failure-cost 1000 --action-cost warn=1 cordon=10 migrate=50 drain=200 \
    --p-save warn=0.1 cordon=0.5 migrate=0.8 drain=0.9"  # what-if costs; overwrites these files
```
