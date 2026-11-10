# Lift tiers against cost tiers: lightgbm_chrono_h30_seed0

Written by `scripts/cost_tier_report.py` from validation rows only (score table `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`, column `score_lightgbm`; commit `e669a84`). Both tier sets are tuned on these validation drives, so the cost tiers' net value is in-sample by construction: read this as where the cost tiers would sit and what the lift tiers assume, not as one set beating the other. Late hits (failures in days 31-60) are not credited; on the chronological split those days lie in the test period.

Validation: 356,811 drives, 684 failing (0.192%).

## Ratios in force (source: action_cost_ratio (placeholders))

`r` is how many unnecessary actions of that kind one prevented failure is worth; an action pays while the marginal precision of the drives it adds is at least `1 / (1 + r)`.

| Tier | r | Break-even marginal precision |
|---|---|---|
| warn | 100 | 1.0% |
| cordon | 20 | 4.8% |
| migrate | 5 | 16.7% |
| drain | 2 | 33.3% |

These are the placeholder ratios. Fill in `threshold.cost_model` in `configs/model.yaml` (or pass `--failure-cost`, `--action-cost`, `--p-save`) to derive them from costs.

## What each lift tier assumes

The range of `r` for which the lift tier's threshold is the cost-optimal cut on the validation drives, and the marginal precision that implies. A lift tier whose implied `r` is below the ratio in force is stricter than its cost justifies (flagging more would pay); above it, looser; within 10%, about the same. A precision target usually lands on a jagged point just under the curve's convex hull, where no `r` is exactly optimal; its implied `r` is then read from the hull segment above it (3 of 4 tiers here).

| Tier | Lift threshold | Implied r | Marginal precision at the cut | r in force | Verdict |
|---|---|---|---|---|---|
| warn | 0.5234 | 34.6 | 2.8% | 100 | stricter than its cost |
| cordon | 0.7403 | 17.3 | 5.5% | 20 | stricter than its cost |
| migrate | 0.8657 | 5.8 | 14.7% | 5 | looser than its cost |
| drain | 0.9348 | 2.2 to 2.6 | 27.6% to 31.6% | 2 | about the same |

## Validation drive table

Each tier read on its own: every drive at or above its threshold takes that action. Net value is `r x caught - false alarms`, in actions of that tier.

| Tier | Rule | Threshold | Flagged | Caught | False alarms | Precision | Recall | Net (actions) |
|---|---|---|---|---|---|---|---|---|
| warn | lift | 0.5234 | 3,741 | 406 | 3,335 | 10.9% | 59.4% | 37,265 |
| warn | cost | 0.2863 | 6,904 | 469 | 6,435 | 6.8% | 68.6% | 40,465 |
| cordon | lift | 0.7403 | 1,842 | 333 | 1,509 | 18.1% | 48.7% | 5,151 |
| cordon | cost | 0.6861 | 2,293 | 364 | 1,929 | 15.9% | 53.2% | 5,351 |
| migrate | lift | 0.8657 | 887 | 257 | 630 | 29.0% | 37.6% | 655 |
| migrate | cost | 0.8759 | 783 | 245 | 538 | 31.3% | 35.8% | 687 |
| drain | lift | 0.9348 | 306 | 133 | 173 | 43.5% | 19.4% | 93 |
| drain | cost | 0.9433 | 226 | 107 | 119 | 47.3% | 15.6% | 95 |

## Point in time

9 validation snapshots, 3,182,259 drive-snapshots, 2,819 failing within 30 days (`data/audit/agent_pit_validation_chrono_h30/lightgbm_observations.parquet`). Per snapshot means; snapshots overlap, so they are not additive. Precision and recall are per drive-snapshot, lower than per drive.

| Tier | Rule | Flagged / snapshot | Caught / snapshot | Precision | Recall | Net / snapshot |
|---|---|---|---|---|---|---|
| warn | lift | 1,833.4 | 150.0 | 8.2% | 47.9% | 13,316.6 |
| warn | cost | 4,191.0 | 187.0 | 4.5% | 59.7% | 14,696.0 |
| cordon | lift | 739.6 | 111.0 | 15.0% | 35.4% | 1,591.4 |
| cordon | cost | 944.0 | 121.1 | 12.8% | 38.7% | 1,599.3 |
| migrate | lift | 304.8 | 70.8 | 23.2% | 22.6% | 119.9 |
| migrate | cost | 262.8 | 63.9 | 24.3% | 20.4% | 120.6 |
| drain | lift | 88.0 | 28.8 | 32.7% | 9.2% | -1.7 |
| drain | cost | 61.0 | 20.4 | 33.5% | 6.5% | 0.3 |

## Cross-check

Recomputed lift tiers match `frozen_runs/stack_chrono_h30_seed0/agent_thresholds_lightgbm_chrono_h30_seed0.json` exactly.
