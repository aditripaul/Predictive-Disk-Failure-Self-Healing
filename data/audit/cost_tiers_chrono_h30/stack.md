# Lift tiers against cost tiers: stack_chrono_h30_seed0

Written by `scripts/cost_tier_report.py` from validation rows only (score table `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`, column `score`; commit `e669a84`). Both tier sets are tuned on these validation drives, so the cost tiers' net value is in-sample by construction: read this as where the cost tiers would sit and what the lift tiers assume, not as one set beating the other. Late hits (failures in days 31-60) are not credited; on the chronological split those days lie in the test period.

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

The range of `r` for which the lift tier's threshold is the cost-optimal cut on the validation drives, and the marginal precision that implies. A lift tier whose implied `r` is below the ratio in force is stricter than its cost justifies (flagging more would pay); above it, looser; within 10%, about the same. A precision target usually lands on a jagged point just under the curve's convex hull, where no `r` is exactly optimal; its implied `r` is then read from the hull segment above it (4 of 4 tiers here).

| Tier | Lift threshold | Implied r | Marginal precision at the cut | r in force | Verdict |
|---|---|---|---|---|---|
| warn | 0.9141 | 35.6 | 2.7% | 100 | stricter than its cost |
| cordon | 0.9651 | 16.4 | 5.8% | 20 | stricter than its cost |
| migrate | 0.9844 | 5.1 | 16.3% | 5 | about the same |
| drain | 0.9951 | 3.6 | 21.7% | 2 | looser than its cost |

## Validation drive table

Each tier read on its own: every drive at or above its threshold takes that action. Net value is `r x caught - false alarms`, in actions of that tier.

| Tier | Rule | Threshold | Flagged | Caught | False alarms | Precision | Recall | Net (actions) |
|---|---|---|---|---|---|---|---|---|
| warn | lift | 0.9141 | 3,741 | 406 | 3,335 | 10.9% | 59.4% | 37,265 |
| warn | cost | 0.8197 | 6,405 | 465 | 5,940 | 7.3% | 68.0% | 40,560 |
| cordon | lift | 0.9651 | 1,853 | 335 | 1,518 | 18.1% | 49.0% | 5,182 |
| cordon | cost | 0.9506 | 2,474 | 374 | 2,100 | 15.1% | 54.7% | 5,380 |
| migrate | lift | 0.9844 | 922 | 267 | 655 | 29.0% | 39.0% | 680 |
| migrate | cost | 0.9861 | 838 | 254 | 584 | 30.3% | 37.1% | 686 |
| drain | lift | 0.9951 | 354 | 154 | 200 | 43.5% | 22.5% | 108 |
| drain | cost | 0.9953 | 344 | 154 | 190 | 44.8% | 22.5% | 118 |

## Point in time

9 validation snapshots, 3,182,259 drive-snapshots, 2,819 failing within 30 days (`data/audit/agent_pit_validation_chrono_h30/stack_observations.parquet`). Per snapshot means; snapshots overlap, so they are not additive. Precision and recall are per drive-snapshot, lower than per drive.

| Tier | Rule | Flagged / snapshot | Caught / snapshot | Precision | Recall | Net / snapshot |
|---|---|---|---|---|---|---|
| warn | lift | 1,863.1 | 151.8 | 8.1% | 48.5% | 13,466.4 |
| warn | cost | 3,893.0 | 185.3 | 4.8% | 59.2% | 14,825.7 |
| cordon | lift | 750.7 | 111.9 | 14.9% | 35.7% | 1,599.0 |
| cordon | cost | 1,060.0 | 125.8 | 11.9% | 40.2% | 1,581.3 |
| migrate | lift | 329.6 | 73.1 | 22.2% | 23.3% | 109.1 |
| migrate | cost | 290.3 | 68.3 | 23.5% | 21.8% | 119.7 |
| drain | lift | 99.9 | 34.8 | 34.8% | 11.1% | 4.4 |
| drain | cost | 95.4 | 34.1 | 35.7% | 10.9% | 6.9 |

## Cross-check

Recomputed lift tiers match `frozen_runs/stack_chrono_h30_seed0/agent_thresholds_stack_chrono_h30_seed0.json` exactly.
