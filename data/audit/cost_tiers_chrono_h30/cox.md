# Lift tiers against cost tiers: cox_chrono_h30_seed0

Written by `scripts/cost_tier_report.py` from validation rows only (score table `data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`, column `score_cox`; commit `e669a84`). Both tier sets are tuned on these validation drives, so the cost tiers' net value is in-sample by construction: read this as where the cost tiers would sit and what the lift tiers assume, not as one set beating the other. Late hits (failures in days 31-60) are not credited; on the chronological split those days lie in the test period.

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
| warn | 0.7551 | 35.6 | 2.7% | 100 | stricter than its cost |
| cordon | 0.8709 | 17.6 | 5.4% | 20 | stricter than its cost |
| migrate | 0.9184 | 6.8 | 12.8% | 5 | looser than its cost |
| drain | 0.9465 | 3.3 | 23.2% | 2 | looser than its cost |

## Validation drive table

Each tier read on its own: every drive at or above its threshold takes that action. Net value is `r x caught - false alarms`, in actions of that tier.

| Tier | Rule | Threshold | Flagged | Caught | False alarms | Precision | Recall | Net (actions) |
|---|---|---|---|---|---|---|---|---|
| warn | lift | 0.7551 | 3,778 | 410 | 3,368 | 10.9% | 59.9% | 37,632 |
| warn | cost | 0.5300 | 6,559 | 468 | 6,091 | 7.1% | 68.4% | 40,709 |
| cordon | lift | 0.8709 | 1,853 | 335 | 1,518 | 18.1% | 49.0% | 5,182 |
| cordon | cost | 0.8606 | 2,049 | 351 | 1,698 | 17.1% | 51.3% | 5,322 |
| migrate | lift | 0.9184 | 894 | 259 | 635 | 29.0% | 37.9% | 660 |
| migrate | cost | 0.9264 | 709 | 237 | 472 | 33.4% | 34.6% | 713 |
| drain | lift | 0.9465 | 366 | 159 | 207 | 43.4% | 23.2% | 111 |
| drain | cost | 0.9500 | 301 | 141 | 160 | 46.8% | 20.6% | 122 |

## Point in time

9 validation snapshots, 3,182,259 drive-snapshots, 2,819 failing within 30 days (`data/audit/agent_pit_validation_chrono_h30/cox_observations.parquet`). Per snapshot means; snapshots overlap, so they are not additive. Precision and recall are per drive-snapshot, lower than per drive.

| Tier | Rule | Flagged / snapshot | Caught / snapshot | Precision | Recall | Net / snapshot |
|---|---|---|---|---|---|---|
| warn | lift | 1,927.6 | 154.0 | 8.0% | 49.2% | 13,626.4 |
| warn | cost | 4,153.6 | 186.3 | 4.5% | 59.5% | 14,666.1 |
| cordon | lift | 748.0 | 110.8 | 14.8% | 35.4% | 1,578.3 |
| cordon | cost | 843.1 | 116.8 | 13.9% | 37.3% | 1,609.2 |
| migrate | lift | 309.4 | 72.3 | 23.4% | 23.1% | 124.6 |
| migrate | cost | 241.2 | 61.3 | 25.4% | 19.6% | 126.8 |
| drain | lift | 99.6 | 35.1 | 35.3% | 11.2% | 5.8 |
| drain | cost | 79.0 | 29.9 | 37.8% | 9.5% | 10.7 |

## Cross-check

Recomputed lift tiers match `frozen_runs/stack_chrono_h30_seed0/agent_thresholds_cox_chrono_h30_seed0.json` exactly.
