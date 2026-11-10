# Lift tiers against cost tiers: cascade_6ea7a5bc

Written by `scripts/cost_tier_report.py` from validation rows only (score table `data/tmp/lightgbm_scores_h30_chrono_h30.parquet`, column `score`; commit `e669a84`). Both tier sets are tuned on these validation drives, so the cost tiers' net value is in-sample by construction: read this as where the cost tiers would sit and what the lift tiers assume, not as one set beating the other. Late hits (failures in days 31-60) are not credited; on the chronological split those days lie in the test period.

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
| warn | 0.5298 | 34.0 | 2.9% | 100 | stricter than its cost |
| cordon | 0.7336 | 19.9 | 4.8% | 20 | about the same |
| migrate | 0.8572 | 6.2 | 13.9% | 5 | looser than its cost |
| drain | 0.9347 | 1.9 | 33.9% | 2 | about the same |

## Validation drive table

Each tier read on its own: every drive at or above its threshold takes that action. Net value is `r x caught - false alarms`, in actions of that tier.

| Tier | Rule | Threshold | Flagged | Caught | False alarms | Precision | Recall | Net (actions) |
|---|---|---|---|---|---|---|---|---|
| warn | lift | 0.5298 | 3,778 | 410 | 3,368 | 10.9% | 59.9% | 37,632 |
| warn | cost | 0.2875 | 7,296 | 473 | 6,823 | 6.5% | 69.2% | 40,477 |
| cordon | lift | 0.7336 | 1,869 | 338 | 1,531 | 18.1% | 49.4% | 5,229 |
| cordon | cost | 0.6833 | 2,319 | 365 | 1,954 | 15.7% | 53.4% | 5,346 |
| migrate | lift | 0.8572 | 898 | 260 | 638 | 29.0% | 38.0% | 662 |
| migrate | cost | 0.8699 | 780 | 246 | 534 | 31.5% | 36.0% | 696 |
| drain | lift | 0.9347 | 241 | 105 | 136 | 43.6% | 15.4% | 74 |
| drain | cost | 0.9137 | 425 | 172 | 253 | 40.5% | 25.1% | 91 |

## Point in time

9 validation snapshots, 3,182,259 drive-snapshots, 2,819 failing within 30 days (`data/audit/agent_pit_validation_chrono_h30/cascade_observations.parquet`). Per snapshot means; snapshots overlap, so they are not additive. Precision and recall are per drive-snapshot, lower than per drive.

| Tier | Rule | Flagged / snapshot | Caught / snapshot | Precision | Recall | Net / snapshot |
|---|---|---|---|---|---|---|
| warn | lift | 1,855.0 | 152.2 | 8.2% | 48.6% | 13,519.4 |
| warn | cost | 4,557.2 | 189.4 | 4.2% | 60.5% | 14,576.7 |
| cordon | lift | 751.1 | 111.8 | 14.9% | 35.7% | 1,596.2 |
| cordon | cost | 968.4 | 123.8 | 12.8% | 39.5% | 1,630.9 |
| migrate | lift | 318.8 | 70.1 | 22.0% | 22.4% | 101.9 |
| migrate | cost | 271.0 | 63.1 | 23.3% | 20.1% | 107.7 |
| drain | lift | 69.1 | 22.6 | 32.6% | 7.2% | -1.4 |
| drain | cost | 132.6 | 37.9 | 28.6% | 12.1% | -18.9 |

## Cross-check

Recomputed lift tiers match `frozen_runs/6ea7a5bcfb954deeb4a4208e4d802744/action_tiers.json` exactly.
