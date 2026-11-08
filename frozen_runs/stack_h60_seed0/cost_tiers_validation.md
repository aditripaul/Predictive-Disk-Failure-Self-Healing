# Cost-optimal operating points on validation (drive level)

For each ratio r (how many unnecessary actions one prevented failure is worth), the
threshold that maximises `r * caught - false_alarms` on the validation drive table, and
its precision, recall and share of the fleet flagged. Written by
`scripts/freeze_stack_scorer.py`; validation only. The optimum depends on the failure
rate: on a period with half the failure rate (Q3 has about half of validation's), the
same threshold flags more than is optimal there.

## stack (validation failure rate 0.679%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9840 | 66.2% | 43.2% | 208 | 314 | 0.4% |
| 5 | 0.9686 | 52.4% | 54.8% | 264 | 504 | 0.7% |
| 10 | 0.9375 | 39.7% | 63.1% | 304 | 766 | 1.1% |
| 20 | 0.8845 | 29.5% | 67.4% | 325 | 1,101 | 1.6% |
| 50 | 0.7679 | 20.5% | 72.8% | 351 | 1,711 | 2.4% |
| 100 | 0.2158 | 6.9% | 82.2% | 396 | 5,777 | 8.1% |

## lightgbm (validation failure rate 0.679%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.8287 | 61.1% | 47.3% | 228 | 373 | 0.5% |
| 5 | 0.7947 | 46.9% | 58.7% | 283 | 604 | 0.9% |
| 10 | 0.7434 | 39.4% | 62.4% | 301 | 764 | 1.1% |
| 20 | 0.5242 | 22.9% | 71.4% | 344 | 1,499 | 2.1% |
| 50 | 0.4817 | 20.3% | 73.2% | 353 | 1,742 | 2.5% |
| 100 | 0.2158 | 6.9% | 82.2% | 396 | 5,777 | 8.1% |

## cox (validation failure rate 0.679%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.8767 | 62.0% | 47.1% | 227 | 366 | 0.5% |
| 5 | 0.8469 | 48.9% | 56.2% | 271 | 554 | 0.8% |
| 10 | 0.7871 | 36.3% | 64.1% | 309 | 852 | 1.2% |
| 20 | 0.7208 | 29.4% | 67.8% | 327 | 1,114 | 1.6% |
| 50 | 0.5904 | 20.3% | 72.8% | 351 | 1,729 | 2.4% |
| 100 | 0.4650 | 14.9% | 75.5% | 364 | 2,441 | 3.4% |

## cascade (frozen LightGBM run, its validation rows) (validation failure rate 0.679%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.8424 | 57.8% | 49.8% | 240 | 415 | 0.6% |
| 5 | 0.8085 | 48.1% | 57.7% | 278 | 578 | 0.8% |
| 10 | 0.7772 | 42.6% | 61.6% | 297 | 698 | 1.0% |
| 20 | 0.5861 | 26.0% | 69.3% | 334 | 1,285 | 1.8% |
| 50 | 0.4961 | 20.8% | 73.0% | 352 | 1,690 | 2.4% |
| 100 | 0.2365 | 7.2% | 82.0% | 395 | 5,497 | 7.7% |
