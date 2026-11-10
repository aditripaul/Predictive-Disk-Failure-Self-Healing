# Cost-optimal operating points on validation (drive level)

For each ratio r (how many unnecessary actions one prevented failure is worth), the
threshold that maximises `r * caught - false_alarms` on the validation drive table, and
its precision, recall and share of the fleet flagged. Written by
`scripts/freeze_stack_scorer.py`; validation only. The optimum depends on the failure
rate: on a period with half the failure rate (Q3 has about half of validation's), the
same threshold flags more than is optimal there.

## stack (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9976 | 51.3% | 14.8% | 101 | 197 | 0.1% |
| 5 | 0.9866 | 32.1% | 37.7% | 258 | 803 | 0.2% |
| 10 | 0.9799 | 25.8% | 43.1% | 295 | 1,143 | 0.3% |
| 20 | 0.9566 | 16.5% | 52.9% | 362 | 2,197 | 0.6% |
| 50 | 0.8514 | 8.5% | 65.6% | 449 | 5,297 | 1.5% |
| 100 | 0.8117 | 7.2% | 68.1% | 466 | 6,467 | 1.8% |

## lightgbm (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9482 | 45.0% | 17.7% | 121 | 269 | 0.1% |
| 5 | 0.8904 | 33.6% | 33.2% | 227 | 676 | 0.2% |
| 10 | 0.7909 | 21.4% | 47.2% | 323 | 1,507 | 0.4% |
| 20 | 0.7056 | 16.9% | 52.5% | 359 | 2,120 | 0.6% |
| 50 | 0.3613 | 8.3% | 65.6% | 449 | 5,420 | 1.5% |
| 100 | 0.3108 | 7.1% | 67.8% | 464 | 6,551 | 1.8% |

## cox (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9479 | 42.3% | 22.5% | 154 | 364 | 0.1% |
| 5 | 0.9267 | 33.6% | 36.1% | 247 | 735 | 0.2% |
| 10 | 0.9042 | 24.6% | 43.9% | 300 | 1,218 | 0.3% |
| 20 | 0.8588 | 17.4% | 51.9% | 355 | 2,042 | 0.6% |
| 50 | 0.6343 | 8.8% | 65.1% | 445 | 5,057 | 1.4% |
| 100 | 0.5392 | 7.1% | 68.1% | 466 | 6,562 | 1.8% |

## cascade (frozen LightGBM run, its validation rows) (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9137 | 40.5% | 25.1% | 172 | 425 | 0.1% |
| 5 | 0.8699 | 31.5% | 36.0% | 246 | 780 | 0.2% |
| 10 | 0.8363 | 25.9% | 42.8% | 293 | 1,130 | 0.3% |
| 20 | 0.6833 | 15.7% | 53.4% | 365 | 2,319 | 0.6% |
| 50 | 0.4908 | 10.2% | 61.5% | 421 | 4,122 | 1.2% |
| 100 | 0.2875 | 6.5% | 69.2% | 473 | 7,296 | 2.0% |
