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
| 2 | 0.9953 | 44.8% | 22.5% | 154 | 344 | 0.1% |
| 5 | 0.9861 | 30.3% | 37.1% | 254 | 838 | 0.2% |
| 10 | 0.9797 | 25.2% | 43.3% | 296 | 1,176 | 0.3% |
| 20 | 0.9506 | 15.1% | 54.7% | 374 | 2,474 | 0.7% |
| 50 | 0.8803 | 9.2% | 63.6% | 435 | 4,725 | 1.3% |
| 100 | 0.8197 | 7.3% | 68.0% | 465 | 6,405 | 1.8% |

## lightgbm (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9433 | 47.3% | 15.6% | 107 | 226 | 0.1% |
| 5 | 0.8759 | 31.3% | 35.8% | 245 | 783 | 0.2% |
| 10 | 0.8173 | 23.4% | 44.3% | 303 | 1,296 | 0.4% |
| 20 | 0.6861 | 15.9% | 53.2% | 364 | 2,293 | 0.6% |
| 50 | 0.3667 | 8.2% | 65.5% | 448 | 5,446 | 1.5% |
| 100 | 0.2863 | 6.8% | 68.6% | 469 | 6,904 | 1.9% |

## cox (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9500 | 46.8% | 20.6% | 141 | 301 | 0.1% |
| 5 | 0.9264 | 33.4% | 34.6% | 237 | 709 | 0.2% |
| 10 | 0.8959 | 23.1% | 45.9% | 314 | 1,359 | 0.4% |
| 20 | 0.8606 | 17.1% | 51.3% | 351 | 2,049 | 0.6% |
| 50 | 0.6642 | 9.2% | 64.0% | 438 | 4,783 | 1.3% |
| 100 | 0.5300 | 7.1% | 68.4% | 468 | 6,559 | 1.8% |

## cascade (frozen LightGBM run, its validation rows) (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9137 | 40.5% | 25.1% | 172 | 425 | 0.1% |
| 5 | 0.8699 | 31.5% | 36.0% | 246 | 780 | 0.2% |
| 10 | 0.8363 | 25.9% | 42.8% | 293 | 1,130 | 0.3% |
| 20 | 0.6833 | 15.7% | 53.4% | 365 | 2,319 | 0.6% |
| 50 | 0.4908 | 10.2% | 61.5% | 421 | 4,122 | 1.2% |
| 100 | 0.2875 | 6.5% | 69.2% | 473 | 7,296 | 2.0% |
