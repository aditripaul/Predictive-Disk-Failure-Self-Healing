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
| 2 | 0.9968 | 47.4% | 18.9% | 129 | 272 | 0.1% |
| 5 | 0.9906 | 36.1% | 32.6% | 223 | 617 | 0.2% |
| 10 | 0.9801 | 26.5% | 43.0% | 294 | 1,108 | 0.3% |
| 20 | 0.9558 | 16.4% | 52.6% | 360 | 2,190 | 0.6% |
| 50 | 0.8933 | 10.0% | 62.3% | 426 | 4,277 | 1.2% |
| 100 | 0.7689 | 6.3% | 69.2% | 473 | 7,509 | 2.1% |

## lightgbm (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9347 | 42.7% | 23.7% | 162 | 379 | 0.1% |
| 5 | 0.8898 | 33.4% | 33.9% | 232 | 695 | 0.2% |
| 10 | 0.7990 | 24.3% | 43.7% | 299 | 1,230 | 0.3% |
| 20 | 0.6346 | 14.8% | 53.8% | 368 | 2,483 | 0.7% |
| 50 | 0.3680 | 9.0% | 63.5% | 434 | 4,799 | 1.3% |
| 100 | 0.2774 | 6.8% | 67.5% | 462 | 6,753 | 1.9% |

## cox (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9491 | 42.4% | 24.6% | 168 | 396 | 0.1% |
| 5 | 0.9333 | 34.9% | 34.4% | 235 | 673 | 0.2% |
| 10 | 0.9065 | 25.7% | 43.1% | 295 | 1,148 | 0.3% |
| 20 | 0.8550 | 16.3% | 52.6% | 360 | 2,214 | 0.6% |
| 50 | 0.6090 | 8.5% | 65.1% | 445 | 5,219 | 1.5% |
| 100 | 0.4758 | 6.3% | 69.3% | 474 | 7,516 | 2.1% |

## cascade (frozen LightGBM run, its validation rows) (validation failure rate 0.192%)

| r | Threshold | Precision | Recall | Caught | Flagged | Share of fleet |
|---|---|---|---|---|---|---|
| 2 | 0.9137 | 40.5% | 25.1% | 172 | 425 | 0.1% |
| 5 | 0.8699 | 31.5% | 36.0% | 246 | 780 | 0.2% |
| 10 | 0.8363 | 25.9% | 42.8% | 293 | 1,130 | 0.3% |
| 20 | 0.6833 | 15.7% | 53.4% | 365 | 2,319 | 0.6% |
| 50 | 0.4908 | 10.2% | 61.5% | 421 | 4,122 | 1.2% |
| 100 | 0.2875 | 6.5% | 69.2% | 473 | 7,296 | 2.0% |
