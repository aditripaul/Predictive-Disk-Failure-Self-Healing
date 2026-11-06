# Cluster-based action policy (30-day run)

Run `frozen_runs/c459046e4f1645169151e7be25ff064d`; thresholds stage1 0.235, warn 0.653, cordon 0.836, migrate 0.926, drain 0.926, primary 0.919. Validation: 4,729 suspects (335 failing) of 70,977 drives.

## Clustering ablation (validation suspects)

| Algorithm | Param | Clusters | Separation | Stability (ARI) | Silhouette | Smallest cluster | Noise | Seconds |
|---|---|---|---|---|---|---|---|---|
| agglomerative | 3 | 3 | 1.45e-04 | 0.87 | 0.40 | 14.4% | 0.0% | 2.1 |
| agglomerative | 4 | 4 | 1.31e-04 | 0.85 | 0.40 | 4.3% | 0.0% | 1.8 |
| agglomerative | 5 | 5 | 4.56e-04 | 0.86 | 0.40 | 2.7% | 0.0% | 1.8 |
| agglomerative | 6 | 6 | 6.59e-04 | 0.61 | 0.41 | 1.3% | 0.0% | 1.8 |
| agglomerative | 7 | 7 | 7.18e-04 | 0.79 | 0.33 | 1.3% | 0.0% | 1.8 |
| agglomerative | 8 | 8 | 8.01e-04 | 0.85 | 0.34 | 1.3% | 0.0% | 1.7 |
| agglomerative | 9 | 9 | 8.60e-04 | 0.85 | 0.34 | 1.3% | 0.0% | 1.7 |
| agglomerative | 10 | 10 | 9.06e-04 | 0.84 | 0.35 | 1.1% | 0.0% | 1.8 |
| gmm | 3 | 3 | 1.67e-04 | 0.74 | 0.26 | 21.4% | 0.0% | 45.9 |
| gmm | 4 | 4 | 9.52e-04 | 0.83 | 0.25 | 9.6% | 0.0% | 58.3 |
| gmm | 5 | 5 | 7.68e-04 | 0.79 | 0.21 | 7.6% | 0.0% | 68.2 |
| gmm | 6 | 6 | 7.80e-04 | 0.67 | 0.19 | 5.8% | 0.0% | 93.2 |
| gmm | 7 | 7 | 2.65e-04 | 0.71 | 0.21 | 2.9% | 0.0% | 106.3 |
| gmm | 8 | 8 | 1.06e-03 | 0.68 | 0.23 | 1.4% | 0.0% | 101.3 |
| gmm | 9 | 9 | 7.74e-04 | 0.69 | 0.24 | 2.3% | 0.0% | 99.1 |
| gmm | 10 | 10 | 1.01e-03 | 0.75 | 0.24 | 1.5% | 0.0% | 108.7 |
| hdbscan | 50 | 10 | 3.63e-04 | 0.79 | 0.13 | 4.1% | 52.9% | 2.2 |
| hdbscan | 100 | 8 | 3.13e-04 | 0.99 | 0.10 | 6.6% | 57.4% | 2.5 |
| hdbscan | 200 | 4 | 1.82e-04 | 1.00 | 0.13 | 6.7% | 56.5% | 3.7 |
| kmeans | 3 | 3 | 7.73e-05 | 0.98 | 0.41 | 13.8% | 0.0% | 6.2 |
| kmeans | 4 | 4 | -1.31e-05 | 0.86 | 0.35 | 5.7% | 0.0% | 20.6 |
| kmeans | 5 | 5 | 4.82e-04 | 0.85 | 0.37 | 2.8% | 0.0% | 10.4 |
| kmeans | 6 | 6 | 4.66e-04 | 0.80 | 0.31 | 2.7% | 0.0% | 28.5 |
| kmeans | 7 | 7 | 2.80e-04 | 0.90 | 0.32 | 2.7% | 0.0% | 9.9 |
| kmeans | 8 | 8 | 7.35e-04 | 0.92 | 0.34 | 1.4% | 0.0% | 1.9 |
| kmeans | 9 | 9 | 4.44e-04 | 0.89 | 0.35 | 1.3% | 0.0% | 3.2 |
| kmeans | 10 | 10 | 7.65e-04 | 0.89 | 0.35 | 1.2% | 0.0% | 3.6 |
| spectral | 3 | 3 | 2.46e-04 | 0.48 | -0.09 | 8.1% | 0.0% | 1.2 |
| spectral | 4 | 4 | 2.14e-05 | 0.80 | 0.11 | 6.4% | 0.0% | 1.1 |
| spectral | 5 | 5 | 1.89e-04 | 0.64 | 0.14 | 6.3% | 0.0% | 1.2 |
| spectral | 6 | 6 | 6.03e-04 | 0.59 | 0.10 | 1.4% | 0.0% | 1.1 |
| spectral | 7 | 7 | 3.92e-04 | 0.63 | 0.13 | 1.4% | 0.0% | 1.1 |
| spectral | 8 | 8 | 7.01e-04 | 0.67 | 0.06 | 1.4% | 0.0% | 1.3 |
| spectral | 9 | 9 | 8.33e-04 | 0.67 | 0.09 | 1.4% | 0.0% | 1.3 |
| spectral | 10 | 10 | 9.83e-04 | 0.56 | 0.08 | 1.1% | 0.0% | 1.3 |

Chosen: **gmm, 4 clusters** (rule met).

## Clusters

| Cluster | Name (largest standardised features) | Action kind | Share of suspects | Failure rate |
|---|---|---|---|---|
| C0 | no clear damage | watch | 53.5% | 1.1% |
| C1 | reallocated sectors rising, active defect acceleration 7d vs 30d | replace | 14.9% | 4.7% |
| C2 | pending vs reallocated, pending sectors | migrate data | 22.1% | 11.7% |
| C3 | reallocated sectors rising, active defect velocity 30d | replace | 9.6% | 33.4% |

Band failure rates (validation): monitor 1.8%, warn 17.1%, cordon 55.4%, migrate -, drain 100.0%

## Tier x cluster matrix: Validation

| Cluster | Band | Drives | Failing | Failure rate | Median lead (days) | Removed | Active | Top family |
|---|---|---|---|---|---|---|---|---|
| C0 | monitor | 2,467 | 16 | 0.6% | 22 | 2.2% | 95.7% | Seagate HDD |
| C0 | warn | 10 | 3 | 30.0% | 28 | 0.0% | 60.0% | Toshiba HDD |
| C0 | cordon | 5 | 4 | 80.0% | 22 | 20.0% | 0.0% | Seagate HDD |
| C1 | monitor | 576 | 6 | 1.0% | 24 | 1.9% | 91.5% | Seagate HDD |
| C1 | warn | 38 | 2 | 5.3% | 14 | 13.2% | 65.8% | Seagate HDD |
| C1 | cordon | 37 | 23 | 62.2% | 12 | 2.7% | 29.7% | Seagate HDD |
| C2 | monitor | 695 | 35 | 5.0% | 30 | 3.0% | 88.2% | Western Digital HDD |
| C2 | warn | 247 | 35 | 14.2% | 25 | 5.7% | 70.9% | HGST HDD |
| C2 | cordon | 80 | 39 | 48.8% | 23 | 2.5% | 35.0% | HGST HDD |
| C3 | monitor | 193 | 12 | 6.2% | 30 | 4.1% | 83.9% | Seagate HDD |
| C3 | warn | 167 | 39 | 23.4% | 28 | 8.4% | 58.1% | Seagate HDD |
| C3 | cordon | 212 | 119 | 56.1% | 23 | 7.1% | 22.6% | Seagate HDD |
| C3 | drain | 2 | 2 | 100.0% | 24 | 0.0% | 0.0% | Seagate HDD |

## Tier x cluster matrix: Q3 test

| Cluster | Band | Drives | Failing | Failure rate | Median lead (days) | Removed | Active | Top family |
|---|---|---|---|---|---|---|---|---|
| C0 | monitor | 11,364 | 47 | 0.4% | 17 | 0.7% | 98.9% | Seagate HDD |
| C0 | warn | 69 | 2 | 2.9% | 12 | 8.7% | 88.4% | Toshiba HDD |
| C0 | cordon | 31 | 20 | 64.5% | 20 | 0.0% | 35.5% | Seagate HDD |
| C1 | monitor | 2,968 | 27 | 0.9% | 27 | 1.5% | 97.6% | Seagate HDD |
| C1 | warn | 207 | 14 | 6.8% | 20 | 4.3% | 88.9% | Seagate HDD |
| C1 | cordon | 120 | 44 | 36.7% | 15 | 6.7% | 56.7% | Seagate HDD |
| C1 | drain | 2 | 1 | 50.0% | 25 | 0.0% | 50.0% | Toshiba HDD |
| C2 | monitor | 3,660 | 84 | 2.3% | 26 | 1.9% | 95.8% | Western Digital HDD |
| C2 | warn | 1,399 | 83 | 5.9% | 17 | 4.4% | 89.5% | HGST HDD |
| C2 | cordon | 331 | 69 | 20.8% | 16 | 14.5% | 64.0% | Toshiba HDD |
| C3 | monitor | 1,473 | 52 | 3.5% | 30 | 2.2% | 94.2% | Seagate HDD |
| C3 | warn | 860 | 105 | 12.2% | 23 | 5.7% | 82.0% | Seagate HDD |
| C3 | cordon | 858 | 334 | 38.9% | 21 | 12.2% | 47.1% | Seagate HDD |
| C3 | drain | 4 | 1 | 25.0% | 4 | 25.0% | 50.0% | Toshiba HDD |

## Action map (fixed on validation)

| Cell | Drives | Failure rate | Effective band | Reason |
|---|---|---|---|---|
| C0 / monitor | 2,467 | 0.6% | monitor | matches band rate |
| C0 / warn | 10 | 30.0% | warn | fewer than 30 drives |
| C0 / cordon | 5 | 80.0% | cordon | fewer than 30 drives |
| C1 / monitor | 576 | 1.0% | monitor | matches band rate |
| C1 / warn | 38 | 5.3% | monitor (down) | fails less often than its band |
| C1 / cordon | 37 | 62.2% | cordon | matches band rate |
| C2 / monitor | 695 | 5.0% | monitor | matches band rate |
| C2 / warn | 247 | 14.2% | monitor (down) | fails less often than its band |
| C2 / cordon | 80 | 48.8% | warn (down) | fails less often than its band |
| C3 / monitor | 193 | 6.2% | monitor | matches band rate |
| C3 / warn | 167 | 23.4% | warn | matches band rate |
| C3 / cordon | 212 | 56.1% | cordon | matches band rate |
| C3 / drain | 2 | 100.0% | drain | fewer than 30 drives |

## Policy comparison: Validation, in sample (tiers counted cumulatively)

| Tier | Tier-only P / R / flagged | Cluster P / R / flagged | Precision gap [95%] | Recall gap [95%] | Moved down (failing) | Moved up (failing) | Cost tier-only / cluster |
|---|---|---|---|---|---|---|---|
| warn | 33.3% / 63.6% / 798 | 44.6% / 54.8% / 513 | +11.3% [+9.0%, +13.6%] | -8.9% [-11.4%, -6.3%] | 285 (37) | 0 (0) | 3,572 / 4,064 |
| cordon | 55.7% / 44.7% / 336 | 57.8% / 35.4% / 256 | +2.2% [-0.8%, +5.4%] | -9.3% [-12.3%, -6.5%] | 80 (39) | 0 (0) | 4,769 / 5,508 |
| migrate | 100.0% / 0.5% / 2 | 100.0% / 0.5% / 2 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | 0 (0) | 0 (0) | 8,320 / 8,320 |
| drain | 100.0% / 0.5% / 2 | 100.0% / 0.5% / 2 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | 0 (0) | 0 (0) | 8,320 / 8,320 |

## Policy comparison: Q3 test (tiers counted cumulatively)

| Tier | Tier-only P / R / flagged | Cluster P / R / flagged | Precision gap [95%] | Recall gap [95%] | Moved down (failing) | Moved up (failing) | Cost tier-only / cluster |
|---|---|---|---|---|---|---|---|
| warn | 17.3% / 60.9% / 3,881 | 25.3% / 52.1% / 2,275 | +8.0% [+7.1%, +8.9%] | -8.8% [-10.5%, -7.1%] | 1,606 (97) | 0 (0) | 11,848 / 12,279 |
| cordon | 34.8% / 42.4% / 1,346 | 39.4% / 36.2% / 1,015 | +4.6% [+3.1%, +5.9%] | -6.2% [-7.7%, -4.9%] | 331 (69) | 0 (0) | 13,597 / 14,715 |
| migrate | 33.3% / 0.2% / 6 | 33.3% / 0.2% / 6 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | 0 (0) | 0 (0) | 22,064 / 22,064 |
| drain | 33.3% / 0.2% / 6 | 33.3% / 0.2% / 6 | +0.0% [+0.0%, +0.0%] | +0.0% [+0.0%, +0.0%] | 0 (0) | 0 (0) | 22,064 / 22,064 |

## Stability and the rule for the agent integration

| Cluster | Validation share | Q3 share | Validation failure rate | Q3 failure rate |
|---|---|---|---|---|
| C0 | 53.5% | 49.6% | 1.1% | 0.7% |
| C1 | 14.9% | 14.3% | 4.7% | 2.7% |
| C2 | 22.1% | 23.6% | 11.7% | 4.5% |
| C3 | 9.6% | 12.4% | 33.4% | 16.1% |

Refit on Q3 against the assignment: ARI 0.87.

| Check | Result |
|---|---|
| warn_precision_up | pass |
| warn_recall_not_lower | fail |
| cordon_precision_up | pass |
| cordon_recall_not_lower | fail |
| shares_within_2x | pass |
| refit_ari_ge_0_5 | pass |
| no_small_cell_moved | pass |

**Rule not passed: the agent integration is not built.**
