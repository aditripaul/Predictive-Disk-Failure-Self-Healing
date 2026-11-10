# Figures of the chronological split run (2026-11-10)

Embedded in `docs/model_status_and_runbook.md` Sections 1.1.8 and 1.1.9. The split is train
2026-01-01 .. 04-20, validation 05-21 .. 06-16, test 07-17 .. 08-31, with 30-day gaps; all figures are
drive level, 30-day horizon.

| File | Source | What |
|---|---|---|
| `chrono_split_timeline.png`, `chrono_precision_recall.png`, `chrono_goal_validation_vs_test.png`, `chrono_base_rate_transfer.png`, `chrono_true_positives_isolated.png`, `chrono_calibration.png` | `scripts/plot_chrono_split.py`, from `data/audit/stack_chrono_h30/` and `data/audit/data_quality_reports/model_evaluation_report_chrono_h30.json` (no model is scored and no test row is read; `make plot-chrono-split` redraws them) | The chronological split (1.1.9) on a date axis; precision against recall of Stage 1 and the three Stage 2 variants with the goal's region; precision at the 10% level on validation and test with each seed; base rates and precision of both designs; failing test drives isolated (count and share) by the Stage 1 net, Stage 1 alone, the LightGBM and Cox Stage 2 and the stack at the 10, 20 and 35% recall levels; reliability of the raw and the isotonic-calibrated score per drive-day (runbook 1.3) |
| `model_h30_chrono_{train,test,validation}_{precision_recall,confusion}.png` | `data/audit/plots/train/chrono_h30/` (`make train`, tag `chrono_h30`, MLflow `6ea7a5bcfb954deeb4a4208e4d802744`, frozen in `frozen_runs/`) | The LightGBM two-stage model (Stage 1 + LightGBM Stage 2): the drive-level precision-recall curve (solid) and the drive-day curve (dashed) with the primary threshold, the action tiers and the top-K points marked; confusion matrices at the primary threshold and each tier |
| `stack_chrono_precision_recall.png`, `stack_chrono_confusion.png` | `data/audit/stack_chrono_h30/figures/cascade_*_seed0.png` (MLflow `08aedf53015243a3a12b467468421163`) | Stage 1 alone against the 0.5 and 0.8 LightGBM nets, Cox and the stack, **seed 0 only** (the tables average three seeds); confusion matrices at the 10% and 60% recall levels |

Colours are the reference palette of the dataviz skill: train blue, validation orange, test aqua in every
figure, and the same three hues for the three Stage 2 kinds where they are never on one chart with the
roles. Aqua is below 3:1 contrast on the surface, so every aqua mark is also labelled or tabled.
