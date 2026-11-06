# Figures of the Q1+Q2 2026 -> Q3 2026 runs

Copies of the figures in the committed result folders
(`data/audit/cascade_q3_h<H>/figures/`, `data/audit/horizon_q3/figures/`), placed
here so the runbook (`docs/model_status_and_runbook.md` section 1.1.4) and
`docs/q3_horizon_and_cascade_run.md` can embed them next to the text. All are drive-level, on the
Q3 2026 test set, drawn by `src/reporting/evaluation_figures.py` (curves thinned
to about 2,000 points before drawing; marker values are exact).

| File | Source | What |
|---|---|---|
| `cascade_h<H>_precision_recall.png`, `cascade_h<H>_confusion.png` | `data/audit/cascade_q3_h<H>/figures/cascade_*_seed0.png` (MLflow `f760b06d` / `a4f74561` / `cabc3023`, artifact `figures/`) | Stage 1 alone against each net and Stage 2 kind, **seed 0 only** (the tables average three seeds); confusion matrices at the 10% and 60% recall levels |
| `stack_topk_caught.png` | `scripts/plot_stack_topk.py` from the saved `top_k` results in `data/audit/stack_q3_h<H>/experiment_two_stage.json` (3 seeds) | Failing drives caught by Cox and the stack minus the LightGBM Stage 2 when exactly K drives are flagged; runbook section 1.1.4a. Unlike the seed-0 curves it resolves the strict end |
| `stack_h<H>_precision_recall.png`, `stack_h<H>_confusion.png` | `data/audit/stack_q3_h<H>/figures/cascade_*_seed0.png` (MLflow `5ab8f0bb` / `7a568b77` / `47d8ff0c`, tracked, checksummed in `data/audit/stack_q3_logs/CHECKSUMS.sha256`) | The stacked Stage 2 (LightGBM + Cox) with Stage 1 alone, the 0.5 and 0.8 LightGBM nets and Cox, **seed 0 only**; runbook section 1.1.4a |
| `horizon_labels_<H>d_precision_recall.png`, `horizon_labels_<H>d_confusion.png` | `data/audit/horizon_q3/figures/labels_<H>d_*.png` | The 30-, 44- and 60-day models scored against the <H>-day labels on the same drives (2026-07-01 .. 08-01); confusion matrices at the 0.1% false-alarm rate |

The per-run training figures (train, validation, test) are in
`frozen_runs/<run id>/figures/`.
