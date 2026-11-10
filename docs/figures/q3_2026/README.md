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
| `cascade_funnel_h30.png` | `scripts/plot_cascade_funnel.py`, from the published counts of runbook sections 1.1.4, 1.1.5 and 1.9 (no data read, no test look) | The two-stage funnel of the 30-day model on Q3: failing drives reached, healthy drives flagged and precision at the Stage 1 net and each threshold; project report Section 8.4 |
| `model_h<H>_<split>_precision_recall.png`, `model_h<H>_<split>_confusion.png` (`<split>` train, validation, test) | `frozen_runs/<run id>/figures/<split>_*.png`: h30 `c459046e4f1645169151e7be25ff064d`, h44 `a2cd06d53e624c09b0d12f81076def8a`, h60 `804a20cec10048abb1bc3ccf510f96e1` | The 30-, 44- and 60-day drive-holdout runs (Q1+Q2 2026 -> Q3 2026) on each split; the h30 test pair is the previous model of project report Section 9.2; runbook sections 1.1.1 and 1.1.4 |
| `deep_dive_<name>.png` | `data/audit/deep_dive_q3_h30/figures/<name>.png` (`scripts/deep_dive_q3.py`) | Missed failures, false-alarm outcomes, weekly stability, validation against test; runbook section 1.1.5, project report Sections 9.4, 9.6 to 9.8 |
| `cluster_<name>.png` | `data/audit/cluster_policy_h30/figures/<name>.png` (`scripts/cluster_action_policy.py`; a source name that already starts with `cluster_` is not prefixed again, and `policy_*` lost its prefix: `cluster_policy_comparison.png`) | Tier thresholds and lift targets, cluster signatures, families, stability, the (tier, cluster) matrices, lead time and outcomes by cell, the clustering ablation and the policy's Q3 comparison; runbook sections 1.1.6 and 1.9, project report Sections 9.3 and 9.9 |

The `model_`, `deep_dive_` and `cluster_` copies exist so `docs/report.md` and
`docs/model_status_and_runbook.md` embed only files under `docs/`, which editor previews will load
(a preview may refuse images outside the document's folder or workspace). Re-copy a figure here when its source is redrawn.

The per-run training figures (train, validation, test) are in
`frozen_runs/<run id>/figures/`.
