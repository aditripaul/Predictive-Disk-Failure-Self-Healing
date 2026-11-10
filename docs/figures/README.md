# Figures embedded in the docs

Every image that `docs/*.md` embeds lives under `docs/`, so editor previews and any viewer rooted at
`docs/` can load it. Figures drawn elsewhere are copied here; re-copy one when its source is redrawn.

| Folder or file | Source |
|---|---|
| `chrono_split/` | The chronological run (runbook 1.1.9); its README maps each figure to its source |
| `q3_2026/` | The Q1+Q2 2026 -> Q3 2026 runs, frozen-run curves and the cluster policy; its README maps each figure |
| `late_hits_*.png` | `make plot-late-hits` (`scripts/plot_late_hits.py`, written to `data/audit/plots/`) |
| `sequence_*.png` | `scripts/plot_sequence_comparison.py` (written to `data/audit/plots/`) |
| `class_imbalance.png`, `metric_comparison.png`, `precision_at_k_test.png` | `pipelines/generate_performance_plots.py`, copied from `data/audit/plots/` (runbook 4.6) |
| `*_quarter_aligned.png`, `failures_by_family.png` | The quarter-aligned run of 2026-10-18 (runbook 4.6, report 10.6) |
