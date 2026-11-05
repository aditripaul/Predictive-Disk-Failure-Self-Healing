# Frozen run 839d92bcd305494a8930d0dfbbb3f2a6 (`h30_seed0`)

The files MLflow holds for the frozen quarter-aligned run, copied here so the
run can be inspected and checked without the machine that trained it. Why it is
frozen, its results, and the rule not to retrain it: `docs/model_status_and_runbook.md`
Section 1.1.1.

| Path | What |
|---|---|
| `frozen_spec.json` | Drive threshold, the 246 feature columns, the action-tier cut-offs, the calibrator |
| `action_tiers.json` | warn / cordon / migrate / drain cut-offs |
| `two_stage.json` | Second-stage settings and the stage-1 precision-at-recall table |
| `model_card.json`, `model_card.md` | The model card written by `make train` |
| `stage1/` | Stage-1 LightGBM classifier (204 trees, 246 features), MLflow `lightgbm` flavor, skops format |
| `stage2/` | Stage-2 LightGBM classifier (46 trees, 247 features: stage 1's score is the extra input) |
| `SHA256SUMS` | Checksums of every other file here; `sha256sum -c SHA256SUMS` |

Not included: the 264 MB score table `lightgbm_scores_h30_h30_seed0.parquet`
(its checksum is in the runbook), and each model's `conda.yaml` (`MLmodel`
still names it; the pinned `requirements.txt` and `python_env.yaml` are here).

Trained with mlflow 3.16.1, lightgbm 4.7.0, Python 3.14.8.

Load a stage, with the matching library versions installed:

```python
import mlflow
stage1 = mlflow.lightgbm.load_model("frozen_runs/839d92bcd305494a8930d0dfbbb3f2a6/stage1")
```

`score-fleet` and `evaluate-frozen` still load by MLflow run id; they do not
read this folder.
