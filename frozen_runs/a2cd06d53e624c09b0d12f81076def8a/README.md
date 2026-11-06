# Frozen run a2cd06d53e624c09b0d12f81076def8a (`h44_q1q2_q3`)

The files MLflow holds for the 44-day run of the Q1+Q2 -> Q3 window
(`make train TRAIN_TAG=h44_q1q2_q3 TRAIN_ARGS="--horizon-days 44 --keep-work-dir --save-scores"`,
2026-11-06), copied here so the run can be inspected and checked without the
machine that trained it. Frozen 2026-11-06 after its Q3 test results were
recorded. Why it is frozen, its results and the rule not to retrain it:
`docs/model_status_and_runbook.md` Section 1.1.4 and
`docs/q3_horizon_and_cascade_run.md`.

Window: train and a 20% drive-holdout validation over 2026-01-01..05-01, test
Q3 2026 from 07-01 to 2026-08-17 (label table `data/audit/dataset_versions/20261106T091944670108Z.json`).

| Path | What |
|---|---|
| `frozen_spec.json` | Drive threshold, the 246 feature columns, the action-tier cut-offs, the calibrator, the Stage 1 flag rate |
| `action_tiers.json` | warn / cordon / migrate / drain cut-offs |
| `two_stage.json` | Second-stage settings (0.8 net), candidate counts and the stage-1 precision-at-recall table |
| `model_card.json`, `model_card.md` | The model card written by `make train` |
| `figures/` | Drive precision-recall curves and confusion matrices for train (in-sample), validation and test |
| `stage1/` | Stage-1 LightGBM classifier (225 trees, 246 features), MLflow `lightgbm` flavor, skops format |
| `stage2/` | Stage-2 LightGBM classifier (201 trees, 247 features: stage 1's score is the extra input) |
| `SHA256SUMS` | Checksums of every other file here; `sha256sum -c SHA256SUMS` |

Not included: the score table `lightgbm_scores_h44_h44_q1q2_q3.parquet`
(validation and test scores; SHA-256 `e1d74c527e00a838b6a1bd153f527c6b4c1d62d91c43e20d597af6891b355681`),
and each model's `conda.yaml` (`MLmodel` still names it; the pinned
`requirements.txt` and `python_env.yaml` are here).

Trained with mlflow 3.16.1, lightgbm 4.7.0, Python 3.14.8.

Load a stage, with the matching library versions installed:

```python
import mlflow
stage1 = mlflow.lightgbm.load_model("frozen_runs/a2cd06d53e624c09b0d12f81076def8a/stage1")
```

`score-fleet` and `evaluate-frozen` still load by MLflow run id; they do not
read this folder.
