# Frozen merged-approach run fb58d9576c6146ba94c13389005fa15b (`h30_seed0`)

The result of `make train-merged MERGED_TAG=h30_seed0`: AFT and Cox survival
models plus rank blends, compared with the LightGBM two-stage, LSTM and TCN
runs of the same tag on identical validation and test rows. Why it is frozen,
its results and caveats: `docs/model_status_and_runbook.md` Section 1.1.2.

## Committed here

| Path | What |
|---|---|
| `merged_evaluation_report.json` | Every model's and blend's drive-level results, thresholds, action tiers, recall at matched false-alarm rates, row coverage, split dates |
| `agent_thresholds.json` | Score column (`lgbm_lstm_aft_tcn`) and action tiers the agent acts on |
| `aft.json`, `cox.json` | The trained XGBoost AFT (138 trees) and Cox (147 trees) boosters |
| `weights/lstm.pt`, `weights/tcn.pt` | The sequence networks' weights (best-validation epoch) |
| `SHA256SUMS` | Checksums of the files above: `cd` here and run `sha256sum -c SHA256SUMS` |

The LightGBM two-stage member is the frozen run in `frozen_runs/839d92bcd305494a8930d0dfbbb3f2a6/`.

## Not committed (over GitHub's 100 MB file limit), pinned by checksum

`not_committed.sha256` lists SHA-256 for the four score files this run read or
wrote: the 1 GB blended score table, the LightGBM scores, and the LSTM and TCN
score archives. `<scratch>` is the repository's `data/tmp/`. A regenerated file
that does not match is a different run.

## Blend definition

Equal-weight mean of each member's rank against its own *validation* score
distribution (`src/models/ensemble.py`): `lgbm_aft_tcn` = LightGBM, AFT, TCN;
`lgbm_lstm_aft_tcn` = LightGBM, LSTM, AFT, TCN. Seed 0; AFT and Cox unweighted.

## What this folder cannot do

- Scoring new drives with a blend needs each member's validation scores as the
  rank reference. They are inside the uncommitted score table, so the blend
  cannot be rebuilt from this folder alone.
- No script loads from this folder. `score-fleet` and `evaluate-frozen` read
  MLflow, and this run's MLflow store is local.
- The run's test-access-log line records git commit `f50db78`, apparently
  because the checkout was switched to `main` while the run was in progress.
  The run was started on the branch at `e465403` (`f50db78` plus a startup
  input check that changes no computation).
