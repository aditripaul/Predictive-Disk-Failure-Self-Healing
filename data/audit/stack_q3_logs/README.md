# Stacked Stage 2 run (2026-11-06, frozen)

Logs and launcher of the LightGBM + Cox stacked Stage 2 experiment on the Q1+Q2 -> Q3
runs. Method, results and verdict: `docs/model_status_and_runbook.md` Section 1.1.4a.

| File | What it is |
|---|---|
| `run_stack.sh` | Launcher; takes the repository root as its first argument. Runs `make experiment-model ... --two-stage --stage2-kinds lightgbm cox stack` for 30, 44 and 60 days. |
| `stack_h30.log`, `stack_h44.log`, `stack_h60.log` | One log per horizon. One MLflow hint line that contained a local absolute path was deleted from each before committing; nothing else was changed. |
| `STACK_DONE` | Written when the last horizon finished. |
| `CHECKSUMS.sha256` | SHA-256 of each horizon's `experiment_two_stage.json`, `tables.md`, `code.patch` and of `run_stack.sh`. |

Results are in `data/audit/stack_q3_h30/`, `stack_q3_h44/` and `stack_q3_h60/`.

The outputs are frozen: do not edit them, and do not re-run into these folders (the runner
refuses to overwrite). To check nothing changed, from the repository root:

    sha256sum -c data/audit/stack_q3_logs/CHECKSUMS.sha256
