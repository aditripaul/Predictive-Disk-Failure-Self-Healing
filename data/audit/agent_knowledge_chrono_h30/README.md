# The agent's knowledge store on the chronological validation window, 30-day horizon

Runbook Section 1.1.9, "The agent's evidence on this split". Written by `make chrono-knowledge` (launcher
`data/audit/chrono_logs/run_agent_evidence.sh`, log `data/audit/chrono_logs/knowledge.log`), 2026-11-10 19:02 to
19:10 IST, code commit `553770b` plus the uncommitted Makefile targets and builder guard of that day.

Built from the seed-0 stack scorer's validation scores (`data/tmp/stack_scores/scores_chrono_h30_seed0.parquet`,
candidate days 2026-05-21..06-16, tiers warn 0.9141, cordon 0.9651, primary 0.9979, candidate threshold 0.1340);
presence and failures from silver `drive_day` to 2026-07-16, the day before the test starts, so every 30-day outcome
resolves. No test row is read (the test-access log stayed at 146 entries). The store holds 27 days of history.

**The store is tracked compressed:** `agent_knowledge.sqlite.gz` (42.2 MB, a deterministic gzip of the 193.7 MB
SQLite file, which is over GitHub's file limit), with `agent_knowledge_pack.json` recording the packed file's SHA-256
(`37e70847528784dc...`) and the store's content fingerprint (`f7b37c5fbb29f8e9...`). Restore it with
`make chrono-unpack-knowledge` (to `data/audit/agent_knowledge_chrono_h30.sqlite`; it checks the SHA-256 before and
the fingerprint after); pack it again with `make chrono-pack-knowledge`. A store rebuilt with `make chrono-knowledge`
has the same content fingerprint even where its file bytes differ.

| File | What |
|---|---|
| `build.json` | Row counts, per-chunk timings, the scorer's settings |
| `knowledge_checks.json` | Five self-checks, all pass, and the episode summaries |
| `coverage_2026-06-16.json`, `drift_2026-06-04_2026-06-16.json` | Coverage and drift (the drift flag is an artifact of 13- and 14-day windows) |
| `agent_knowledge.sqlite.gz`, `agent_knowledge_pack.json` | The packed store and its hashes |

Reproduce (run `make chrono-pit-validation` first: its cross-check is one of the checks):

```bash
make chrono-knowledge               # about 8 minutes: build, check, coverage, drift
make chrono-pack-knowledge          # optional: the gzip and its hashes
make chrono-unpack-knowledge        # restore it instead of rebuilding (ARGS=--overwrite to replace)
```

`CHECKSUMS.sha256` covers every tracked file here; verify with `sha256sum -c CHECKSUMS.sha256` from this folder.
