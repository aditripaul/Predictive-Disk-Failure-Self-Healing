# Knowledge store checks and the knowledge-rule screen, 60-day horizon (frozen)

Phases 1 and 2 of `docs/agent_knowledge_base.md`. **Validation only: no test row was read
and the test-access log stayed at 139 lines.**

## The store (Phase 1)

Built by `data/audit/agent_knowledge_h60_logs/run_build.sh` (`scripts/build_agent_knowledge.py
build`, then `check`, `coverage`, `drift`) at commit `a96d8a3`, 2026-11-07 14:12-14:16 UTC
(26 weekly chunks, 193 s). Scores: the frozen stacked Stage 2's validation scores
(`data/tmp/stack_scores/scores_h60_seed0.parquet`, 2026-01-01..05-01, lift tiers warn 0.9342,
cordon 0.9825, primary 0.9961, candidate threshold 0.2716); presence and failures from silver
`drive_day` to 2026-06-30, for the 70,969 validation drives.

**The store itself is tracked compressed:** `agent_knowledge.sqlite.gz` (24.1 MB, a
deterministic gzip of the 100.5 MB SQLite file, which is over GitHub's file limit), with
`agent_knowledge_pack.json` recording the packed file's SHA-256 (`8e39368aa8d75a3c...`) and the
store's content fingerprint (`KnowledgeBase.content_hash`, `662ab2647de72126...`). Restore it
with `make unpack-knowledge` (to `data/audit/agent_knowledge.sqlite`; it checks the SHA-256
before and the fingerprint after); pack it again with `make pack-knowledge`. A store rebuilt with
`make agent-knowledge` has the same content fingerprint even where its file bytes differ.

| Table | Rows |
|---|---|
| candidate_days | 234,275 (= the frozen stack run's validation candidate rows) |
| presence_weeks | 1,864,602 |
| drive_info | 70,969 |
| failure_days | 483 |
| cohort_daily | 15,593 |
| outcomes (final only) | 234,034 |

**Self-checks (`knowledge_checks.json`): all pass.**

| Check | Result |
|---|---|
| a, incremental equals batch | 2026-02-01..14 rebuilt in memory forward and in reverse: 23,532 candidate days, 37 failures, 1,204 cohort rows identical |
| b, as-of replay | 500 fixed drives' contexts recomputed at 26 chunk ends equal the hashes recorded while building |
| d, outcomes vs gold labels | 234,034 resolved rows, 0 disagreements (221,247 zeros, 12,787 ones); 241 unresolved, all gold 0, none a gold failure |
| f, idempotence | re-ingesting 2026-05-01 changes nothing |
| h, score agreement | the stack's model and score table agree on 2026-02-23 (from the point-in-time run) |

Episodes as of 2026-06-30: 1,419 on 797 drives (median 7 flagged days, 90th percentile 39);
431 failed, 979 survived their horizon, 2 censored, 7 pending. Of the 222 episodes with 7 or
more days above cordon, 138 failed and 83 survived.

Coverage (`coverage_*.json`): every validation drive scored on 2026-03-02 (67,732) and on
2026-05-01 (70,066) has history; 66,581 and 67,564 of them 60 days or more.

Drift (`drift_2026-04-01_2026-05-01.json`, reference 2026-01-01..03-31): the daily shares of
drives at Stage 1, warn and cordon rose from 2.71% / 0.24% / 0.057% to 3.17% / 0.31% / 0.092%;
PSI flags a significant shift on all three (`retrain_recommended: true`) and a moderate one
on the candidate scores (0.23). PSI on 31 daily values against 90 is coarse; the direction
matches the rising Stage 1 pass share runbook 1.1.5 saw in Q3.

## The screen (Phase 2)

`data/audit/agent_knowledge_h60_logs/run_screen.sh` (`scripts/screen_agent_knowledge.py`) at
commit `4842ea3` (clean tree; an identical first run at `941a78a` recorded a dirty tree because
of a whitespace-only docstring change, since reverted). Rule: section 8 of the design
document with Amendment 1. Confirm half 2026-02-16..03-02: 1,013,502 drive-days, 68,064 drives,
row failure rate 0.223%.

| Rule | Verdict |
|---|---|
| R1 cohort prior (tau 2.0 chosen) | Failed: warn precision -0.2 [-2.2, +1.8], cordon precision +1.9 [-2.5, +6.3]; not better than a plain threshold at matched false-alarm rate |
| R2 chronic demote | Failed at selection: no value beat "no rule" on the select half |
| R2b first-flag promote | Failed at selection: no value beat "no rule" on the select half |
| R3 outcome memory | Not screenable at the 60-day horizon (declared in advance) |

**No rule passed, so there is no Q3 look.** `tables.md` and `validation.json` hold the details.

**Descriptive only (`tier_rules_descriptive.json`, `--describe-tier-rules`, commit `d58e0cc`):**
the select-half criterion compares rankings and cannot see a pure tier rule (a demoted drive
stays above the rest of its new tier). At the fixed cordon threshold on the confirm half, R2
lowers recall significantly at every N (N = 7: -10.2 points [-14.1, -6.3]; N = 28: -4.1
[-7.0, -1.7]) and does not raise precision: more than half the demoted drive-days belong to
drives that failed. R2b changes 2 drive-days. This changes no verdict.

`CHECKSUMS.sha256` covers every file here, the packed store included; `sha256sum -c
CHECKSUMS.sha256` from this folder. The logs print absolute paths and stay local.
