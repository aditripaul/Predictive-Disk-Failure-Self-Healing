# Knowledge-rule screen (validation)

Screen set 2026-02-01..2026-03-02, select half to 2026-02-15, confirm half from 2026-02-16; scorer `frozen_runs/stack_h60_seed0/agent_thresholds_stack_h60_seed0.json`, tier set `action_tiers`. Commit `4842ea35b571650c2e8bf7ce44d14e07828fef88`.

Confirm half: 1,013,502 drive-days, 68,064 drives, row failure rate 0.223%.

| Rule | Chosen | Judged at | Changed drive-days (drives) | Precision diff [95% CI] | Recall diff [95% CI] | C1 | C2 | C3 | C4 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| r1_cohort_prior | {"tau": 2.0} | warn, cordon | 748 (201) | warn: -0.2 [-2.2, +1.8]; cordon: +1.9 [-2.5, +6.3] | warn: +0.5 [-1.9, +2.9]; cordon: +1.9 [+0.3, +3.7] | no | no | yes | yes | failed |
| r2_chronic_demote | none | - | - | - | - | - | - | - | no | failed: no value beat "no rule" on the select half |
| r2b_first_flag_promote | none | - | - | - | - | - | - | - | no | failed: no value beat "no rule" on the select half |
| r3_outcome_memory | - | - | - | - | - | - | - | - | - | not screenable: no validation date has an episode resolved 60 days after it closed (section 7.1) |

C1-C4 are the criteria of docs/agent_knowledge_base.md section 8 (Amendment 1 applied);
points are precision or recall points, rule minus "no rule", per drive-day.
