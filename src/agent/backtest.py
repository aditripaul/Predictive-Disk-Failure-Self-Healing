"""Scoring what the agent actually did, against what happened.

The model's test figures - 57.1% precision at 12.0% recall - describe a
threshold applied to scores. They say nothing about the agent, because
between a score and an action sit the guardrails, the 0.80 confidence
floor, two-cycle hysteresis, the per-drive cooldown, the control arm and
one-action-per-cycle. Every one of those can only *remove* actions, and
nobody has measured by how much.

This module answers two questions from a run of
`pipelines/run_agent_fleet.py` over a labelled window:

  * **Agent precision and recall** - of the drives it actually cordoned,
    migrated or drained, how many failed; and of the failing drives in the
    window, how many it reached. Not the model's numbers: the agent's.
  * **Where the attrition happened** - how many candidates were lost to
    each filter, so "the agent catches less than the model" becomes a
    breakdown rather than a single disappointing number.

Two things it deliberately does not do. It does not score the control arm
as failures of the agent: those drives were withheld on purpose, and their
outcomes are the uncensored labels the whole arrangement exists to protect.
And it does not treat a drive the agent never reached as a miss caused by
the policy, because one action per cycle means `--cycles` bounds reach
independently of any decision rule - that is counted separately as
`never_selected`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from data_contracts.schemas import ActionTier

#: Tiers that remove a drive from service, i.e. the agent acted.
ACTIVE_ACTIONS = {ActionTier.CORDON.value, ActionTier.MIGRATE.value, ActionTier.DRAIN.value}


def load_labels(
    labels_path: Path, *, horizon_days: int, start_date: dt.date, end_date: dt.date
) -> pl.DataFrame:
    """Observed (non-censored) labels for the horizon, inside the window.

    Censored rows are dropped rather than read as negatives - an outcome
    that cannot be observed is not evidence that nothing happened
    (`src/labels/labeling.py`)."""
    return (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("horizon_days") == horizon_days)
            & (pl.col("date") >= start_date)
            & (pl.col("date") <= end_date)
            & pl.col("label").is_not_null()
        )
        .select(["drive_id", "date", "label"])
        .collect()
    )


def failing_drives(labels: pl.DataFrame) -> set[str]:
    """Drives with at least one positive row in the window - the drive-level
    convention the model metrics use (`src/models/evaluation.py`)."""
    return set(labels.filter(pl.col("label") == 1)["drive_id"].to_list())


def score_decisions(
    decisions: list[dict[str, Any]],
    labels: pl.DataFrame,
    *,
    candidate_count: int,
    above_threshold_count: int | None = None,
) -> dict[str, Any]:
    """Agent-level precision and recall, plus the attrition breakdown.

    `decisions` are the records `pipelines/run_agent_fleet.py` writes, one
    per cycle. A drive decided more than once counts once, at the strongest
    action it received.

    `above_threshold_count` is how many drives cleared the lowest threshold
    before `--max-candidates` truncated the pool. Without it a capped run
    reports every drive as reached and the attrition reads as zero, which
    looks like the filters rejecting nothing rather than the agent never
    having been shown the rest."""
    failing = failing_drives(labels)
    observed = set(labels["drive_id"].to_list())

    strongest: dict[str, dict[str, Any]] = {}
    order = [
        ActionTier.MONITOR.value,
        ActionTier.WARN.value,
        ActionTier.CORDON.value,
        ActionTier.MIGRATE.value,
        ActionTier.DRAIN.value,
    ]
    rank = {tier: i for i, tier in enumerate(order)}
    for decision in decisions:
        drive_id = decision["drive_id"]
        current = strongest.get(drive_id)
        if current is None or rank.get(decision["proposed_action"], 0) > rank.get(
            current["proposed_action"], 0
        ):
            strongest[drive_id] = decision

    acted = {d for d, rec in strongest.items() if rec["proposed_action"] in ACTIVE_ACTIONS}
    control = {d for d, rec in strongest.items() if rec.get("control_arm")}
    review = {d for d, rec in strongest.items() if rec.get("human_review_required")}
    # Only drives whose outcome is observable can be scored either way.
    acted_observed = acted & observed
    caught = acted_observed & failing

    precision = len(caught) / len(acted_observed) if acted_observed else 0.0
    recall = len(caught) / len(failing) if failing else 0.0

    eligible = above_threshold_count if above_threshold_count is not None else candidate_count
    return {
        "failing_drives_in_window": len(failing),
        "drives_above_threshold": eligible,
        # Capped away before the agent could see them: not a policy decision,
        # and the dominant term whenever --max-candidates binds.
        "truncated_by_max_candidates": max(eligible - candidate_count, 0),
        "recall_ceiling_from_pool": (
            min(candidate_count, len(failing)) / len(failing) if failing else None
        ),
        "drives_with_observed_labels": len(observed),
        "candidates_in_snapshot": candidate_count,
        "drives_decided": len(strongest),
        "drives_acted_on": len(acted),
        "drives_acted_on_with_labels": len(acted_observed),
        "caught": len(caught),
        "agent_precision": precision,
        "agent_recall": recall,
        "attrition": {
            # Each count is drives lost before an action, by cause. They do
            # not sum to a single total: a drive can be withheld AND have
            # been sent to review in different cycles.
            "never_selected": max(candidate_count - len(strongest), 0),
            "withheld_by_control_arm": len(control),
            "sent_to_human_review": len(review),
            "decided_but_not_acted": len(strongest) - len(acted),
        },
        "actions_by_tier": {
            tier: sum(1 for rec in strongest.values() if rec["proposed_action"] == tier)
            for tier in order
        },
    }
