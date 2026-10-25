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


#: Ascending, so "strongest action a drive received" is well defined.
TIER_ORDER = [
    ActionTier.MONITOR.value,
    ActionTier.WARN.value,
    ActionTier.CORDON.value,
    ActionTier.MIGRATE.value,
    ActionTier.DRAIN.value,
]


def strongest_per_drive(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One decision per drive: the strongest action it received.

    Hysteresis warns a drive before escalating it, so most drives appear
    twice. Counting both would double them and understate precision."""
    rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    strongest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        drive_id = decision["drive_id"]
        current = strongest.get(drive_id)
        if current is None or rank.get(decision["proposed_action"], 0) > rank.get(
            current["proposed_action"], 0
        ):
            strongest[drive_id] = decision
    return strongest


def was_downgraded_to_cordon(decision: dict[str, Any], action_thresholds: dict[str, float]) -> bool:
    """Whether a cordon is a *downgraded* migrate/drain.

    `determine_action_tier` returns CORDON when a drive scores high enough
    for migrate or drain but fails the confidence floor, the maturity gate
    or the staleness check - "insufficient evidence to justify a destructive
    action", which still acts, just more gently. Hysteresis and cooldown cap
    at WARN rather than CORDON, so a cordon whose p_fail clears the migrate
    threshold can only have come from that downgrade.

    Separating these matters: they are the population where the score is
    least trustworthy, so if their precision is materially worse than a
    genuine cordon's, the gate should downgrade to WARN instead of acting."""
    if decision["proposed_action"] != ActionTier.CORDON.value:
        return False
    migrate_threshold = action_thresholds.get(ActionTier.MIGRATE.value)
    return migrate_threshold is not None and decision["p_fail"] >= migrate_threshold


def _bucket(
    label: str, drives: list[dict[str, Any]], failing: set[str], observed: set[str]
) -> dict[str, Any]:
    with_labels = [d for d in drives if d["drive_id"] in observed]
    caught = [d for d in with_labels if d["drive_id"] in failing]
    return {
        "tier": label,
        "drives": len(drives),
        "with_labels": len(with_labels),
        "caught": len(caught),
        "precision": len(caught) / len(with_labels) if with_labels else None,
    }


def precision_by_tier(
    decisions: list[dict[str, Any]],
    labels: pl.DataFrame,
    *,
    action_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Precision of each tier the agent assigned, strongest action per drive.

    94% of a real run's actions were `cordon`, the cheapest and only
    reversible active tier. An overall precision figure is therefore
    dominated by it, and says little about whether the expensive tiers are
    trustworthy - which is the question that decides whether a false-positive
    rate is a problem or a price.

    With `action_thresholds`, cordons are split into genuine ones and
    migrate/drain downgraded by the evidence gates."""
    failing = failing_drives(labels)
    observed = set(labels["drive_id"].to_list())
    strongest = strongest_per_drive(decisions)

    rows: list[dict[str, Any]] = []
    for tier in TIER_ORDER:
        at_tier = [d for d in strongest.values() if d["proposed_action"] == tier]
        if not at_tier:
            continue
        if tier == ActionTier.CORDON.value and action_thresholds:
            genuine = [d for d in at_tier if not was_downgraded_to_cordon(d, action_thresholds)]
            downgraded = [d for d in at_tier if was_downgraded_to_cordon(d, action_thresholds)]
            rows.append(_bucket("cordon (scored)", genuine, failing, observed))
            if downgraded:
                rows.append(_bucket("cordon (downgraded)", downgraded, failing, observed))
        else:
            rows.append(_bucket(tier, at_tier, failing, observed))
    return rows


def score_decisions(
    decisions: list[dict[str, Any]],
    labels: pl.DataFrame,
    *,
    candidate_count: int,
    above_threshold_count: int | None = None,
    action_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Agent-level precision and recall, plus the attrition breakdown.

    `decisions` are the records `pipelines/run_agent_fleet.py` writes, one
    per cycle. A drive decided more than once counts once, at the strongest
    action it received.

    `action_thresholds` adds the per-tier precision breakdown, which is
    where the useful detail is: an overall figure is dominated by `cordon`.

    `above_threshold_count` is how many drives cleared the lowest threshold
    before `--max-candidates` truncated the pool. Without it a capped run
    reports every drive as reached and the attrition reads as zero, which
    looks like the filters rejecting nothing rather than the agent never
    having been shown the rest."""
    failing = failing_drives(labels)
    observed = set(labels["drive_id"].to_list())

    strongest = strongest_per_drive(decisions)

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
            for tier in TIER_ORDER
        },
        "by_tier": precision_by_tier(decisions, labels, action_thresholds=action_thresholds),
    }
