"""Self-checks of a built knowledge store on real data (docs/agent_knowledge_base.md
section 5.5). Each returns a dict with `passed`; `scripts/build_agent_knowledge.py check`
writes them to `knowledge_checks.json`. Checks (a), (b), (c), (f) and (j) also have
synthetic unit tests; (i) is an integration test; (h) is the point-in-time cross-check;
(k) is a regression test of the loop."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from src.agent.knowledge.build import (
    context_hash,
    presence_chunk,
    scope_drive_ids,
    score_chunk,
)
from src.agent.knowledge.store import KnowledgeBase


def _source(kb: KnowledgeBase) -> tuple[Path, str | None, str]:
    meta = kb.metadata()
    return Path(meta["source_scores"]), meta.get("source_split"), meta.get("source_column", "score")


def check_incremental_equals_batch(
    kb: KnowledgeBase, *, drive_day_path: Path, start: dt.date, days: int = 14
) -> dict[str, Any]:
    """(a) Rebuilds `days` days from the sources in memory, in one pass and in reverse
    order, and compares their candidate days, failures and cohort counts with the store's."""
    scores_path, split, column = _source(kb)
    end = start + dt.timedelta(days=days - 1)
    drive_ids = scope_drive_ids(scores_path, split=split)
    scores = score_chunk(scores_path, start, end, split=split, column=column)
    presence = presence_chunk(drive_day_path, start, end, drive_ids=drive_ids)
    results = {}
    for name, order in (("forward", 1), ("reverse", -1)):
        fresh = KnowledgeBase.in_memory(
            run_id=kb.run_id,
            thresholds=kb.thresholds,
            candidate_threshold=kb.candidate_threshold,
            horizon_days=kb.horizon_days,
            episode_gap_days=kb.episode_gap_days,
            context_window_days=kb.context_window_days,
        )
        day_list = [start + dt.timedelta(days=i) for i in range(days)][::order]
        for day in day_list:
            p = presence.filter(pl.col("date") == day)
            fresh.ingest_snapshot(
                day,
                scores.filter(pl.col("date") == day).select("drive_id", "p_fail"),
                source="score_table",
                presence=p.select("drive_id", "drive_model", "model_family"),
                failures=p.filter(pl.col("failure") == 1).select("drive_id"),
            )
        results[name] = fresh
    window = (start.isoformat(), end.isoformat())

    def rows(store: KnowledgeBase, sql: str) -> list[tuple]:
        return sorted(store.db.execute(sql, window).fetchall())

    queries = {
        "candidate_days": (
            "SELECT drive_id, date, p_fail FROM candidate_days WHERE date BETWEEN ? AND ?"
        ),
        "failure_days": (
            "SELECT drive_id, failure_date FROM failure_days WHERE failure_date BETWEEN ? AND ?"
        ),
        "cohort_daily": "SELECT * FROM cohort_daily WHERE date BETWEEN ? AND ?",
    }
    comparisons = {}
    for table, sql in queries.items():
        stored = rows(kb, sql)
        comparisons[table] = {
            "rows": len(stored),
            "forward_equal": rows(results["forward"], sql) == stored,
            "reverse_equal": rows(results["reverse"], sql) == stored,
        }
    # cohort model labels depend on drive_info, which in the store may come from earlier days
    passed = all(c["forward_equal"] and c["reverse_equal"] for c in comparisons.values())
    return {
        "check": "a_incremental_equals_batch",
        "window": list(window),
        "tables": comparisons,
        "passed": passed,
    }


def check_replay(kb: KnowledgeBase) -> dict[str, Any]:
    """(b) Recomputes the fixed sample's contexts as of every chunk end from the finished
    store and compares them with the hashes recorded while building."""
    meta = kb.metadata()
    sample, hashes = meta.get("replay_sample") or [], meta.get("replay_hashes") or {}
    mismatched = [
        day
        for day, h in sorted(hashes.items())
        if context_hash(kb, sample, dt.date.fromisoformat(day)) != h
    ]
    return {
        "check": "b_as_of_replay",
        "sample_drives": len(sample),
        "dates": len(hashes),
        "mismatched_dates": mismatched,
        "passed": bool(hashes) and not mismatched,
    }


def check_outcomes_vs_labels(
    kb: KnowledgeBase, *, labels_path: Path, split: str = "validation"
) -> dict[str, Any]:
    """(d) The store's resolved (final) outcomes against the gold labels for the same
    drive, date and horizon: they must agree exactly. Candidate days the store has not
    resolved are reported separately by their gold label; a gold 0 there is expected
    where the gold table used presence after the store's cutoff (data the store must not
    read), and a gold 1 there would be a miss."""
    outcomes = pl.DataFrame(
        kb.db.execute(
            "SELECT drive_id, date, label, basis FROM outcomes WHERE horizon_days = ?",
            (kb.horizon_days,),
        ).fetchall(),
        schema=["drive_id", "date", "kb_label", "basis"],
        orient="row",
    ).with_columns(pl.col("date").str.to_date())
    labels = (
        pl.scan_parquet(labels_path)
        .filter((pl.col("horizon_days") == kb.horizon_days) & (pl.col("split") == split))
        .select("drive_id", "date", pl.col("label").alias("gold_label"))
        .join(outcomes.lazy(), on=["drive_id", "date"], how="inner")
        .collect()
    )
    table = (
        labels.with_columns(
            pl.col("kb_label").cast(pl.String).fill_null("null"),
            pl.col("gold_label").cast(pl.String).fill_null("null"),
        )
        .group_by("kb_label", "gold_label")
        .len()
        .sort("kb_label", "gold_label")
    )
    disagree = labels.filter(
        (
            pl.col("kb_label").is_not_null()
            & pl.col("gold_label").is_not_null()
            & (pl.col("kb_label") != pl.col("gold_label"))
        )
        | (pl.col("kb_label").is_not_null() & pl.col("gold_label").is_null())
        | (pl.col("kb_label").is_null() & pl.col("gold_label").is_not_null())
    )
    unresolved = (
        pl.DataFrame(
            kb.db.execute(
                "SELECT c.drive_id, c.date FROM candidate_days c LEFT JOIN outcomes o "
                "ON o.drive_id = c.drive_id AND o.date = c.date AND o.horizon_days = ? "
                "WHERE o.drive_id IS NULL",
                (kb.horizon_days,),
            ).fetchall(),
            schema=["drive_id", "date"],
            orient="row",
        )
        .with_columns(pl.col("date").str.to_date())
        .join(
            pl.scan_parquet(labels_path)
            .filter((pl.col("horizon_days") == kb.horizon_days) & (pl.col("split") == split))
            .select("drive_id", "date", pl.col("label").alias("gold_label"))
            .collect(),
            on=["drive_id", "date"],
            how="left",
        )
    )
    unresolved_by_gold = (
        unresolved.with_columns(pl.col("gold_label").cast(pl.String).fill_null("null"))
        .group_by("gold_label")
        .len()
        .sort("gold_label")
        .to_dicts()
    )
    missed_failures = int((unresolved["gold_label"] == 1).sum())
    return {
        "check": "d_outcomes_vs_gold_labels",
        "compared": labels.height,
        "not_in_gold": outcomes.height - labels.height,
        "unresolved_candidate_days": unresolved.height,
        "unresolved_by_gold_label": unresolved_by_gold,
        "unresolved_with_gold_failure": missed_failures,
        "matrix": table.to_dicts(),
        "disagreements": disagree.height,
        "disagreement_examples": disagree.head(10)
        .with_columns(pl.col("date").cast(pl.String))
        .to_dicts(),
        "passed": disagree.height == 0 and missed_failures == 0,
    }


def check_idempotence(kb: KnowledgeBase, *, drive_day_path: Path, day: dt.date) -> dict[str, Any]:
    """(f) Re-ingests one day from the sources: the store's contents must not change."""
    scores_path, split, column = _source(kb)
    before = kb.content_hash()
    drive_ids = scope_drive_ids(scores_path, split=split)
    scores = score_chunk(scores_path, day, day, split=split, column=column)
    presence = presence_chunk(drive_day_path, day, day, drive_ids=drive_ids)
    stats = kb.ingest_snapshot(
        day,
        scores.select("drive_id", "p_fail"),
        source="score_table",
        presence=presence.select("drive_id", "drive_model", "model_family"),
        failures=presence.filter(pl.col("failure") == 1).select("drive_id"),
    )
    after = kb.content_hash()
    return {
        "check": "f_idempotence",
        "day": day.isoformat(),
        "stored": stats.stored,
        "conflicts": stats.conflicts,
        "passed": before == after and stats.stored == 0 and stats.conflicts == 0,
    }


def episode_summary(kb: KnowledgeBase, as_of: dt.date) -> dict[str, Any]:
    """Counts and outcomes of the episodes known on `as_of` (descriptive)."""
    episodes = kb.episodes_frame(as_of)
    if episodes.is_empty():
        return {"as_of": as_of.isoformat(), "episodes": 0}
    by_outcome = episodes.group_by("outcome").len().sort("outcome").to_dicts()
    length = episodes["days_flagged"]
    long_ = episodes.filter(pl.col("days_above_cordon") >= 7)
    return {
        "as_of": as_of.isoformat(),
        "episodes": episodes.height,
        "drives": episodes["drive_id"].n_unique(),
        "open": int(episodes["open"].sum()),
        "by_outcome": by_outcome,
        "days_flagged_median": float(length.median() or 0.0),  # type: ignore[arg-type]
        "days_flagged_p90": float(length.quantile(0.9) or 0.0),
        "with_7_or_more_days_above_cordon": long_.height,
        "of_which_failed": int((long_["label"] == 1).sum()),
        "of_which_survived": int((long_["label"] == 0).sum()),
    }
