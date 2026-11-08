"""Filling the knowledge store from a score table and silver presence, one week at a
time (docs/agent_knowledge_base.md section 5.4).

Each chunk reads only its own dates: the score table with a date filter, and the silver
`drive_day` table for the drives in scope (presence, model, family, and `failure == 1`
rows; never its broadcast `failure_date` column). Days after the score table ends can
still be ingested for presence and failures, so outcomes whose horizon runs past the last
scored day can be resolved. Resumable: days up to `last_ingested_date` are skipped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from src.agent.knowledge.store import KnowledgeBase

#: The silver columns the builder reads; anything computed over the whole data range is
#: left out on purpose (tests assert the projection).
DRIVE_DAY_COLUMNS = ("drive_id", "date", "drive_model", "model_family", "failure")
REPLAY_SAMPLE = 500


def chunk_ranges(start: dt.date, end: dt.date, chunk_days: int) -> list[tuple[dt.date, dt.date]]:
    out, a = [], start
    while a <= end:
        b = min(end, a + dt.timedelta(days=chunk_days - 1))
        out.append((a, b))
        a = b + dt.timedelta(days=1)
    return out


def scope_drive_ids(scores_path: Path, *, split: str | None) -> pl.Series:
    scan = pl.scan_parquet(scores_path)
    if split is not None:
        scan = scan.filter(pl.col("split") == split)
    return scan.select("drive_id").unique().collect()["drive_id"].sort()


def replay_sample(
    scores_path: Path, *, split: str | None, column: str, threshold: float
) -> list[str]:
    """The REPLAY_SAMPLE drives with the most candidate days (ties by id): a fixed sample
    rich in episodes, whose contexts the builder hashes after every chunk."""
    scan = pl.scan_parquet(scores_path)
    if split is not None:
        scan = scan.filter(pl.col("split") == split)
    top = (
        scan.filter(pl.col(column) >= threshold)
        .group_by("drive_id")
        .len()
        .sort(["len", "drive_id"], descending=[True, False])
        .head(REPLAY_SAMPLE)
        .collect()
    )
    return sorted(top["drive_id"].to_list())


def score_chunk(
    scores_path: Path, a: dt.date, b: dt.date, *, split: str | None, column: str
) -> pl.DataFrame:
    scan = pl.scan_parquet(scores_path).filter(pl.col("date").is_between(a, b))
    if split is not None:
        scan = scan.filter(pl.col("split") == split)
    return scan.select("drive_id", "date", pl.col(column).alias("p_fail")).collect()


def presence_chunk(
    drive_day_path: Path, a: dt.date, b: dt.date, *, drive_ids: pl.Series
) -> pl.DataFrame:
    return (
        pl.scan_parquet(drive_day_path)
        .select(list(DRIVE_DAY_COLUMNS))
        .filter(pl.col("date").is_between(a, b) & pl.col("drive_id").is_in(drive_ids.implode()))
        .collect()
    )


def context_hash(kb: KnowledgeBase, drives: list[str], as_of: dt.date) -> str:
    contexts = kb.contexts(drives, as_of=as_of)
    digest = hashlib.sha256()
    for drive in sorted(contexts):
        digest.update(repr((drive, sorted(contexts[drive].to_json().items()))).encode())
    return digest.hexdigest()


def backfill(
    kb: KnowledgeBase,
    *,
    scores_path: Path,
    drive_day_path: Path,
    split: str | None,
    start: dt.date,
    end: dt.date,
    presence_end: dt.date | None = None,
    column: str = "score",
    chunk_days: int = 7,
    resume: bool = True,
    source: str = "score_table",
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Ingests every day from `start` to `presence_end` (default `end`): scores up to
    `end`, presence and failures throughout, then resolves outcomes as of each chunk's last
    day and records the replay hash of the fixed sample."""
    presence_end = presence_end or end
    meta = kb.metadata()
    last = meta.get("last_ingested_date")
    drive_ids = scope_drive_ids(scores_path, split=split)
    sample = meta.get("replay_sample") or replay_sample(
        scores_path, split=split, column=column, threshold=kb.candidate_threshold
    )
    hashes: dict[str, str] = dict(meta.get("replay_hashes") or {})
    kb.set_metadata(
        replay_sample=sample,
        source_scores=str(scores_path),
        source_column=column,
        source_split=split,
        built_from_start=start.isoformat(),
        built_from_end=end.isoformat(),
        presence_end=presence_end.isoformat(),
    )
    empty = pl.DataFrame(schema={"drive_id": pl.String, "date": pl.Date, "p_fail": pl.Float64})
    summary: dict[str, Any] = {"chunks": [], "drives_in_scope": drive_ids.len()}
    for a, b in chunk_ranges(start, presence_end, chunk_days):
        if resume and last is not None and b.isoformat() <= last:
            continue
        started = time.perf_counter()
        presence = presence_chunk(drive_day_path, a, b, drive_ids=drive_ids)
        scores = (
            score_chunk(scores_path, a, min(b, end), split=split, column=column)
            if a <= end
            else empty
        )
        stored = present = 0
        day = a
        while day <= b:
            if not (resume and last is not None and day.isoformat() <= last):
                day_presence = presence.filter(pl.col("date") == day)
                stats = kb.ingest_snapshot(
                    day,
                    scores.filter(pl.col("date") == day).select("drive_id", "p_fail"),
                    source=source,
                    presence=day_presence.select("drive_id", "drive_model", "model_family"),
                    failures=day_presence.filter(pl.col("failure") == 1).select("drive_id"),
                )
                stored += stats.stored
                present += stats.present
            day += dt.timedelta(days=1)
        resolved = kb.resolve_outcomes(b)
        hashes[b.isoformat()] = context_hash(kb, sample, b)
        kb.set_metadata(replay_hashes=hashes)
        chunk = {
            "start": a.isoformat(),
            "end": b.isoformat(),
            "presence_rows": presence.height,
            "score_rows": scores.height,
            "stored": stored,
            "present": present,
            "resolved": resolved,
            "seconds": round(time.perf_counter() - started, 2),
        }
        summary["chunks"].append(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
        del presence, scores
    summary["row_counts"] = kb.row_counts()
    return summary
