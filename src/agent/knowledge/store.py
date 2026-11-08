"""The Knowledge component of the MAPE-K loop: a time-indexed store of what the agent
has seen, with views as of a date (docs/agent_knowledge_base.md, section 5).

Only dated facts are stored, appended one day at a time:

- `candidate_days`: the drive-days the scorer passed to its second stage (p_fail at or
  above the Stage 1 candidate threshold), with their score;
- `presence_weeks`: which days each drive reported (a 7-bit mask per week);
- `drive_info`: first day seen, drive model and family;
- `failure_days`: the day a drive failed (silver rows with `failure == 1` only);
- `cohort_daily`: per model, family and fleet, the drives, candidates, flags and failures
  of each day;
- `outcomes`, `decisions`: resolved labels and the agent's recorded decisions.

Everything else (tier flags, episodes, a drive's context, outcomes) is derived at read time
from facts dated on or before the query date. So building day by day and building in one
pass give the same store, and a view as of a date never sees a later fact. Columns computed
from the whole data range (`event_type`, `event_date`, `feature_maturity`, `last_seen_date`,
the broadcast `failure_date`) are never read.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

SCHEMA_VERSION = 1

#: Tables whose contents define the knowledge (the ingest log records timings and is
#: excluded from content comparisons).
DATA_TABLES = (
    "metadata",
    "candidate_days",
    "presence_weeks",
    "drive_info",
    "failure_days",
    "cohort_daily",
    "outcomes",
    "decisions",
)

#: The tier flags derived from a stored score, highest first.
FLAG_TIERS = ("primary", "cordon", "warn")

#: An outcome is censored when, a week after its horizon ended, the drive has neither
#: failed in the window nor been seen at or after the horizon's end.
CENSOR_GRACE_DAYS = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidate_days (
    drive_id TEXT NOT NULL, date TEXT NOT NULL, p_fail REAL NOT NULL,
    source TEXT NOT NULL, PRIMARY KEY (drive_id, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS presence_weeks (
    drive_id TEXT NOT NULL, week_start TEXT NOT NULL, mask INTEGER NOT NULL,
    max_p_fail REAL, PRIMARY KEY (drive_id, week_start)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS drive_info (
    drive_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, drive_model TEXT,
    model_family TEXT) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS failure_days (
    drive_id TEXT PRIMARY KEY, failure_date TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cohort_daily (
    key_type TEXT NOT NULL, key TEXT NOT NULL, date TEXT NOT NULL,
    drives INTEGER NOT NULL, candidates INTEGER NOT NULL, warn INTEGER NOT NULL,
    cordon INTEGER NOT NULL, failures INTEGER NOT NULL,
    PRIMARY KEY (key_type, key, date)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS outcomes (
    drive_id TEXT NOT NULL, date TEXT NOT NULL, horizon_days INTEGER NOT NULL,
    label INTEGER, basis TEXT NOT NULL, failure_date TEXT, resolved_at TEXT NOT NULL,
    PRIMARY KEY (drive_id, date, horizon_days)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY, cycle INTEGER, as_of TEXT, run_id TEXT,
    drive_id TEXT, proposed_tier TEXT, withheld_action TEXT, control_arm INTEGER,
    guardrail_passed INTEGER, human_review_required INTEGER, p_fail REAL,
    knowledge_context TEXT, trust_score_final REAL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ingest_log (
    date TEXT NOT NULL, source TEXT NOT NULL, rows_in INTEGER, present INTEGER,
    stored INTEGER, conflicts INTEGER, seconds REAL, rss_mb REAL,
    PRIMARY KEY (date, source)) WITHOUT ROWID;
"""


class KnowledgeConflict(ValueError):
    """The same (drive, day) arrived twice from one source with different scores."""


@dataclass(frozen=True)
class KnowledgeContext:
    """What the store knew about one drive as of a date. Scalars only, so it can live
    in the agent's checkpointed state."""

    as_of: str
    known: bool
    history_days: int
    days_as_candidate_window: int
    days_above_warn_window: int
    days_above_cordon_window: int
    days_above_primary_window: int
    prior_episodes: int
    open_episode_days: int | None
    open_episode_days_above_cordon: int | None
    days_since_first_flag: int | None
    last_resolved_outcome: str | None
    days_since_last_resolved: int | None
    cohort_rate_window: float | None
    drive_model: str | None
    model_family: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestStats:
    date: str
    source: str
    rows_in: int
    present: int
    stored: int
    conflicts: int


@dataclass(frozen=True)
class Episode:
    first_flag: dt.date
    last_flag: dt.date
    days_flagged: int
    days_above_cordon: int
    max_p_fail: float
    closed_as_of: dt.date


def _iso(day: dt.date) -> str:
    return day.isoformat()


def _date(value: str | dt.date | None) -> dt.date | None:
    if value is None or isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


def week_start(day: dt.date) -> dt.date:
    return day - dt.timedelta(days=day.weekday())


def outcome_as_of(
    day: dt.date,
    horizon_days: int,
    *,
    failure_date: dt.date | None,
    last_presence: dt.date | None,
    as_of: dt.date,
) -> tuple[int | None, str, dt.date | None]:
    """`(label, basis, resolved_at)` for "fails within `horizon_days` after `day`", using
    only facts known on `as_of` (the failure and presence passed in must be dated on or
    before it). Mirrors src/labels/labeling.py: 1 if a failure lies in (day, day + h]; 0 if
    the drive was seen on or after day + h; censored a week after day + h if neither;
    pending before that."""
    end = day + dt.timedelta(days=horizon_days)
    if failure_date is not None and failure_date <= as_of and day < failure_date <= end:
        return 1, "failure", failure_date
    if last_presence is not None and last_presence <= as_of and last_presence >= end:
        return 0, "presence", end
    if as_of >= end + dt.timedelta(days=CENSOR_GRACE_DAYS):
        return None, "censored_removed", end + dt.timedelta(days=CENSOR_GRACE_DAYS)
    return None, "pending", None


def episode_outcome(
    episode: Episode,
    horizon_days: int,
    *,
    failure_date: dt.date | None,
    last_presence: dt.date | None,
    as_of: dt.date,
) -> tuple[int | None, str, dt.date | None]:
    """`(label, basis, resolved_at)` of an episode, from facts known on `as_of`: failed if
    the drive failed after the episode's first flag and within `horizon_days` of its last
    (a drive is usually still flagged on the day it fails, so the row rule of
    `outcome_as_of`, which needs a failure strictly after the day, would miss it); survived
    if the drive was seen `horizon_days` after the last flag; censored a week after that if
    neither; pending before."""
    end = episode.last_flag + dt.timedelta(days=horizon_days)
    if (
        failure_date is not None
        and failure_date <= as_of
        and episode.first_flag < failure_date <= end
    ):
        return 1, "failure", failure_date
    if last_presence is not None and last_presence <= as_of and last_presence >= end:
        return 0, "presence", end
    if as_of >= end + dt.timedelta(days=CENSOR_GRACE_DAYS):
        return None, "censored_removed", end + dt.timedelta(days=CENSOR_GRACE_DAYS)
    return None, "pending", None


def derive_episodes(
    flags: list[tuple[dt.date, float]],
    *,
    cordon: float,
    gap_days: int,
    failure_date: dt.date | None,
) -> list[Episode]:
    """Episodes from a drive's flag days `(date, p_fail)` sorted by date: runs of flag
    days at most `gap_days` apart. An episode closes `gap_days + 1` days after its last
    flag, or on the failure day if the drive fails before that."""
    episodes: list[Episode] = []
    run: list[tuple[dt.date, float]] = []

    def close(run: list[tuple[dt.date, float]]) -> None:
        last = run[-1][0]
        closed = last + dt.timedelta(days=gap_days + 1)
        if failure_date is not None and last < failure_date < closed:
            closed = failure_date
        episodes.append(
            Episode(
                first_flag=run[0][0],
                last_flag=last,
                days_flagged=len(run),
                days_above_cordon=sum(1 for _, p in run if p >= cordon),
                max_p_fail=max(p for _, p in run),
                closed_as_of=closed,
            )
        )

    for day, p in flags:
        if run and (day - run[-1][0]).days > gap_days:
            close(run)
            run = []
        run.append((day, p))
    if run:
        close(run)
    return episodes


class KnowledgeBase:
    """The store and its as-of views. `path=None` keeps it in memory (same code)."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
        thresholds: Mapping[str, float] | None = None,
        candidate_threshold: float | None = None,
        horizon_days: int | None = None,
        episode_gap_days: int | None = None,
        context_window_days: int | None = None,
    ) -> None:
        self.path = None if path is None else Path(path)
        target = ":memory:" if self.path is None else str(self.path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(target)
        if self.path is not None:
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        stored = self.metadata()
        new_store = "run_id" not in stored
        if new_store:
            episode_gap_days = 7 if episode_gap_days is None else episode_gap_days
            context_window_days = 180 if context_window_days is None else context_window_days
        given = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "thresholds": None
            if thresholds is None
            else {k: float(v) for k, v in thresholds.items()},
            "candidate_threshold": candidate_threshold,
            "horizon_days": horizon_days,
            "episode_gap_days": episode_gap_days,
            "context_window_days": context_window_days,
        }
        with self.db:
            for key, value in given.items():
                if value is None:
                    continue
                if key in stored and stored[key] != value:
                    raise ValueError(
                        f"knowledge base {target} was built with {key}={stored[key]!r}, "
                        f"not {value!r}"
                    )
                if key not in stored:
                    self._set(key, value)
        meta = self.metadata()
        missing = {"run_id", "thresholds", "candidate_threshold", "horizon_days"} - set(meta)
        if missing:
            raise ValueError(f"a new knowledge base needs {sorted(missing)}")
        self.run_id: str = meta["run_id"]
        self.thresholds: dict[str, float] = meta["thresholds"]
        for tier in FLAG_TIERS:
            if tier not in self.thresholds:
                raise ValueError(
                    f"thresholds need {list(FLAG_TIERS)}, got {sorted(self.thresholds)}"
                )
        self.candidate_threshold = float(meta["candidate_threshold"])
        self.horizon_days = int(meta["horizon_days"])
        self.episode_gap_days = int(meta["episode_gap_days"])
        self.context_window_days = int(meta["context_window_days"])

    @classmethod
    def in_memory(cls, **meta: Any) -> KnowledgeBase:
        return cls(None, **meta)

    # --- metadata -------------------------------------------------------------------

    def _set(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO metadata VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def metadata(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM metadata")}

    def set_metadata(self, **values: Any) -> None:
        with self.db:
            for key, value in values.items():
                self._set(key, value)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes ---------------------------------------------------------------------

    def ingest_snapshot(
        self,
        day: dt.date,
        scored: pl.DataFrame,
        *,
        source: str,
        presence: pl.DataFrame | None = None,
        failures: pl.DataFrame | None = None,
    ) -> IngestStats:
        """Appends one day: the candidate rows of `scored` (`drive_id, p_fail`), the
        drives present (`presence`: `drive_id` and optionally `drive_model`,
        `model_family`; default: the scored drives), and the drives that failed that day
        (`failures`: `drive_id`). One transaction; re-ingesting the same day changes
        nothing. A different score for a stored (drive, day) from the same source raises
        `KnowledgeConflict`; from another source it is counted and the first is kept."""
        started = time.perf_counter()
        iso = _iso(day)
        if "date" in scored.columns and scored.height and (scored["date"] != day).any():
            raise ValueError(f"scored frame has rows not dated {iso}")
        present = presence if presence is not None else scored
        present = present.unique(subset="drive_id", keep="first")
        failed = (
            set(failures["drive_id"].to_list())
            if failures is not None and failures.height
            else set()
        )
        warn, cordon = self.thresholds["warn"], self.thresholds["cordon"]
        candidates = scored.filter(pl.col("p_fail") >= self.candidate_threshold)

        stored = conflicts = 0
        with self.db:
            existing = {
                drive: (p, src)
                for drive, p, src in self.db.execute(
                    "SELECT drive_id, p_fail, source FROM candidate_days WHERE date = ?", (iso,)
                )
            }
            new_rows = []
            for drive, p in zip(
                candidates["drive_id"].to_list(), candidates["p_fail"].to_list(), strict=True
            ):
                if drive in existing:
                    old_p, old_src = existing[drive]
                    if abs(old_p - p) > 1e-12:
                        if old_src == source:
                            raise KnowledgeConflict(
                                f"{drive} on {iso}: stored p_fail {old_p} from {source}, now {p}"
                            )
                        conflicts += 1
                    continue
                new_rows.append((drive, iso, float(p), source))
            self.db.executemany("INSERT INTO candidate_days VALUES (?, ?, ?, ?)", new_rows)
            stored = len(new_rows)

            p_by_drive = dict(
                zip(scored["drive_id"].to_list(), scored["p_fail"].to_list(), strict=True)
            )
            week, bit = _iso(week_start(day)), 1 << day.weekday()
            self.db.executemany(
                "INSERT INTO presence_weeks VALUES (?, ?, ?, ?) ON CONFLICT(drive_id, week_start) "
                "DO UPDATE SET mask = mask | excluded.mask, "
                "max_p_fail = CASE WHEN excluded.max_p_fail IS NULL THEN max_p_fail "
                "WHEN max_p_fail IS NULL THEN excluded.max_p_fail "
                "ELSE max(max_p_fail, excluded.max_p_fail) END",
                [(d, week, bit, p_by_drive.get(d)) for d in present["drive_id"].to_list()],
            )
            models = (
                present["drive_model"].to_list()
                if "drive_model" in present.columns
                else [None] * present.height
            )
            families = (
                present["model_family"].to_list()
                if "model_family" in present.columns
                else [None] * present.height
            )
            self.db.executemany(
                "INSERT INTO drive_info VALUES (?, ?, ?, ?) ON CONFLICT(drive_id) DO UPDATE SET "
                "first_seen = min(first_seen, excluded.first_seen), "
                "drive_model = CASE WHEN excluded.first_seen >= first_seen "
                "THEN coalesce(excluded.drive_model, drive_model) "
                "ELSE coalesce(drive_model, excluded.drive_model) END, "
                "model_family = CASE WHEN excluded.first_seen >= first_seen "
                "THEN coalesce(excluded.model_family, model_family) "
                "ELSE coalesce(model_family, excluded.model_family) END",
                [
                    (d, iso, m, f)
                    for d, m, f in zip(present["drive_id"].to_list(), models, families, strict=True)
                ],
            )
            self.db.executemany(
                "INSERT INTO failure_days VALUES (?, ?) ON CONFLICT(drive_id) DO UPDATE SET "
                "failure_date = min(failure_date, excluded.failure_date)",
                [(d, iso) for d in sorted(failed)],
            )
            self._write_cohorts(iso, present, p_by_drive, failed, warn, cordon)
            elapsed = time.perf_counter() - started
            self.db.execute(
                "INSERT OR REPLACE INTO ingest_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (iso, source, scored.height, present.height, stored, conflicts, elapsed, _rss_mb()),
            )
            last = self.metadata().get("last_ingested_date")
            if last is None or iso > last:
                self._set("last_ingested_date", iso)
        return IngestStats(iso, source, scored.height, present.height, stored, conflicts)

    def _write_cohorts(
        self,
        iso: str,
        present: pl.DataFrame,
        p_by_drive: Mapping[str, float],
        failed: set[str],
        warn: float,
        cordon: float,
    ) -> None:
        drives = present["drive_id"].to_list()
        info = self._drive_info(drives)
        rows: dict[tuple[str, str], list[int]] = {}
        for d in drives:
            _, model, family = info.get(d, (None, None, None))
            p = p_by_drive.get(d)
            keys = [
                ("fleet", "all"),
                ("drive_model", model or "unknown"),
                ("model_family", family or "unknown"),
            ]
            for key in keys:
                counts = rows.setdefault(key, [0, 0, 0, 0, 0])
                counts[0] += 1
                if p is not None:
                    counts[1] += p >= self.candidate_threshold
                    counts[2] += p >= warn
                    counts[3] += p >= cordon
                counts[4] += d in failed
        self.db.execute("DELETE FROM cohort_daily WHERE date = ?", (iso,))
        self.db.executemany(
            "INSERT INTO cohort_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(kt, k, iso, *c) for (kt, k), c in rows.items()],
        )

    def resolve_outcomes(self, as_of: dt.date) -> dict[str, int]:
        """Writes the outcome of every stored candidate day and decision whose
        `horizon_days` label is final on `as_of`, using only facts dated on or before it:
        a failure inside the horizon (label 1) or presence at or after its end (label 0).
        "Censored" is not final, because a drive that stopped reporting can come back after
        a telemetry gap (the gold label then says 0), so censored rows are counted but not
        written, and a later call resolves them if the drive reappears."""
        h = self.horizon_days
        done = {
            (d, day)
            for d, day in self.db.execute(
                "SELECT drive_id, date FROM outcomes WHERE horizon_days = ?", (h,)
            )
        }
        todo = [
            (d, day)
            for d, day in self.db.execute(
                "SELECT drive_id, date FROM candidate_days WHERE date <= ? "
                "UNION SELECT drive_id, as_of FROM decisions "
                "WHERE as_of IS NOT NULL AND as_of <= ?",
                (_iso(as_of), _iso(as_of)),
            )
            if (d, day) not in done
        ]
        drives = sorted({d for d, _ in todo})
        failures = self._failures(drives, as_of)
        presence = self._last_presence(drives, as_of)
        written = {"failure": 0, "presence": 0, "censored_removed": 0, "pending": 0}
        rows = []
        for d, day in todo:
            label, basis, resolved = outcome_as_of(
                dt.date.fromisoformat(day),
                h,
                failure_date=failures.get(d),
                last_presence=presence.get(d),
                as_of=as_of,
            )
            written[basis] += 1
            if basis in ("pending", "censored_removed"):
                continue
            failure = failures.get(d)
            rows.append(
                (
                    d,
                    day,
                    h,
                    label,
                    basis,
                    None if failure is None else _iso(failure),
                    _iso(resolved) if resolved is not None else day,
                )
            )
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            last = self.metadata().get("last_resolved_date")
            if last is None or _iso(as_of) > last:
                self._set("last_resolved_date", _iso(as_of))
        return written

    def record_cycle(self, state: Mapping[str, Any], *, cycle: int | None = None) -> str | None:
        """Records one agent cycle's decision (from the graph state after Plan and
        Validate). Returns the decision id, or None when the cycle proposed nothing."""
        proposal = state.get("proposed_action")
        if not proposal:
            return None
        guardrail = state.get("guardrail_result") or {}
        decision_id = state.get("decision_record_id") or proposal.get("action_id")
        as_of = proposal.get("as_of") or (state.get("timestamp") or "")[:10] or None
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO decisions "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    decision_id,
                    cycle,
                    as_of,
                    state.get("run_id"),
                    proposal.get("drive_id"),
                    proposal.get("proposed_action"),
                    proposal.get("withheld_action"),
                    int(bool(proposal.get("control_arm"))),
                    int(bool(guardrail.get("passed", True))),
                    int(bool(state.get("human_review_required"))),
                    proposal.get("p_fail"),
                    json.dumps(proposal.get("knowledge_context")),
                ),
            )
        return decision_id

    # --- reads ----------------------------------------------------------------------

    def _with_ids(self, drives: Iterable[str]) -> None:
        self.db.execute("CREATE TEMP TABLE IF NOT EXISTS _ids (drive_id TEXT PRIMARY KEY)")
        self.db.execute("DELETE FROM _ids")
        self.db.executemany("INSERT OR IGNORE INTO _ids VALUES (?)", [(d,) for d in drives])

    def _drive_info(self, drives: list[str]) -> dict[str, tuple[str, str | None, str | None]]:
        self._with_ids(drives)
        return {
            d: (first, model, family)
            for d, first, model, family in self.db.execute(
                "SELECT i.drive_id, first_seen, drive_model, model_family "
                "FROM drive_info i JOIN _ids USING (drive_id)"
            )
        }

    def _failures(self, drives: list[str], as_of: dt.date) -> dict[str, dt.date]:
        self._with_ids(drives)
        return {
            d: dt.date.fromisoformat(f)
            for d, f in self.db.execute(
                "SELECT f.drive_id, failure_date FROM failure_days f JOIN _ids USING (drive_id) "
                "WHERE failure_date <= ?",
                (_iso(as_of),),
            )
        }

    def _last_presence(self, drives: list[str], as_of: dt.date) -> dict[str, dt.date]:
        """The last day on or before `as_of` each drive was seen."""
        self._with_ids(drives)
        out: dict[str, dt.date] = {}
        rows = self.db.execute(
            "SELECT drive_id, week_start, mask FROM ("
            " SELECT p.drive_id, week_start, mask, ROW_NUMBER() OVER "
            " (PARTITION BY p.drive_id ORDER BY week_start DESC) AS rn"
            " FROM presence_weeks p JOIN _ids USING (drive_id) WHERE week_start <= ?)"
            " WHERE rn <= 2",
            (_iso(as_of),),
        ).fetchall()
        for d, ws, mask in sorted(rows, key=lambda r: (r[0], r[1]), reverse=True):
            if d in out:
                continue
            start = dt.date.fromisoformat(ws)
            for offset in range(6, -1, -1):
                day = start + dt.timedelta(days=offset)
                if mask >> offset & 1 and day <= as_of:
                    out[d] = day
                    break
        return out

    def cohort_rates(
        self, as_of: dt.date, *, key_type: str = "drive_model", window_days: int | None = None
    ) -> dict[str, float]:
        """Annualised failure rate per cohort over (as_of - window, as_of]."""
        window = window_days or self.context_window_days
        start = as_of - dt.timedelta(days=window)
        return {
            key: failures / drive_days * 365.0
            for key, drive_days, failures in self.db.execute(
                "SELECT key, SUM(drives), SUM(failures) FROM cohort_daily WHERE key_type = ? "
                "AND date > ? AND date <= ? GROUP BY key",
                (key_type, _iso(start), _iso(as_of)),
            )
            if drive_days
        }

    def contexts(self, drives: Iterable[str], *, as_of: dt.date) -> dict[str, KnowledgeContext]:
        """Each drive's `KnowledgeContext` as of `as_of`, from facts dated on or before it."""
        drives = sorted(set(drives))
        if not drives:
            return {}
        iso = _iso(as_of)
        window_start = as_of - dt.timedelta(days=self.context_window_days)
        info = {d: v for d, v in self._drive_info(drives).items() if v[0] <= iso}
        failures = self._failures(drives, as_of)
        presence = self._last_presence(drives, as_of)
        rates = self.cohort_rates(as_of)
        self._with_ids(drives)
        candidates: dict[str, list[tuple[dt.date, float]]] = {}
        for d, day, p in self.db.execute(
            "SELECT c.drive_id, date, p_fail FROM candidate_days c JOIN _ids USING (drive_id) "
            "WHERE date <= ? ORDER BY c.drive_id, date",
            (iso,),
        ):
            candidates.setdefault(d, []).append((dt.date.fromisoformat(day), p))
        warn, cordon, primary = (self.thresholds[t] for t in ("warn", "cordon", "primary"))
        out: dict[str, KnowledgeContext] = {}
        for d in drives:
            first_seen, model, family = info.get(d, (None, None, None))
            rows = candidates.get(d, [])
            recent = [(day, p) for day, p in rows if day > window_start]
            flags = [(day, p) for day, p in rows if p >= warn]
            failure = failures.get(d)
            episodes = derive_episodes(
                flags, cordon=cordon, gap_days=self.episode_gap_days, failure_date=failure
            )
            closed = [e for e in episodes if e.closed_as_of <= as_of]
            open_ = next((e for e in episodes if e.closed_as_of > as_of), None)
            last_outcome, last_resolved = None, None
            for e in closed:
                label, basis, resolved = episode_outcome(
                    e,
                    self.horizon_days,
                    failure_date=failure,
                    last_presence=presence.get(d),
                    as_of=as_of,
                )
                if basis == "pending" or resolved is None:
                    continue
                if last_resolved is None or resolved >= last_resolved:
                    last_resolved = resolved
                    last_outcome = (
                        "failed" if label == 1 else "survived_horizon" if label == 0 else "censored"
                    )
            out[d] = KnowledgeContext(
                as_of=iso,
                known=first_seen is not None,
                history_days=0
                if first_seen is None
                else (as_of - dt.date.fromisoformat(first_seen)).days,
                days_as_candidate_window=len(recent),
                days_above_warn_window=sum(1 for _, p in recent if p >= warn),
                days_above_cordon_window=sum(1 for _, p in recent if p >= cordon),
                days_above_primary_window=sum(1 for _, p in recent if p >= primary),
                prior_episodes=len(closed),
                open_episode_days=None if open_ is None else (as_of - open_.first_flag).days,
                open_episode_days_above_cordon=None if open_ is None else open_.days_above_cordon,
                days_since_first_flag=None if not flags else (as_of - flags[0][0]).days,
                last_resolved_outcome=last_outcome,
                days_since_last_resolved=None
                if last_resolved is None
                else (as_of - last_resolved).days,
                cohort_rate_window=rates.get(model or "unknown"),
                drive_model=model,
                model_family=family,
            )
        return out

    def context_for(self, drive_id: str, *, as_of: dt.date) -> KnowledgeContext:
        return self.contexts([drive_id], as_of=as_of)[drive_id]

    def annotate(
        self, drives: list[dict[str, Any]], *, default_as_of: dt.date | None = None
    ) -> list[dict[str, Any]]:
        """Copies of the agent's candidate dicts with a `knowledge_context` (as of each
        drive's `as_of`, an ISO date, or `default_as_of`). A drive with no date gets none."""
        by_day: dict[dt.date, list[str]] = {}
        for drive in drives:
            day = _date(drive.get("as_of")) or default_as_of
            if day is not None:
                by_day.setdefault(day, []).append(drive["drive_id"])
        found = {day: self.contexts(ids, as_of=day) for day, ids in by_day.items()}
        out = []
        for drive in drives:
            day = _date(drive.get("as_of")) or default_as_of
            ctx = found.get(day, {}).get(drive["drive_id"]) if day is not None else None
            out.append({**drive, "knowledge_context": None if ctx is None else ctx.to_json()})
        return out

    def episodes_frame(self, as_of: dt.date) -> pl.DataFrame:
        """Every episode known on `as_of`, with its resolved outcome when known."""
        drives = [
            d
            for (d,) in self.db.execute(
                "SELECT DISTINCT drive_id FROM candidate_days WHERE date <= ?", (_iso(as_of),)
            )
        ]
        warn, cordon = self.thresholds["warn"], self.thresholds["cordon"]
        failures = self._failures(drives, as_of)
        presence = self._last_presence(drives, as_of)
        self._with_ids(drives)
        flags: dict[str, list[tuple[dt.date, float]]] = {}
        for d, day, p in self.db.execute(
            "SELECT c.drive_id, date, p_fail FROM candidate_days c JOIN _ids USING (drive_id) "
            "WHERE date <= ? AND p_fail >= ? ORDER BY c.drive_id, date",
            (_iso(as_of), warn),
        ):
            flags.setdefault(d, []).append((dt.date.fromisoformat(day), p))
        rows = []
        for d, fl in flags.items():
            for e in derive_episodes(
                fl, cordon=cordon, gap_days=self.episode_gap_days, failure_date=failures.get(d)
            ):
                label, basis, _ = episode_outcome(
                    e,
                    self.horizon_days,
                    failure_date=failures.get(d),
                    last_presence=presence.get(d),
                    as_of=as_of,
                )
                rows.append(
                    {
                        "drive_id": d,
                        "first_flag": e.first_flag,
                        "last_flag": e.last_flag,
                        "days_flagged": e.days_flagged,
                        "days_above_cordon": e.days_above_cordon,
                        "max_p_fail": e.max_p_fail,
                        "closed_as_of": e.closed_as_of,
                        "open": e.closed_as_of > as_of,
                        "outcome": basis,
                        "label": label,
                    }
                )
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def coverage(self, *, as_of: dt.date, fleet_drive_ids: Iterable[str]) -> dict[str, Any]:
        """How much history the store holds for a fleet on `as_of`."""
        fleet = sorted(set(fleet_drive_ids))
        info = self._drive_info(fleet)
        iso = _iso(as_of)
        firsts = [dt.date.fromisoformat(v[0]) for v in info.values() if v[0] <= iso]
        n = len(fleet)
        with_history = len(firsts)
        return {
            "as_of": iso,
            "fleet_drives": n,
            "with_any_history": with_history,
            "share_with_any_history": with_history / n if n else 0.0,
            "with_60_days_or_more": sum(1 for f in firsts if (as_of - f).days >= 60),
            "first_seen_within_30_days": sum(1 for f in firsts if (as_of - f).days < 30),
            "with_an_episode": len(
                {
                    d
                    for (d,) in self.db.execute(
                        "SELECT DISTINCT c.drive_id FROM candidate_days c "
                        "WHERE date <= ? AND p_fail >= ?",
                        (iso, self.thresholds["warn"]),
                    )
                }
                & set(fleet)
            ),
        }

    def daily_shares(self, start: dt.date, end: dt.date) -> pl.DataFrame:
        """Per day, the fleet's share of drives at Stage 1 (candidates), warn and cordon."""
        rows = self.db.execute(
            "SELECT date, drives, candidates, warn, cordon FROM cohort_daily "
            "WHERE key_type = 'fleet' AND date >= ? AND date <= ? ORDER BY date",
            (_iso(start), _iso(end)),
        ).fetchall()
        frame = pl.DataFrame(
            rows, schema=["date", "drives", "candidates", "warn", "cordon"], orient="row"
        )
        return frame.with_columns(
            (pl.col(c) / pl.col("drives")).alias(f"{c}_share")
            for c in ("candidates", "warn", "cordon")
        )

    def drift_report(
        self, *, reference: tuple[dt.date, dt.date], current: tuple[dt.date, dt.date]
    ) -> dict[str, Any]:
        """`src/reliability/drift.py::build_drift_report` on the daily shares of drives at
        Stage 1, warn and cordon, and on the stored candidate scores: the weekly Stage 1
        pass-share monitor runbook 1.1.5 asked for."""
        from src.reliability.drift import build_drift_report

        ref, cur = self.daily_shares(*reference), self.daily_shares(*current)

        def scores(a: dt.date, b: dt.date) -> np.ndarray:
            return np.array(
                [
                    p
                    for (p,) in self.db.execute(
                        "SELECT p_fail FROM candidate_days WHERE date >= ? AND date <= ?",
                        (_iso(a), _iso(b)),
                    )
                ]
            )

        features_ref = {
            f"{c}_share": ref[f"{c}_share"].to_numpy() for c in ("candidates", "warn", "cordon")
        }
        features_cur = {
            f"{c}_share": cur[f"{c}_share"].to_numpy() for c in ("candidates", "warn", "cordon")
        }
        features_ref["candidate_p_fail"] = scores(*reference)
        features_cur["candidate_p_fail"] = scores(*current)
        report = build_drift_report(features_ref, features_cur)
        report["reference"] = [_iso(reference[0]), _iso(reference[1])]
        report["current"] = [_iso(current[0]), _iso(current[1])]
        report["mean_shares"] = {
            "reference": {
                k: float(np.mean(v)) for k, v in features_ref.items() if k.endswith("_share")
            },
            "current": {
                k: float(np.mean(v)) for k, v in features_cur.items() if k.endswith("_share")
            },
        }
        return report

    def decisions_for_resolution(self) -> tuple[list[dict], dict[tuple[str, dt.date, int], dict]]:
        """Decisions shaped for `src/reliability/batch.py::resolve_final_trust_scores`, and
        the label lookup it needs, built from this store's own resolved outcomes. The
        provisional score is the recorded p_fail (the stand-in the user guide describes)."""
        decisions = []
        for row in self.db.execute(
            "SELECT decision_id, drive_id, as_of, proposed_tier, guardrail_passed, p_fail "
            "FROM decisions "
            "WHERE as_of IS NOT NULL"
        ):
            decision_id, drive, as_of, tier, passed, p = row
            decisions.append(
                {
                    "decision_id": decision_id,
                    "action_id": decision_id,
                    "drive_id": drive,
                    "date": dt.date.fromisoformat(as_of),
                    "horizon_days": self.horizon_days,
                    "proposed_action": tier,
                    "trust_score_provisional": float(p or 0.0),
                    "safety_violation": False,
                    "guardrail_severity": "none" if passed else "hard",
                }
            )
        labels = {}
        for d, day, h, label, failure in self.db.execute(
            "SELECT drive_id, date, horizon_days, label, failure_date FROM outcomes "
            "WHERE label IS NOT NULL"
        ):
            day_d = dt.date.fromisoformat(day)
            labels[(d, day_d, h)] = {
                "label": label,
                "days_to_event": None
                if failure is None
                else (dt.date.fromisoformat(failure) - day_d).days,
            }
        return decisions, labels

    def resolve_trust_scores(self) -> int:
        """Final trust scores for every decision whose outcome is known; stored on the
        decision. Returns how many were resolved."""
        from src.reliability.batch import resolve_final_trust_scores

        decisions, labels = self.decisions_for_resolution()
        resolved = resolve_final_trust_scores(decisions, labels)
        with self.db:
            self.db.executemany(
                "UPDATE decisions SET trust_score_final = ? WHERE decision_id = ?",
                [(r["trust_score_final"], r["decision_id"]) for r in resolved],
            )
        return len(resolved)

    def content_hash(
        self, tables: Iterable[str] = DATA_TABLES, *, exclude_keys: Iterable[str] = ()
    ) -> str:
        """SHA-256 of the sorted contents of `tables` (metadata keys in `exclude_keys`
        left out), for comparing two stores."""
        digest = hashlib.sha256()
        skip = set(exclude_keys)
        for table in tables:
            rows = self.db.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608 (fixed names)
            if table == "metadata":
                rows = [r for r in rows if r[0] not in skip]
            for row in sorted(rows, key=repr):
                digest.update(repr(row).encode())
            digest.update(f"|{table}|".encode())
        return digest.hexdigest()

    def row_counts(self) -> dict[str, int]:
        return {
            t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in (*DATA_TABLES, "ingest_log")
        }  # noqa: S608


def _rss_mb() -> float | None:
    from src.resource_limits import current_rss_bytes

    rss = current_rss_bytes()
    return None if rss is None else rss / 1e6
