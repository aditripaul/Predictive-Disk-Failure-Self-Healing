"""src/agent/knowledge/build.py, checks.py and scripts/build_agent_knowledge.py on a small
synthetic score table and silver drive_day table (docs/agent_knowledge_base.md section 5)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.agent.knowledge.build import DRIVE_DAY_COLUMNS, backfill, presence_chunk
from src.agent.knowledge.store import KnowledgeBase

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_agent_knowledge.py"
D0 = dt.date(2026, 1, 1)
N_DAYS, N_DRIVES, H = 50, 60, 10
META = {
    "run_id": "stub",
    "thresholds": {"warn": 0.6, "cordon": 0.8, "primary": 0.95},
    "candidate_threshold": 0.3,
    "horizon_days": H,
}


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Score table (validation drives d0..d39 scored to day 39), silver drive_day for all
    drives through day 49 (with a full-range failure_date column the builder must not
    read), and gold labels by the label definition."""
    rng = np.random.default_rng(0)
    fail = {
        f"d{k}": D0 + dt.timedelta(days=int(rng.integers(15, 45))) for k in range(0, N_DRIVES, 7)
    }
    leave = {f"d{k}": D0 + dt.timedelta(days=30) for k in range(3, N_DRIVES, 11)}
    score_rows, day_rows = [], []
    for i in range(N_DAYS):
        day = D0 + dt.timedelta(days=i)
        for k in range(N_DRIVES):
            d = f"d{k}"
            if d in fail and day > fail[d] or d in leave and day > leave[d]:
                continue
            near = d in fail and 0 <= (fail[d] - day).days <= 8
            p = float(rng.uniform(0.62, 0.99) if near else rng.uniform(0.0, 0.7))
            day_rows.append(
                {
                    "date": day,
                    "drive_id": d,
                    "drive_model": "M1" if k % 2 else "M2",
                    "model_family": "F1",
                    "failure": int(d in fail and day == fail[d]),
                    "failure_date": fail.get(d),
                }
            )
            if k < 40 and i < 40:
                score_rows.append(
                    {"split": "validation", "drive_id": d, "date": day, "label": 0, "score": p}
                )
    drive_day = tmp_path / "drive_day.parquet"
    pl.DataFrame(day_rows).write_parquet(drive_day)
    last_seen = {}
    for r in day_rows:
        last_seen[r["drive_id"]] = max(last_seen.get(r["drive_id"], D0), r["date"])
    label_rows = []
    for r in score_rows:
        d, day = r["drive_id"], r["date"]
        f = fail.get(d)
        if f is not None and day < f <= day + dt.timedelta(days=H):
            label = 1
        elif last_seen[d] >= day + dt.timedelta(days=H):
            label = 0
        else:
            label = None
        r["label"] = 0 if label is None else label
        label_rows.append(
            {"drive_id": d, "date": day, "horizon_days": H, "label": label, "split": "validation"}
        )
    scores = tmp_path / "scores.parquet"
    pl.DataFrame(score_rows).write_parquet(scores)
    labels = tmp_path / "labels.parquet"
    pl.DataFrame(label_rows, schema_overrides={"label": pl.Int8}).write_parquet(labels)
    return scores, drive_day, labels


def _backfill(kb, scores, drive_day, **kw):
    return backfill(
        kb,
        scores_path=scores,
        drive_day_path=drive_day,
        split="validation",
        start=D0,
        end=D0 + dt.timedelta(days=39),
        presence_end=D0 + dt.timedelta(days=49),
        **kw,
    )


def test_the_builder_reads_only_dated_columns(tmp_path):
    _, drive_day, _ = _sources(tmp_path)
    assert "failure_date" not in DRIVE_DAY_COLUMNS
    chunk = presence_chunk(
        drive_day, D0, D0 + dt.timedelta(days=6), drive_ids=pl.Series(["d0", "d1"])
    )
    assert chunk.columns == list(DRIVE_DAY_COLUMNS) and set(chunk["drive_id"]) == {"d0", "d1"}


def test_an_interrupted_build_resumes_to_the_same_store(tmp_path):  # check (g)
    scores, drive_day, _ = _sources(tmp_path)
    whole = KnowledgeBase.in_memory(**META)
    _backfill(whole, scores, drive_day)

    path = tmp_path / "kb.sqlite"
    calls = []

    def stop_after_three(chunk):
        calls.append(chunk)
        if len(calls) == 3:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        _backfill(KnowledgeBase(path, **META), scores, drive_day, on_chunk=stop_after_three)
    resumed = KnowledgeBase(path)
    summary = _backfill(resumed, scores, drive_day)
    assert summary["chunks"]  # the remaining chunks ran
    assert resumed.content_hash() == whole.content_hash()


def test_presence_after_the_last_score_resolves_late_outcomes(tmp_path):
    scores, drive_day, _ = _sources(tmp_path)
    kb = KnowledgeBase.in_memory(**META)
    _backfill(kb, scores, drive_day)
    # candidate days up to day 39 with a 10-day horizon end by day 49, so presence through
    # day 49 resolves them all, except rows with no final outcome: a row dated on its
    # drive's failure day (a failure must come strictly after the day) and rows of drives
    # that left before their horizon ended (censored, which is not final)
    as_of = D0 + dt.timedelta(days=49)
    kb.resolve_outcomes(as_of)
    unresolved = kb.db.execute(
        "SELECT c.drive_id, c.date, f.failure_date FROM candidate_days c "
        "LEFT JOIN outcomes o ON o.drive_id = c.drive_id AND o.date = c.date "
        "LEFT JOIN failure_days f ON f.drive_id = c.drive_id WHERE o.drive_id IS NULL"
    ).fetchall()
    left = {f"d{k}" for k in range(3, N_DRIVES, 11)}  # these leave after day 30
    assert unresolved
    for drive, day, failure in unresolved:
        assert day == failure or drive in left, (drive, day)
    assert kb.metadata()["presence_end"] == (D0 + dt.timedelta(days=49)).isoformat()


def _script():
    spec = importlib.util.spec_from_file_location("build_agent_knowledge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "stack.json").write_text(json.dumps({"candidate_threshold": 0.3}))
    (run / "t.json").write_text(
        json.dumps(
            {
                "score_column": "score",
                "threshold": 0.95,
                "action_tiers": {"warn": 0.6, "cordon": 0.8, "migrate": None, "drain": None},
            }
        )
    )
    return run


def test_build_then_check_passes_every_self_check(tmp_path):
    scores, drive_day, labels = _sources(tmp_path)
    run = _run_dir(tmp_path)
    script = _script()
    common = [
        "--knowledge",
        str(tmp_path / "kb.sqlite"),
        "--out",
        str(tmp_path / "out"),
        "--drive-day",
        str(drive_day),
    ]
    script.main(
        [
            *common,
            "build",
            "--scores",
            str(scores),
            "--run-dir",
            str(run),
            "--thresholds-file",
            "t.json",
            "--horizon-days",
            str(H),
            "--start",
            D0.isoformat(),
            "--end",
            (D0 + dt.timedelta(days=39)).isoformat(),
            "--presence-end",
            (D0 + dt.timedelta(days=49)).isoformat(),
        ]
    )
    script.main(
        [
            *common,
            "check",
            "--labels",
            str(labels),
            "--slice-start",
            "2026-01-10",
            "--pit-report",
            "none",
        ]
    )
    report = json.loads((tmp_path / "out" / "knowledge_checks.json").read_text())
    results = {c["check"]: c["passed"] for c in report["checks"]}
    assert results == {
        "a_incremental_equals_batch": True,
        "b_as_of_replay": True,
        "d_outcomes_vs_gold_labels": True,
        "f_idempotence": True,
    }, report["checks"]
    assert report["episodes"]


def test_reading_the_test_period_needs_the_flag_and_is_logged(tmp_path, monkeypatch):
    script = _script()
    logged = []
    monkeypatch.setattr(script, "record_test_access", lambda **k: logged.append(k))
    monkeypatch.setattr(script, "sealed_start", lambda: dt.date(2026, 7, 1))
    with pytest.raises(SystemExit, match="sealed start"):
        script.guard_test_dates(dt.date(2026, 7, 15), allow=False, purpose="p", extra={})
    script.guard_test_dates(dt.date(2026, 6, 30), allow=False, purpose="p", extra={})
    assert logged == []
    script.guard_test_dates(dt.date(2026, 7, 15), allow=True, purpose="p", extra={})
    assert len(logged) == 1 and logged[0]["variant"] == "agent_knowledge"


def test_pack_and_unpack_round_trip_and_refuse_a_tampered_file(tmp_path):
    scores, drive_day, _ = _sources(tmp_path)
    run = _run_dir(tmp_path)
    script = _script()
    kb_path, out = tmp_path / "kb.sqlite", tmp_path / "out"
    common = ["--knowledge", str(kb_path), "--out", str(out), "--drive-day", str(drive_day)]
    script.main(
        [
            *common,
            "build",
            "--scores",
            str(scores),
            "--run-dir",
            str(run),
            "--thresholds-file",
            "t.json",
            "--horizon-days",
            str(H),
            "--start",
            D0.isoformat(),
            "--end",
            (D0 + dt.timedelta(days=39)).isoformat(),
            "--presence-end",
            (D0 + dt.timedelta(days=49)).isoformat(),
        ]
    )
    script.main([*common, "pack"])
    info = json.loads((out / "agent_knowledge_pack.json").read_text())
    first = (out / info["packed_file"]).read_bytes()
    script.main([*common, "pack"])  # deterministic: the same bytes again
    assert (out / info["packed_file"]).read_bytes() == first

    with pytest.raises(SystemExit, match="exists"):
        script.main([*common, "unpack"])
    kb_path.unlink()
    script.main([*common, "unpack"])
    assert KnowledgeBase(kb_path).content_hash() == info["content_hash"]

    (out / info["packed_file"]).write_bytes(first[:-10] + b"0" * 10)
    with pytest.raises(SystemExit, match="SHA-256"):
        script.main([*common, "unpack", "--overwrite"])
