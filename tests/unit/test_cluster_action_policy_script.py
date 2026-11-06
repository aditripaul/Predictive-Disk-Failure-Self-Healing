"""End to end on a toy fleet: validation fits and writes the map without a test
look; the Q3 step records exactly one look; --figures-only reads no data."""

import datetime as dt
import json

import numpy as np
import polars as pl

import scripts.cluster_action_policy as script
from src.models.failure_clusters import SOURCE_COLUMNS

THRESH = {"warn": 0.5, "cordon": 0.7, "migrate": 0.9, "drain": 0.9}


def _split(rng, prefix: str, n: int, start: dt.date):
    """n drives x 4 days. Half the suspects show pending sectors and fail often;
    the other half show CRC errors and rarely fail."""
    gold, scores, meta = [], [], []
    for i in range(n):
        d = f"{prefix}{i}"
        kind = i % 3  # 0: pending, 1: crc, 2: quiet
        fails = rng.random() < (0.5 if kind == 0 else 0.08 if kind == 1 else 0.02)
        base = rng.uniform(0.25, 0.95) if kind < 2 else rng.uniform(0.0, 0.3)
        for day in range(4):
            date = start + dt.timedelta(days=day)
            row = dict.fromkeys(SOURCE_COLUMNS, 0.0)
            row.update(
                drive_id=d,
                date=date,
                model_family="X" if i % 2 else "Y",
                current_pending_sector_count=float(rng.poisson(30)) if kind == 0 else 0.0,
                udma_crc_error_count=float(rng.poisson(40)) if kind == 1 else 0.0,
                power_on_hours=float(rng.uniform(1, 6) * 8766),
                telemetry_coverage_30d=1.0,
            )
            for c in SOURCE_COLUMNS:
                if c.endswith("_days_since_last_increase"):
                    row[c] = 999
                if c.endswith("_zero_to_nonzero"):
                    row[c] = False
            gold.append(row)
            scores.append(
                {
                    "split": prefix,
                    "drive_id": d,
                    "date": date,
                    "label": int(fails and day >= 1),
                    "score": float(min(base + 0.05 * day, 0.99)),
                }
            )
        meta.append(
            {
                "drive_id": d,
                "failure_date": start + dt.timedelta(days=5) if fails else None,
                "last_seen_date": start + dt.timedelta(days=5 if fails else 90),
            }
        )
    return gold, scores, meta


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    g1, s1, m1 = _split(rng, "validation", 600, dt.date(2026, 3, 1))
    g2, s2, m2 = _split(rng, "test", 600, dt.date(2026, 7, 1))
    gold = pl.DataFrame(g1 + g2, infer_schema_length=None)
    gold = gold.with_columns(
        [
            pl.col(c).cast(pl.Int16)
            for c in SOURCE_COLUMNS
            if c.endswith("_days_since_last_increase")
        ]
    )
    gold.write_parquet(tmp_path / "features.parquet")
    pl.DataFrame(s1 + s2).with_columns(pl.col("label").cast(pl.Int8)).write_parquet(
        tmp_path / "scores.parquet"
    )
    pl.DataFrame(m1 + m2).write_parquet(tmp_path / "meta.parquet")
    run = tmp_path / "run"
    run.mkdir()
    (run / "frozen_spec.json").write_text(
        json.dumps({"action_tier_thresholds": THRESH, "drive_threshold": 0.8})
    )
    (run / "two_stage.json").write_text(json.dumps({"candidate_threshold": 0.3}))
    return [
        "--run-dir",
        str(run),
        "--scores",
        str(tmp_path / "scores.parquet"),
        "--features",
        str(tmp_path / "features.parquet"),
        "--metadata",
        str(tmp_path / "meta.parquet"),
        "--out",
        str(tmp_path / "out"),
        "--ks",
        "2",
        "3",
        "--hdbscan-sizes",
        "20",
        "--n-boot",
        "50",
        "--min-drives",
        "10",
    ]


def test_validation_then_q3_then_figures_only(tmp_path, monkeypatch):
    looks = []
    monkeypatch.setattr(script, "record_test_access", lambda **kw: looks.append(kw))
    args = _setup(tmp_path)
    out = tmp_path / "out"

    script.main([*args, "--validation-only"])
    assert looks == []
    assert (out / "validation.json").exists() and not (out / "q3_evaluation.json").exists()
    val = json.loads((out / "validation.json").read_text())
    assert val["chosen"]["algorithm"] in {"kmeans", "gmm", "agglomerative", "spectral", "hdbscan"}
    assert (out / "action_map.json").exists() and (
        out / "figures" / "tier_thresholds_validation.png"
    ).exists()
    model_before = (out / "cluster_model.json").read_text()

    script.main(args)
    assert len(looks) == 1
    assert (out / "cluster_model.json").read_text() == model_before  # the map is not refit
    test = json.loads((out / "q3_evaluation.json").read_text())
    assert set(test["comparison"]) == {"warn", "cordon", "migrate", "drain"}
    assert "passed" in test["rule"]
    for name in (
        "policy_comparison",
        "action_outcomes_by_cell",
        "tier_cluster_matrix_test",
        "policy_precision_recall",
        "lead_time_by_cell",
        "cluster_stability",
        "clustering_ablation",
        "cluster_signatures",
        "cluster_families",
        "tier_lift_targets",
    ):
        assert (out / "figures" / f"{name}.png").exists(), name

    for p in (out / "figures").iterdir():
        p.unlink()
    script.main([*args, "--figures-only"])
    assert len(looks) == 1
    assert len(list((out / "figures").iterdir())) == 12
    assert "Policy comparison: Q3 test" in (out / "tables.md").read_text()


def test_validation_resumes_from_partial_ablation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "record_test_access", lambda **kw: None)
    args = _setup(tmp_path)
    out = tmp_path / "out"
    script.main([*args, "--validation-only"])
    partial = json.loads((out / "ablation_partial.json").read_text())
    done = len(partial["rows"])
    assert done >= 5
    # Simulate an interrupted run: only the first three candidates were saved.
    partial["rows"] = partial["rows"][:3]
    (out / "ablation_partial.json").write_text(json.dumps(partial))
    (out / "validation.json").unlink()
    capsys.readouterr()
    script.main([*args, "--validation-only"])
    assert "resuming the ablation: 3 candidates already done" in capsys.readouterr().out
    assert len(json.loads((out / "validation.json").read_text())["ablation"]) == done
