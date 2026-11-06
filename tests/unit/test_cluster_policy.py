import datetime as dt

import numpy as np
import polars as pl

from src.agent.cluster_policy import (
    add_outcomes,
    apply_policy,
    band_rates,
    compare_policies,
    derive_action_map,
    drive_table,
    tier_matrix,
)
from src.models.evaluation import drive_level_table

T = {"stage1": 0.2, "warn": 0.5, "cordon": 0.7, "migrate": 0.9, "drain": 0.9}
D0 = dt.date(2026, 7, 1)


def _rows():
    """Three days per drive. f*: failing (positive on days 1-2), h*: healthy."""
    spec = {
        "f_warn": ([0.1, 0.6, 0.55], [0, 1, 1]),
        "f_early_alert": ([0.95, 0.3, 0.3], [0, 1, 1]),  # alert only before its window
        "h_cordon": ([0.75, 0.1, 0.1], [0, 0, 0]),
        "h_monitor": ([0.25, 0.1, 0.1], [0, 0, 0]),
        "h_none": ([0.1, 0.1, 0.1], [0, 0, 0]),
    }
    rows = []
    for d, (scores, labels) in spec.items():
        for i, (s, y) in enumerate(zip(scores, labels, strict=True)):
            rows.append({"drive_id": d, "date": D0 + dt.timedelta(days=i), "score": s, "label": y})
    return pl.DataFrame(rows)


def test_drive_table_follows_the_drive_level_convention():
    rows = _rows()
    drives = drive_table(rows, T)
    got = {r["drive_id"]: r for r in drives.to_dicts()}
    assert "h_none" not in got  # below Stage 1
    assert got["f_warn"]["band"] == "warn" and got["f_warn"]["decision_date"] == D0 + dt.timedelta(
        days=1
    )
    # A failing drive is scored on its positive rows only: the pre-window alert does not count.
    assert got["f_early_alert"]["band"] == "monitor"
    assert got["h_cordon"]["band"] == "cordon" and got["h_cordon"]["decision_date"] == D0
    # Bands agree with drive_level_table's scores.
    ids, scores = drive_level_table(
        rows["drive_id"].to_numpy(), rows["label"].to_numpy(), rows["score"].to_numpy()
    )
    assert np.isclose(
        sorted(drives["drive_score"].to_list()), sorted(s for s in scores if s >= T["stage1"])
    ).all()


def test_outcomes():
    drives = drive_table(_rows(), T)
    meta = pl.DataFrame(
        {
            "drive_id": ["f_warn", "f_early_alert", "h_cordon", "h_monitor"],
            "failure_date": [
                D0 + dt.timedelta(days=3),
                D0 + dt.timedelta(days=3),
                D0 + dt.timedelta(days=45),
                None,
            ],
            "last_seen_date": [
                D0 + dt.timedelta(days=3),
                D0 + dt.timedelta(days=3),
                D0 + dt.timedelta(days=45),
                D0 + dt.timedelta(days=10),
            ],
        }
    )
    out = {
        r["drive_id"]: r
        for r in add_outcomes(drives, meta, data_end=dt.date(2026, 9, 30)).to_dicts()
    }
    assert out["f_warn"]["outcome"] == "correct" and out["f_warn"]["lead_days"] == 2
    assert out["h_cordon"]["outcome"] == "early"
    assert out["h_monitor"]["outcome"] == "removed"


def _fleet(seed: int = 0, n: int = 3000):
    """Suspects in warn and cordon bands, two clusters; cluster 1 at warn fails
    like cordon drives, cluster 0 at cordon fails like warn drives."""
    rng = np.random.default_rng(seed)
    band = rng.choice(["monitor", "warn", "cordon"], size=n, p=[0.5, 0.3, 0.2])
    cluster = rng.integers(0, 2, size=n)
    base = {"monitor": 0.03, "warn": 0.15, "cordon": 0.45}
    p = np.array([base[b] for b in band])
    p = np.where((band == "warn") & (cluster == 1), 0.5, p)
    p = np.where((band == "cordon") & (cluster == 0), 0.12, p)
    failing = rng.random(n) < p
    return pl.DataFrame(
        {
            "drive_id": [f"d{i}" for i in range(n)],
            "band": band,
            "cluster": cluster,
            "failing": failing,
            "outcome": np.where(failing, "correct", "active"),
            "model_family": ["X"] * n,
        }
    ).with_columns(
        lead_days=pl.when(pl.col("failing")).then(pl.lit(10)).otherwise(None).cast(pl.Int64)
    )


def test_action_map_moves_cells_one_step_by_band_rate():
    drives = _fleet()
    matrix = tier_matrix(drives, ["c0", "c1"])
    amap = {(m["cluster"], m["band"]): m for m in derive_action_map(matrix, band_rates(drives))}
    assert amap[(1, "warn")]["effective_band"] == "cordon"  # fails like cordon drives
    assert amap[(0, "cordon")]["effective_band"] == "warn"  # fails like warn drives
    assert all(abs(m["move"]) <= 1 for m in amap.values())
    # Small cells keep their band.
    tiny = derive_action_map(
        matrix.head(1).with_columns(pl.lit(5).alias("drives")), band_rates(drives)
    )
    assert tiny[0]["move"] == 0 and "fewer than" in tiny[0]["reason"]


def test_compare_policies_paired():
    drives = _fleet()
    total_failing = int(drives["failing"].sum()) + 20  # 20 failing drives Stage 1 never saw
    same = apply_policy(drives, [])
    comp = compare_policies(same, total_failing=total_failing, n_boot=50)
    assert comp["warn"]["paired"]["difference"] == 0.0 and comp["warn"]["moved_down"] == 0
    amap = derive_action_map(tier_matrix(drives, ["c0", "c1"]), band_rates(drives))
    policy = apply_policy(drives, amap)
    comp = compare_policies(policy, total_failing=total_failing, n_boot=200)
    # Demoting the weak cordon cell raises cordon precision; promoting the strong
    # warn cell keeps recall.
    assert comp["cordon"]["cluster"]["precision"] > comp["cordon"]["tier_only"]["precision"]
    assert comp["cordon"]["moved_down"] > 0 and comp["cordon"]["moved_up"] > 0
    assert comp["warn"]["tier_only"]["recall"] <= 1.0
    assert sum(comp["warn"]["outcomes"]["cluster"].values()) == comp["warn"]["cluster"]["flagged"]
