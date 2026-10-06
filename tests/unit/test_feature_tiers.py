import datetime as dt
from pathlib import Path

import polars as pl
import yaml

from pipelines.build_gold_features import (
    build_gold_features,
    resolve_feature_plan,
    resolve_secondary_plan,
)
from src.features.derivatives import add_secondary_deltas
from src.features.lifecycle import add_lifecycle_features
from src.features.registry import build_registry
from src.ingest.backblaze import ingest_backblaze_file

FEATURES_CONFIG = yaml.safe_load(Path("configs/features.yaml").read_text())
CORE = {
    "reallocated_sector_count",
    "current_pending_sector_count",
    "offline_uncorrectable",
    "reported_uncorrectable_errors",
    "command_timeout",
}


def _wide(n_days: int = 40) -> pl.DataFrame:
    rows = []
    for drive in ("a", "b"):
        for day in range(n_days):
            rows.append(
                {
                    "drive_id": drive,
                    "date": dt.date(2026, 1, 1) + dt.timedelta(days=day),
                    "reallocated_sector_count": float(day // 10),
                    "current_pending_sector_count": 0.0,
                    "offline_uncorrectable": 0.0,
                    "reported_uncorrectable_errors": 0.0,
                    "command_timeout": 0.0,
                    "seek_error_rate": float(day * 100),
                    "power_on_hours": 1000.0 + 24 * day,
                    "temperature_celsius": 30.0 if drive == "a" else None,
                    "udma_crc_error_count": 0.0,
                    "capacity_gb": 4000.0,
                    "telemetry_coverage_30d": 1.0,
                    "days_since_last_telemetry": 0,
                }
            )
    return pl.DataFrame(rows).sort("date")


def test_config_keeps_only_the_core_counters_in_the_full_family():
    assert set(FEATURES_CONFIG["priority_smart_attributes"]) == CORE
    assert set(FEATURES_CONFIG["secondary_smart_attributes"]).isdisjoint(CORE)


def test_secondary_attributes_get_the_light_family_only():
    gold = build_gold_features(_wide(), FEATURES_CONFIG)
    for attr in ("seek_error_rate", "power_on_hours", "temperature_celsius"):
        produced = {c for c in gold.columns if c.startswith(attr)}
        assert produced == {attr, f"{attr}_7d_delta", f"{attr}_30d_delta"}, produced
    # the core counters still get the full family
    assert "reallocated_sector_count_30d_mean" in gold.columns
    assert "reallocated_sector_count_days_since_last_increase" in gold.columns
    assert {"power_on_days", "temperature_spike", "active_defect_velocity_3d"} <= set(gold.columns)
    assert "active_defect_acceleration_7d_vs_30d" in gold.columns


def test_feature_confidence_ignores_secondary_attributes_that_are_missing():
    gold = build_gold_features(_wide(), FEATURES_CONFIG)
    # drive b has no temperature at all; its confidence must not be reduced for it
    assert gold.filter(pl.col("drive_id") == "b")["attribute_coverage_factor"].min() == 1.0


def test_gold_column_count_stays_bounded():
    gold = build_gold_features(_wide(), FEATURES_CONFIG)
    assert len(gold.columns) < 240, len(gold.columns)


def test_resolve_plans_only_use_columns_that_exist():
    columns = ["drive_id", "date", "reallocated_sector_count", "power_on_hours"]
    attributes, _, _ = resolve_feature_plan(columns, FEATURES_CONFIG)
    assert attributes == ["reallocated_sector_count"]
    assert resolve_secondary_plan(columns, FEATURES_CONFIG) == (["power_on_hours"], (7, 30))
    assert resolve_secondary_plan(columns, {"priority_smart_attributes": []})[0] == []


def test_add_secondary_deltas_is_change_over_n_observations_per_drive():
    frame = pl.DataFrame({"drive_id": ["a", "a", "a", "b"], "x": [1.0, 4.0, 9.0, 5.0]})
    out = add_secondary_deltas(frame, ["x"], windows_days=(1, 2))
    assert out["x_1d_delta"].to_list() == [None, 3.0, 5.0, None]
    assert out["x_2d_delta"].to_list() == [None, None, 8.0, None]
    assert add_secondary_deltas(frame, [], windows_days=(1,)).equals(frame)


def test_lifecycle_adds_power_on_days_only_when_power_on_hours_exists():
    frame = pl.DataFrame(
        {"drive_id": ["a"], "date": [dt.date(2026, 1, 1)], "power_on_hours": [48.0]}
    )
    assert add_lifecycle_features(frame)["power_on_days"].to_list() == [2.0]
    assert "power_on_days" not in add_lifecycle_features(frame.drop("power_on_hours")).columns


def test_registry_lists_the_new_features_that_were_produced():
    gold = build_gold_features(_wide(), FEATURES_CONFIG)
    secondary, windows = resolve_secondary_plan(gold.columns, FEATURES_CONFIG)
    names = {
        e.feature_name
        for e in build_registry(
            sorted(CORE),
            (7, 14, 30),
            {},
            secondary_attributes=secondary,
            secondary_windows_days=windows,
            gold_columns=gold.columns,
        )
    }
    assert {
        "power_on_hours_30d_delta",
        "active_defect_velocity_3d",
        "temperature_spike",
        "power_on_days",
        "reallocated_sector_count_days_since_last_increase",
    } <= names


def test_ingest_reads_an_optional_smart_column_that_starts_empty(tmp_path: Path):
    """smart_194 is empty for the first rows and numeric later: without the
    dtype override the column is inferred as String and lands as text."""
    header = (
        "date,serial_number,model,capacity_bytes,failure,"
        "smart_5_raw,smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw,"
        "smart_9_raw,smart_194_raw\n"
    )
    empty = "".join(f"2026-01-01,S{i},M,1000,0,0,0,0,0,0,10,\n" for i in range(10_050))
    late = "2026-01-01,LATE,M,1000,0,0,0,0,0,0,20,35.5\n"
    csv_path = tmp_path / "2026-01-01.csv"
    csv_path.write_text(header + empty + late)
    ingest_backblaze_file(csv_path, tmp_path / "bronze")
    df = pl.read_parquet(tmp_path / "bronze" / "**" / "*.parquet")
    assert df.schema["smart_194_raw"] == pl.Float64
    assert df.filter(pl.col("drive_id") == "LATE")["smart_194_raw"].to_list() == [35.5]
    assert "smart_7_raw" not in df.columns  # absent from the file: simply not ingested
