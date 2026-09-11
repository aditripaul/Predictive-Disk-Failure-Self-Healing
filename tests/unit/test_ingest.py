from pathlib import Path

import polars as pl

from src.ingest.backblaze import ingest_backblaze_file
from src.ingest.profiling import profile_bronze_dataset
from src.ingest.smartz import ingest_smartz_file
from src.ingest.synthetic_stub import build_synthetic_stub, ingest_synthetic_stub

BACKBLAZE_CSV = """date,serial_number,model,capacity_bytes,failure,\
smart_5_raw,smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw
2024-01-01,Z1,ST4000DM000,4000787030016,0,0,0,0,0,0
2024-01-02,Z1,ST4000DM000,4000787030016,0,1,0,0,0,0
2024-02-01,Z2,ST4000DM000,4000787030016,1,10,2,0,3,0
,Z3,ST4000DM000,4000787030016,0,0,0,0,0,0
"""

SMARTZ_CSV = """date,drive_id,vendor_attr_1,vendor_attr_2
2024-01-01,V1,10,20
2024-01-15,V2,15,25
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_ingest_backblaze_file_lands_bronze_partitions(tmp_path: Path):
    csv_path = _write(tmp_path, "2024_sample.csv", BACKBLAZE_CSV)
    bronze_root = tmp_path / "bronze" / "backblaze"

    written = ingest_backblaze_file(csv_path, bronze_root)

    assert len(written) == 2  # year=2024/month=01 and month=02
    df = pl.read_parquet(bronze_root / "**" / "*.parquet")
    assert df.height == 3  # the null-date row is dropped
    assert set(df.columns) >= {
        "drive_id",
        "drive_model",
        "capacity_bytes",
        "failure",
        "source_dataset",
        "source_file",
        "ingested_at",
        "schema_version",
    }
    assert (df["source_dataset"] == "backblaze").all()


def test_ingest_backblaze_file_raises_on_missing_columns(tmp_path: Path):
    csv_path = _write(tmp_path, "bad.csv", "date,serial_number\n2024-01-01,Z1\n")
    bronze_root = tmp_path / "bronze"

    try:
        ingest_backblaze_file(csv_path, bronze_root)
        raise AssertionError("expected ValueError for missing columns")
    except ValueError as exc:
        assert "missing required Backblaze columns" in str(exc)


def test_ingest_smartz_file_lands_bronze_partitions(tmp_path: Path):
    csv_path = _write(tmp_path, "smartz_sample.csv", SMARTZ_CSV)
    bronze_root = tmp_path / "bronze" / "smartz"

    written = ingest_smartz_file(csv_path, bronze_root)

    assert len(written) == 1
    df = pl.read_parquet(bronze_root / "**" / "*.parquet")
    assert df.height == 2
    assert (df["source_dataset"] == "smartz").all()


def test_synthetic_stub_shape():
    df = build_synthetic_stub()
    assert df.height == 5 * 10
    assert df["drive_id"].n_unique() == 5


def test_ingest_synthetic_stub_lands_bronze(tmp_path: Path):
    bronze_root = tmp_path / "bronze" / "synthetic"
    written = ingest_synthetic_stub(bronze_root)
    assert len(written) >= 1
    df = pl.read_parquet(bronze_root / "**" / "*.parquet")
    assert (df["source_dataset"] == "synthetic").all()


def test_profile_bronze_dataset_reports_row_count_and_failure_rate(tmp_path: Path):
    csv_path = _write(tmp_path, "2024_sample.csv", BACKBLAZE_CSV)
    bronze_root = tmp_path / "bronze" / "backblaze"
    ingest_backblaze_file(csv_path, bronze_root)

    report = profile_bronze_dataset(bronze_root)

    assert report["row_count"] == 3
    assert report["failure_rate"] == round(1 / 3, 6)
    assert "drive_id" in report["null_percentage_by_column"]
