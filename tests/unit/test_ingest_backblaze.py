from datetime import date
from pathlib import Path

from src.ingest.backblaze import filter_files_by_date_range


def _paths(*names: str) -> list[Path]:
    return [Path(f"data/raw/backblaze/{name}") for name in names]


def test_filter_files_by_date_range_is_a_noop_when_neither_bound_is_set():
    files = _paths("2026-01-01.csv", "2026-02-15.csv")
    assert filter_files_by_date_range(files) == files


def test_filter_files_by_date_range_keeps_only_the_inclusive_range():
    files = _paths("2026-01-01.csv", "2026-01-15.csv", "2026-01-31.csv", "2026-02-01.csv")
    result = filter_files_by_date_range(files, start_date=None, end_date=None)
    assert result == files

    result = filter_files_by_date_range(
        files, start_date=date(2026, 1, 1), end_date=date(2026, 1, 31)
    )
    assert [p.name for p in result] == ["2026-01-01.csv", "2026-01-15.csv", "2026-01-31.csv"]


def test_filter_files_by_date_range_supports_an_open_ended_bound():
    files = _paths("2026-01-01.csv", "2026-01-15.csv", "2026-01-31.csv")
    result = filter_files_by_date_range(files, start_date=date(2026, 1, 15))
    assert [p.name for p in result] == ["2026-01-15.csv", "2026-01-31.csv"]

    result = filter_files_by_date_range(files, end_date=date(2026, 1, 15))
    assert [p.name for p in result] == ["2026-01-01.csv", "2026-01-15.csv"]


def test_filter_files_by_date_range_keeps_files_whose_stem_is_not_a_plain_date():
    files = _paths("2026-01-01.csv", "readme.csv")
    result = filter_files_by_date_range(files, start_date=date(2026, 6, 1))
    assert [p.name for p in result] == ["readme.csv"]
