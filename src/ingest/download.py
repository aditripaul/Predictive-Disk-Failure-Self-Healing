"""Configurable raw-data download helpers (docs/dataset_strategy.md
section 3). Backblaze publishes its quarterly Hard Drive Stats archives
as ZIP files at a stable, predictable URL; this module downloads and
extracts the configured quarters into `data/raw/backblaze/`, skipping
ones already present so re-running the download is cheap and idempotent.

SMART-Z has no public bulk-download API (docs/dataset_strategy.md section
3.2: "Request/download the SMART-Z dataset...") - `download_smartz`
downloads from a directly-configured URL if the caller has one (e.g. an
institutional access link), and raises a clear, actionable error
otherwise rather than silently doing nothing.

Downloads are streamed to disk in fixed-size chunks and ZIP members are
copied out via `shutil.copyfileobj`, never fully materialized in memory -
consistent with this project's RAM-conscious ingestion discipline
(docs/dataset_strategy.md section 5).
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

DEFAULT_BACKBLAZE_BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB streamed chunks


def backblaze_quarter_url(base_url: str, quarter: str) -> str:
    """`quarter` is e.g. `"Q1_2026"`, matching Backblaze's own file naming
    (`data_Q1_2026.zip`)."""
    return f"{base_url.rstrip('/')}/data_{quarter}.zip"


def parse_quarter(quarter: str) -> tuple[int, int]:
    """Parses a `"Q<n>_<year>"` string (e.g. `"Q1_2026"`) into
    `(year, quarter_number)`."""
    try:
        q_part, year_part = quarter.split("_")
        if not q_part.startswith("Q"):
            raise ValueError
        quarter_number = int(q_part[1:])
        year = int(year_part)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid quarter string: {quarter!r} (expected e.g. 'Q1_2026')") from exc
    if not 1 <= quarter_number <= 4:
        raise ValueError(f"Invalid quarter number in {quarter!r}: must be 1-4")
    return year, quarter_number


def format_quarter(year: int, quarter_number: int) -> str:
    return f"Q{quarter_number}_{year}"


def generate_quarter_range(start_quarter: str, end_quarter: str) -> list[str]:
    """Expands an inclusive range of quarters - e.g.
    `("Q3_2025", "Q2_2026")` -> `["Q3_2025", "Q4_2025", "Q1_2026",
    "Q2_2026"]` - so the configurable time period (`configs/data.yaml`'s
    `download.backblaze.start_quarter`/`end_quarter`, or `--start-quarter`/
    `--end-quarter`) never needs every individual quarter spelled out."""
    start_year, start_q = parse_quarter(start_quarter)
    end_year, end_q = parse_quarter(end_quarter)
    if (start_year, start_q) > (end_year, end_q):
        raise ValueError(f"start_quarter {start_quarter!r} is after end_quarter {end_quarter!r}")

    quarters = []
    year, q = start_year, start_q
    while (year, q) <= (end_year, end_q):
        quarters.append(format_quarter(year, q))
        q += 1
        if q > 4:
            q = 1
            year += 1
    return quarters


def download_file(url: str, dest_path: Path, *, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> Path:
    """Streams `url` to `dest_path`, writing to a `.part` sibling first so a
    failed/interrupted download never leaves a file that looks complete."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(dest_path)
    return dest_path


def extract_csv_members(zip_path: Path, extract_dir: Path) -> list[Path]:
    """Extracts every `.csv` member of `zip_path` into `extract_dir`,
    flattening any nested directory structure inside the archive (some
    Backblaze quarterly ZIPs nest CSVs one directory deep)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            target_path = extract_dir / Path(member).name
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target_path)
    return extracted


def resolve_quarters(
    *,
    cli_quarters: list[str] | None = None,
    cli_start_quarter: str | None = None,
    cli_end_quarter: str | None = None,
    config_quarters: list[str] | None = None,
    config_start_quarter: str | None = None,
    config_end_quarter: str | None = None,
) -> list[str]:
    """Resolves the configurable time period for `pipelines/
    download_backblaze.py` - no quarter is ever hardcoded as a real
    default. An explicit `quarters` list (CLI, then config) wins if
    given; otherwise a start/end range (CLI, then config) is expanded via
    `generate_quarter_range`. Raises `ValueError` if neither is
    configured anywhere."""
    quarters = cli_quarters or config_quarters
    if quarters:
        return list(quarters)

    start_quarter = cli_start_quarter or config_start_quarter
    end_quarter = cli_end_quarter or config_end_quarter
    if start_quarter and end_quarter:
        return generate_quarter_range(start_quarter, end_quarter)

    raise ValueError(
        "No time period configured. Set download.backblaze.quarters (or "
        "start_quarter/end_quarter) in configs/data.yaml, or pass "
        "--quarters Q1_2025 [...] / --start-quarter Q1_2025 --end-quarter Q4_2025."
    )


def download_backblaze_quarter(
    quarter: str,
    *,
    base_url: str = DEFAULT_BACKBLAZE_BASE_URL,
    raw_dir: Path,
    archive_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Downloads and extracts one Backblaze quarterly archive (e.g.
    `"Q1_2026"`) into `raw_dir`. A no-op if that quarter was already
    downloaded (tracked by a `.{quarter}.downloaded` marker file in
    `raw_dir`), unless `force=True`."""
    marker = raw_dir / f".{quarter}.downloaded"
    if marker.exists() and not force:
        return []

    archive_dir = archive_dir or raw_dir / "_archives"
    zip_path = archive_dir / f"data_{quarter}.zip"
    # Re-download only if the archive isn't already on disk (e.g. after an
    # extraction that got interrupted before the marker was written) - a
    # multi-hundred-MB-to-multi-GB quarterly archive shouldn't be re-pulled
    # over the network just to finish an incomplete local extraction.
    if not zip_path.exists() or force:
        download_file(backblaze_quarter_url(base_url, quarter), zip_path)

    extracted = extract_csv_members(zip_path, raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
    return extracted


def download_backblaze(
    quarters: list[str],
    *,
    base_url: str = DEFAULT_BACKBLAZE_BASE_URL,
    raw_dir: Path,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Downloads every quarter in `quarters` (the configurable time
    period - see `configs/data.yaml`'s `download.backblaze.quarters`).
    Returns `{quarter: [extracted_csv_paths]}`; a quarter already
    downloaded maps to `[]`."""
    return {
        quarter: download_backblaze_quarter(
            quarter, base_url=base_url, raw_dir=raw_dir, force=force
        )
        for quarter in quarters
    }


def download_smartz(raw_dir: Path, *, url: str | None = None, force: bool = False) -> list[Path]:
    """SMART-Z has no public bulk-download endpoint, so `url` must be
    supplied (either directly or via `configs/data.yaml`'s
    `download.smartz.url`) once access has been requested
    (docs/dataset_strategy.md section 3.2). Raises `ValueError` with that
    guidance if `url` is absent, rather than silently doing nothing."""
    if not url:
        raise ValueError(
            "SMART-Z has no public bulk-download API (docs/dataset_strategy.md "
            "section 3.2). Request access, then either set download.smartz.url "
            f"in configs/data.yaml to the archive URL you were given, or place "
            f"its CSV files directly into {raw_dir} yourself."
        )

    marker = raw_dir / ".smartz.downloaded"
    if marker.exists() and not force:
        return []

    archive_path = raw_dir / "_archives" / "smartz.zip"
    download_file(url, archive_path)

    extracted = extract_csv_members(archive_path, raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
    return extracted
