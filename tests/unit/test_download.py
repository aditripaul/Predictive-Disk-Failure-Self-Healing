import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingest.download import (
    DEFAULT_BACKBLAZE_BASE_URL,
    QuartersNotPublishedError,
    backblaze_quarter_url,
    download_backblaze,
    download_backblaze_quarter,
    download_file,
    download_smartz,
    extract_csv_members,
    format_quarter,
    generate_quarter_range,
    parse_quarter,
    resolve_quarters,
)


def _fake_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [content]
    response.headers = {"Content-Length": str(len(content))}
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_backblaze_quarter_url_matches_known_naming_convention():
    url = backblaze_quarter_url(DEFAULT_BACKBLAZE_BASE_URL, "Q1_2026")
    assert url == f"{DEFAULT_BACKBLAZE_BASE_URL}/data_Q1_2026.zip"


def test_backblaze_quarter_url_strips_trailing_slash_on_base_url():
    url = backblaze_quarter_url("https://example.com/base/", "Q1_2026")
    assert url == "https://example.com/base/data_Q1_2026.zip"


def test_parse_quarter_extracts_year_and_number():
    assert parse_quarter("Q3_2025") == (2025, 3)


def test_parse_quarter_rejects_malformed_strings():
    for bad in ("2025_Q3", "Q3-2025", "Q5_2025", "Q0_2025", "notaquarter"):
        with pytest.raises(ValueError):
            parse_quarter(bad)


def test_format_quarter_round_trips_with_parse_quarter():
    assert format_quarter(2025, 3) == "Q3_2025"
    assert parse_quarter(format_quarter(2025, 3)) == (2025, 3)


def test_generate_quarter_range_within_one_year():
    assert generate_quarter_range("Q1_2025", "Q3_2025") == ["Q1_2025", "Q2_2025", "Q3_2025"]


def test_generate_quarter_range_crosses_a_year_boundary():
    assert generate_quarter_range("Q3_2025", "Q2_2026") == [
        "Q3_2025",
        "Q4_2025",
        "Q1_2026",
        "Q2_2026",
    ]


def test_generate_quarter_range_single_quarter():
    assert generate_quarter_range("Q1_2025", "Q1_2025") == ["Q1_2025"]


def test_generate_quarter_range_rejects_start_after_end():
    with pytest.raises(ValueError):
        generate_quarter_range("Q2_2025", "Q1_2025")


def test_resolve_quarters_prefers_cli_explicit_list():
    result = resolve_quarters(
        cli_quarters=["Q1_2025"],
        config_quarters=["Q2_2025"],
        config_start_quarter="Q1_2020",
        config_end_quarter="Q1_2021",
    )
    assert result == ["Q1_2025"]


def test_resolve_quarters_falls_back_to_config_explicit_list():
    result = resolve_quarters(config_quarters=["Q2_2025", "Q3_2025"])
    assert result == ["Q2_2025", "Q3_2025"]


def test_resolve_quarters_uses_cli_range_when_no_explicit_list():
    result = resolve_quarters(cli_start_quarter="Q1_2025", cli_end_quarter="Q2_2025")
    assert result == ["Q1_2025", "Q2_2025"]


def test_resolve_quarters_uses_config_range_when_nothing_else_given():
    result = resolve_quarters(config_start_quarter="Q3_2025", config_end_quarter="Q4_2025")
    assert result == ["Q3_2025", "Q4_2025"]


def test_resolve_quarters_cli_range_overrides_config_range():
    result = resolve_quarters(
        cli_start_quarter="Q1_2025",
        cli_end_quarter="Q1_2025",
        config_start_quarter="Q1_2020",
        config_end_quarter="Q4_2020",
    )
    assert result == ["Q1_2025"]


def test_resolve_quarters_raises_a_clear_error_when_nothing_is_configured():
    with pytest.raises(ValueError, match="No time period configured"):
        resolve_quarters()


def test_download_file_streams_content_to_disk(tmp_path):
    dest = tmp_path / "out.bin"
    with patch("src.ingest.download.requests.get", return_value=_fake_response(b"hello world")):
        result = download_file("https://example.com/file.bin", dest)
    assert result == dest
    assert dest.read_bytes() == b"hello world"
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_file_raises_on_http_error(tmp_path):
    dest = tmp_path / "out.bin"
    response = _fake_response(b"")
    response.raise_for_status.side_effect = RuntimeError("404")
    with patch("src.ingest.download.requests.get", return_value=response):
        with pytest.raises(RuntimeError):
            download_file("https://example.com/missing.bin", dest)
    assert not dest.exists()


def _build_zip(tmp_path, name: str, members: dict[str, bytes]) -> Path:
    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for member_name, content in members.items():
            zf.writestr(member_name, content)
    return zip_path


def test_extract_csv_members_flattens_nested_paths_and_skips_non_csv(tmp_path):
    zip_path = _build_zip(
        tmp_path,
        "data.zip",
        {
            "2026-01-01.csv": b"drive_id,date\nZ1,2026-01-01\n",
            "nested/dir/2026-01-02.csv": b"drive_id,date\nZ2,2026-01-02\n",
            "readme.txt": b"not a csv",
        },
    )
    extract_dir = tmp_path / "extracted"
    extracted = extract_csv_members(zip_path, extract_dir)

    names = {p.name for p in extracted}
    assert names == {"2026-01-01.csv", "2026-01-02.csv"}
    assert (extract_dir / "2026-01-02.csv").read_bytes() == b"drive_id,date\nZ2,2026-01-02\n"
    assert not (extract_dir / "readme.txt").exists()


def test_download_backblaze_quarter_downloads_and_marks_complete(tmp_path):
    raw_dir = tmp_path / "raw"
    zip_bytes = _zip_bytes({"data_Q1_2026_01.csv": b"drive_id,date\nA,2026-01-01\n"})

    with patch("src.ingest.download.requests.get", return_value=_fake_response(zip_bytes)):
        extracted = download_backblaze_quarter("Q1_2026", raw_dir=raw_dir)

    assert len(extracted) == 1
    assert (raw_dir / "data_Q1_2026_01.csv").exists()
    assert (raw_dir / ".Q1_2026.downloaded").exists()


def test_download_backblaze_quarter_skips_when_already_downloaded(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / ".Q1_2026.downloaded").write_text("")

    with patch("src.ingest.download.requests.get") as mock_get:
        extracted = download_backblaze_quarter("Q1_2026", raw_dir=raw_dir)

    mock_get.assert_not_called()
    assert extracted == []


def test_download_backblaze_quarter_reuses_an_already_downloaded_archive(tmp_path):
    """If a previous run downloaded the zip but got interrupted before
    extraction finished (no marker file yet), re-running must extract from
    the existing archive rather than re-downloading it."""
    raw_dir = tmp_path / "raw"
    archive_dir = raw_dir / "_archives"
    archive_dir.mkdir(parents=True)
    zip_bytes = _zip_bytes({"data_Q1_2026_01.csv": b"drive_id,date\nA,2026-01-01\n"})
    (archive_dir / "data_Q1_2026.zip").write_bytes(zip_bytes)

    with patch("src.ingest.download.requests.get") as mock_get:
        extracted = download_backblaze_quarter("Q1_2026", raw_dir=raw_dir)

    mock_get.assert_not_called()
    assert len(extracted) == 1
    assert (raw_dir / ".Q1_2026.downloaded").exists()


def test_download_backblaze_quarter_force_redownloads(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / ".Q1_2026.downloaded").write_text("")
    zip_bytes = _zip_bytes({"data_Q1_2026_01.csv": b"drive_id,date\nA,2026-01-01\n"})

    with patch(
        "src.ingest.download.requests.get", return_value=_fake_response(zip_bytes)
    ) as mock_get:
        extracted = download_backblaze_quarter("Q1_2026", raw_dir=raw_dir, force=True)

    mock_get.assert_called_once()
    assert len(extracted) == 1


def test_download_backblaze_downloads_every_configured_quarter(tmp_path):
    raw_dir = tmp_path / "raw"
    zip_bytes = _zip_bytes({"data.csv": b"drive_id,date\nA,2026-01-01\n"})

    with patch(
        "src.ingest.download.requests.get", return_value=_fake_response(zip_bytes)
    ) as mock_get:
        results = download_backblaze(["Q1_2026", "Q2_2026"], raw_dir=raw_dir)

    assert mock_get.call_count == 2
    assert set(results.keys()) == {"Q1_2026", "Q2_2026"}
    assert (raw_dir / ".Q1_2026.downloaded").exists()
    assert (raw_dir / ".Q2_2026.downloaded").exists()


def _fake_error_response(status_code: int) -> MagicMock:
    """A response whose `raise_for_status` fails the way `requests` does, with
    the response (and so the status code) attached to the exception."""
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} Client Error", response=response
    )
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _responses_by_quarter(ok_bytes: bytes, failures: dict[str, int]):
    """Serves `ok_bytes` for every quarter except those in `failures`, which
    get that status code - so one run can mix published and unpublished
    quarters, as happens while waiting on a quarter that just closed."""

    def get(url: str, **_kwargs) -> MagicMock:
        for quarter, status_code in failures.items():
            if f"data_{quarter}.zip" in url:
                return _fake_error_response(status_code)
        return _fake_response(ok_bytes)

    return get


def test_download_backblaze_still_downloads_the_quarters_that_are_published(tmp_path):
    """A quarter Backblaze has not released yet must not cost the caller the
    ones it has - the released quarters are multi-GB downloads."""
    raw_dir = tmp_path / "raw"
    zip_bytes = _zip_bytes({"data.csv": b"drive_id,date\nA,2026-01-01\n"})

    with patch(
        "src.ingest.download.requests.get",
        side_effect=_responses_by_quarter(zip_bytes, {"Q3_2026": 404}),
    ):
        with pytest.raises(QuartersNotPublishedError) as excinfo:
            download_backblaze(["Q1_2026", "Q3_2026", "Q2_2026"], raw_dir=raw_dir)

    # Q2 follows the 404 in the list, so this also pins that the run keeps
    # going rather than stopping at the first missing quarter.
    assert (raw_dir / ".Q1_2026.downloaded").exists()
    assert (raw_dir / ".Q2_2026.downloaded").exists()
    assert not (raw_dir / ".Q3_2026.downloaded").exists()
    assert excinfo.value.unavailable == ["Q3_2026"]
    assert set(excinfo.value.downloaded) == {"Q1_2026", "Q2_2026"}


def test_download_backblaze_names_the_unpublished_quarters_in_its_message():
    """The message has to say which quarter is missing and what to do, since
    this is the common state right after a quarter closes."""
    error = QuartersNotPublishedError(["Q3_2026"], {})
    assert "Q3_2026" in str(error)
    assert "download.backblaze.quarters" in str(error)


def test_download_backblaze_aborts_on_an_http_error_that_is_not_a_404(tmp_path):
    """A 500 is a fault, not a release that has yet to happen, so it must not
    be reported as "not published" or swallowed."""
    raw_dir = tmp_path / "raw"
    zip_bytes = _zip_bytes({"data.csv": b"drive_id,date\nA,2026-01-01\n"})

    with patch(
        "src.ingest.download.requests.get",
        side_effect=_responses_by_quarter(zip_bytes, {"Q1_2026": 500}),
    ):
        with pytest.raises(requests.HTTPError):
            download_backblaze(["Q1_2026", "Q2_2026"], raw_dir=raw_dir)


def test_download_backblaze_quarter_raises_on_a_404_rather_than_marking_it_done(tmp_path):
    """The single-quarter helper keeps propagating; only the multi-quarter
    wrapper tolerates a 404. A marker here would permanently skip the quarter."""
    raw_dir = tmp_path / "raw"
    with patch("src.ingest.download.requests.get", return_value=_fake_error_response(404)):
        with pytest.raises(requests.HTTPError):
            download_backblaze_quarter("Q3_2026", raw_dir=raw_dir)
    assert not (raw_dir / ".Q3_2026.downloaded").exists()


def test_download_backblaze_cli_exits_non_zero_when_a_quarter_is_missing():
    """`make download-backblaze` must not report success for a quarter it was
    asked for and did not get."""
    from pipelines.download_backblaze import main

    with patch(
        "pipelines.download_backblaze.download_backblaze",
        side_effect=QuartersNotPublishedError(["Q3_2026"], {"Q1_2026": []}),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(["--quarters", "Q1_2026", "Q3_2026"])
    assert excinfo.value.code == 1


def test_download_smartz_raises_a_clear_error_without_a_url(tmp_path):
    with pytest.raises(ValueError, match="SMART-Z"):
        download_smartz(tmp_path / "raw")


def test_download_smartz_downloads_when_url_is_provided(tmp_path):
    raw_dir = tmp_path / "raw"
    zip_bytes = _zip_bytes({"smartz_2026_01.csv": b"disk_id,date\nS1,2026-01-01\n"})

    with patch("src.ingest.download.requests.get", return_value=_fake_response(zip_bytes)):
        extracted = download_smartz(raw_dir, url="https://example.com/smartz.zip")

    assert len(extracted) == 1
    assert (raw_dir / ".smartz.downloaded").exists()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()
