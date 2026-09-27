import json

from src.logging_config import configure_logging, get_logger


def test_configure_logging_console_output_is_human_readable(capsys):
    configure_logging(json_output=False)
    logger = get_logger("test.console")
    logger.info("something_happened", drive_id="D-1", count=3)

    out = capsys.readouterr().out
    assert "something_happened" in out
    assert "drive_id" in out
    assert "D-1" in out


def test_configure_logging_json_output_is_valid_json_per_line(capsys):
    configure_logging(json_output=True)
    logger = get_logger("test.json")
    logger.info("something_happened", drive_id="D-1", count=3)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "something_happened"
    assert payload["drive_id"] == "D-1"
    assert payload["count"] == 3
    assert payload["level"] == "info"
