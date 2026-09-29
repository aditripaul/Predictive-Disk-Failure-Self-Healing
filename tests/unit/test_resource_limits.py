from unittest.mock import patch

import pytest

from src.resource_limits import apply_memory_limit_from_config, apply_memory_limit_gb


def test_apply_memory_limit_gb_is_a_noop_when_none():
    with patch("src.resource_limits.resource") as mock_resource:
        apply_memory_limit_gb(None)
    mock_resource.setrlimit.assert_not_called()


def test_apply_memory_limit_gb_is_a_noop_when_zero_or_negative():
    with patch("src.resource_limits.resource") as mock_resource:
        apply_memory_limit_gb(0)
        apply_memory_limit_gb(-5)
    mock_resource.setrlimit.assert_not_called()


def test_apply_memory_limit_gb_converts_gigabytes_to_bytes():
    with patch("src.resource_limits.resource") as mock_resource:
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (-1, -1)
        apply_memory_limit_gb(12)

    mock_resource.setrlimit.assert_called_once()
    args, _ = mock_resource.setrlimit.call_args
    limit_name, (soft, hard) = args
    assert limit_name is mock_resource.RLIMIT_AS
    assert soft == 12 * 1024**3
    assert hard == 12 * 1024**3


def test_apply_memory_limit_gb_never_raises_an_existing_tighter_hard_limit():
    with patch("src.resource_limits.resource") as mock_resource:
        mock_resource.RLIM_INFINITY = -1
        existing_hard = 8 * 1024**3  # a tighter hard limit already in place
        mock_resource.getrlimit.return_value = (-1, existing_hard)
        apply_memory_limit_gb(12)

    args, _ = mock_resource.setrlimit.call_args
    _limit_name, (soft, hard) = args
    assert soft == 12 * 1024**3
    assert hard == existing_hard  # kept, not raised to 12GB


def test_apply_memory_limit_gb_logs_a_warning_instead_of_raising_on_failure():
    with patch("src.resource_limits.resource") as mock_resource:
        mock_resource.RLIM_INFINITY = -1
        mock_resource.getrlimit.return_value = (-1, -1)
        mock_resource.setrlimit.side_effect = OSError("nope")
        apply_memory_limit_gb(12)  # must not raise


def test_apply_memory_limit_gb_is_a_noop_on_platforms_without_the_resource_module():
    with patch("src.resource_limits.resource", None):
        apply_memory_limit_gb(12)  # must not raise (e.g. on Windows)


def test_apply_memory_limit_from_config_reads_configs_data_yaml():
    config = {"resource_limits": {"max_memory_gb": 4}}
    with (
        patch("src.resource_limits.load_yaml", return_value=config),
        patch("src.resource_limits.apply_memory_limit_gb") as mock_apply,
    ):
        apply_memory_limit_from_config()
    mock_apply.assert_called_once_with(4)


def test_apply_memory_limit_from_config_defaults_to_none_when_section_absent():
    with (
        patch("src.resource_limits.load_yaml", return_value={}),
        patch("src.resource_limits.apply_memory_limit_gb") as mock_apply,
    ):
        apply_memory_limit_from_config()
    mock_apply.assert_called_once_with(None)


@pytest.fixture(autouse=True)
def _never_actually_limit_this_test_process():
    """Belt-and-suspenders: even though every test above mocks `resource`
    directly, this guards against a future test accidentally calling the
    real apply_memory_limit_gb/from_config and capping the pytest
    process's own memory."""
    with patch("resource.setrlimit") as mock_setrlimit:
        yield mock_setrlimit
