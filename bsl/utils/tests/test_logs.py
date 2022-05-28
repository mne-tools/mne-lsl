"""Test _logs.py"""

import logging
from typing import Optional, Union

import pytest

from bsl.utils._logs import init_logger, logger, set_log_level, verbose

logger.propagate = True


def test_default_log_level(caplog):
    """Test the default log level."""
    init_logger()

    caplog.clear()
    logger.debug("101")
    assert "101" not in caplog.text

    caplog.clear()
    logger.info("101")
    assert "101" not in caplog.text

    caplog.clear()
    logger.warning("101")
    assert "101" in caplog.text

    caplog.clear()
    logger.error("101")
    assert "101" in caplog.text

    caplog.clear()
    logger.critical("101")
    assert "101" in caplog.text


@pytest.mark.parametrize(
    "level", ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
)
def test_logger(level, caplog):
    """Test basic logger functionalities."""
    level_functions = {
        "DEBUG": logger.debug,
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
        "CRITICAL": logger.critical,
    }
    level_int = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    set_log_level(level)

    for level_, function in level_functions.items():
        caplog.clear()
        function("101")
        if level_int[level] <= level_int[level_]:
            assert "101" in caplog.text
        else:
            assert "101" not in caplog.text


def test_verbose(caplog):
    """Test verbose decorator."""

    @verbose
    def foo(verbose: Optional[Union[bool, str, int]] = None):
        logger.debug("101")

    set_log_level("DEBUG")
    caplog.clear()
    foo()
    assert "101" in caplog.text

    for level in (20, 25, 30, True, False, "WARNING", "ERROR"):
        caplog.clear()
        foo(verbose=level)
        assert "101" not in caplog.text

    caplog.clear()
    foo(verbose="DEBUG")
    assert "101" in caplog.text
