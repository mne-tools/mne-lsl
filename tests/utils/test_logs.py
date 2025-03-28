from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from mne_lsl.utils.logs import _use_log_level, add_file_handler, logger, verbose, warn

if TYPE_CHECKING:
    from pathlib import Path


def test_default_log_level(caplog: pytest.LogCaptureFixture):
    """Test the default log level."""
    with _use_log_level("WARNING"):  # set to default
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


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_logger(level: str, caplog: pytest.LogCaptureFixture):
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
    with _use_log_level(level):
        for level_, function in level_functions.items():
            caplog.clear()
            function("101")
            if level_int[level] <= level_int[level_]:
                assert "101" in caplog.text
            else:
                assert "101" not in caplog.text


def test_verbose_on_function(caplog: pytest.LogCaptureFixture):
    """Test verbose decorator on functions."""

    @verbose
    def foo(verbose: bool | str | int | None = None):
        """Foo function."""
        logger.debug("101")

    assert foo.__doc__ == "Foo function."
    assert foo.__name__ == "foo"
    with _use_log_level("INFO"):
        caplog.clear()
        foo()
        assert "101" not in caplog.text

    for level in (20, 25, 30, True, False, "WARNING", "ERROR"):
        caplog.clear()
        foo(verbose=level)
        assert "101" not in caplog.text

    caplog.clear()
    foo(verbose="DEBUG")
    assert "101" in caplog.text


def test_verbose_on_method(caplog: pytest.LogCaptureFixture):
    """Test verbose decorator on methods."""

    class Foo:
        def __init__(self):
            pass

        @verbose
        def foo(self, verbose: bool | str | int | None = None):
            logger.debug("101")

        @staticmethod
        @verbose
        def foo2(verbose: bool | str | int | None = None):
            logger.debug("101")

    foo = Foo()
    with _use_log_level("INFO"):
        caplog.clear()
        foo.foo()
        assert "101" not in caplog.text
        caplog.clear()
        foo.foo(verbose="DEBUG")
        assert "101" in caplog.text

        # static method
        caplog.clear()
        Foo.foo2()
        assert "101" not in caplog.text
        caplog.clear()
        Foo.foo2(verbose="DEBUG")
        assert "101" in caplog.text


def test_file_handler(tmp_path: Path):
    """Test adding a file handler."""
    fname = tmp_path / "logs.txt"
    add_file_handler(fname)
    with _use_log_level("WARNING"):
        logger.warning("test1")
        logger.info("test2")
    with _use_log_level("INFO"):
        logger.info("test3")
    logger.handlers[-1].close()
    with open(fname) as file:
        lines = file.readlines()
    assert len(lines) == 2
    assert "test1" in lines[0]
    assert "test2" not in lines[0]
    assert "test2" not in lines[1]
    assert "test3" in lines[1]


def test_warn(tmp_path: Path):
    """Test warning functions."""
    with _use_log_level("ERROR"):
        warn("This is a warning.", RuntimeWarning)
    with (
        _use_log_level("WARNING"),
        pytest.warns(RuntimeWarning, match="This is a warning."),
    ):
        warn("This is a warning.", RuntimeWarning)
    fname = tmp_path / "logs.txt"
    add_file_handler(fname)
    with pytest.warns(RuntimeWarning, match="Grrrrr"):
        warn("Grrrrr", RuntimeWarning)
    with _use_log_level("ERROR"):
        warn("WoooW", RuntimeWarning)
    logger.handlers[-1].close()
    with open(fname) as file:
        lines = file.readlines()
    assert len(lines) == 1
    assert "Grrrrr" in lines[0]
