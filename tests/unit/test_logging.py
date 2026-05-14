"""Unit tests for setup_logging."""

from __future__ import annotations

from pathlib import Path

from memorymesh.observability.logging import setup_logging


def test_setup_logging_pretty_no_file() -> None:
    setup_logging(level="WARNING", log_file=None, log_format="pretty")


def test_setup_logging_json_no_file() -> None:
    setup_logging(level="DEBUG", log_file=None, log_format="json")


def test_setup_logging_with_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "test.log"
    setup_logging(level="INFO", log_file=log_file, log_format="pretty")
    assert log_file.parent.exists()
