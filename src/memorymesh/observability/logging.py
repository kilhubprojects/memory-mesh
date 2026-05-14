"""Centralized logging configuration for MemoryMesh using loguru.

All modules obtain the shared ``logger`` by importing directly from loguru::

    from loguru import logger

This module is responsible only for *configuring* the sink(s) at process startup.
Call :func:`setup_logging` exactly once, as early as possible in ``cli.py`` or
``server/app.py``, before any other MemoryMesh module emits a log line.

Privacy invariant: document content, chunk text, and query text must NEVER appear
in any log line. Log only paths, hashes, counts, sizes, and operational events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    log_format: Literal["pretty", "json"] = "pretty",
) -> None:
    """Configure loguru sinks for the MemoryMesh process.

    Removes the default loguru handler, then installs:
    - A stderr sink (always).
    - A rotating file sink (only when *log_file* is provided).

    Args:
        level: Minimum log level string — ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, or ``CRITICAL``.
        log_file: Absolute path to the rotating log file.  ``None`` disables
            file logging (e.g. during ``memorymesh info`` or tests).
        log_format: ``"pretty"`` for coloured, human-readable output (default);
            ``"json"`` for newline-delimited JSON suitable for log aggregators.
    """
    logger.remove()

    if log_format == "json":
        fmt = "{time:ISO8601} | {level} | {name}:{function}:{line} | {message}"
        serialize = True
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        serialize = False

    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        serialize=serialize,
        colorize=(log_format == "pretty"),
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            format=fmt,
            serialize=serialize,
            rotation="10 MB",
            retention="14 days",
            encoding="utf-8",
        )
