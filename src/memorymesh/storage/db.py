"""SQLite connection management helpers for MemoryMesh storage layer.

Provides a shared :class:`ConnectionFactory` type and the :func:`transact`
context manager so repository classes can work with the same connection without
duplicating connection-management logic.

All repository classes receive a ``ConnectionFactory`` at construction time so
they remain independent of how the connection is opened.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

# Type alias: a zero-argument callable that returns the shared SQLite connection.
ConnectionFactory = Callable[[], sqlite3.Connection]


def get_connection(path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transact(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on exit and rolls back on exception.

    Usage::

        with transact(conn) as c:
            c.execute("INSERT INTO ...")

    Args:
        conn: The SQLite connection to wrap.

    Yields:
        The same *conn*, ready for DML statements.

    Raises:
        Exception: Re-raises any exception after rolling back.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
