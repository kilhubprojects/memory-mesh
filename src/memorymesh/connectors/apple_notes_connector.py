"""Apple Notes connector for MemoryMesh.

Reads notes from the Apple Notes SQLite database (``NoteStore.sqlite``)
and yields one :class:`~memorymesh.core.models.ParsedDocument` per note.

Database location
-----------------
macOS only: ``~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite``

This connector is a no-op on non-macOS platforms and will yield no
documents with a warning log.

Features
--------
* **SQLite only** - no external dependencies beyond the stdlib.
* **ZSNIPPET field** - note body text is read from the ``ZSNIPPET`` column
  of the ``ZICCLOUDSYNCINGOBJECT`` table.
* **Date filtering** - notes not modified within ``days_past`` are skipped.
* **Empty note skipping** - notes with no snippet text are skipped.

Usage
-----
::

    connector = AppleNotesConnector(AppleNotesConfig(days_past=365))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import platform
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_DEFAULT_DB = Path.home() / ("Library/Group Containers/group.com.apple.notes/NoteStore.sqlite")

# Apple Core Data epoch offset: seconds from 2001-01-01 to 1970-01-01
_APPLE_EPOCH_OFFSET = 978307200


class AppleNotesConfig(BaseModel):
    """Configuration for an Apple Notes source.

    Args:
        db_path: Explicit path to ``NoteStore.sqlite``.  Defaults to the
            standard macOS location.
        days_past: Only include notes modified within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    db_path: Path = _DEFAULT_DB
    days_past: int = 365
    source_name: str = "apple_notes"


class AppleNotesConnector:
    """Reads Apple Notes from SQLite and yields one ParsedDocument per note.

    Args:
        config: Database path, date filter, and source settings.
    """

    def __init__(self, config: AppleNotesConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Read notes from the SQLite database and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            note with content, with ``file_type=".applenotes"`` and
            metadata containing ``z_pk``, ``title``, ``created_at``,
            and ``modified_at``.
        """
        if platform.system() != "Darwin":
            logger.warning("AppleNotesConnector: skipped - not running on macOS")
            return

        if not self._cfg.db_path.exists():
            logger.warning(f"AppleNotesConnector: database not found: {self._cfg.db_path}")
            return

        cutoff = self._cutoff()
        total = 0

        with tempfile.TemporaryDirectory() as tmp:
            tmp_db = Path(tmp) / "NoteStore.sqlite"
            shutil.copy2(self._cfg.db_path, tmp_db)

            try:
                conn = sqlite3.connect(str(tmp_db))
                conn.row_factory = sqlite3.Row
                rows = self._query(conn)
                conn.close()
            except Exception as exc:
                logger.warning(f"AppleNotesConnector: SQLite error: {exc}")
                return

        for row in rows:
            doc = self._build_doc(dict(row), cutoff)
            if doc is not None:
                yield doc
                total += 1

        logger.info(f"AppleNotesConnector: yielded {total} note(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _query(self, conn: sqlite3.Connection) -> list[Any]:
        """Run the SQL query to fetch note rows.

        Args:
            conn: Open SQLite connection.

        Returns:
            List of row objects.
        """
        sql = """
            SELECT
                Z_PK,
                ZTITLE1 AS title,
                ZSNIPPET AS snippet,
                ZCREATIONDATE AS created,
                ZMODIFICATIONDATE AS modified
            FROM ZICCLOUDSYNCINGOBJECT
            WHERE ZSNIPPET IS NOT NULL
              AND ZMARKEDFORDELETION = 0
            ORDER BY ZMODIFICATIONDATE DESC
        """
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.OperationalError:
            # Column names differ across macOS versions - fallback query
            sql_fallback = """
                SELECT
                    Z_PK,
                    ZTITLE AS title,
                    ZSNIPPET AS snippet,
                    ZCREATIONDATE AS created,
                    ZMODIFICATIONDATE AS modified
                FROM ZICCLOUDSYNCINGOBJECT
                WHERE ZSNIPPET IS NOT NULL
            """
            return conn.execute(sql_fallback).fetchall()

    @staticmethod
    def _apple_ts_to_iso(ts: float | None) -> str:
        """Convert Apple Core Data timestamp to ISO-8601 string.

        Args:
            ts: Seconds since 2001-01-01 (Apple epoch).

        Returns:
            ISO-8601 UTC string, or empty string on failure.
        """
        if ts is None:
            return ""
        try:
            unix_ts = float(ts) + _APPLE_EPOCH_OFFSET
            return datetime.fromtimestamp(unix_ts, tz=UTC).isoformat()
        except (ValueError, OSError, OverflowError):
            return ""

    def _build_doc(self, row: dict[str, Any], cutoff: datetime | None) -> ParsedDocument | None:
        """Convert a SQLite row to a ParsedDocument.

        Args:
            row: Dict of column name -> value.
            cutoff: UTC datetime cutoff; skip notes modified before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the note should be skipped.
        """
        z_pk = row.get("Z_PK")
        if z_pk is None:
            return None

        snippet = (row.get("snippet") or "").strip()
        if not snippet:
            return None

        title = (row.get("title") or "Untitled").strip()
        modified_raw = row.get("modified")
        created_raw = row.get("created")

        if cutoff and modified_raw is not None:
            try:
                unix_ts = float(modified_raw) + _APPLE_EPOCH_OFFSET
                dt = datetime.fromtimestamp(unix_ts, tz=UTC)
                if dt < cutoff:
                    return None
            except (ValueError, OSError, OverflowError) as exc:
                logger.debug(f"AppleNotesConnector: ignoring unparsable timestamp: {exc}")

        created_at = self._apple_ts_to_iso(created_raw)
        modified_at = self._apple_ts_to_iso(modified_raw)

        text = f"# {title}\n\n{snippet}"

        return ParsedDocument(
            path=Path(f"applenotes://{z_pk}.applenotes"),
            text=text,
            file_type=".applenotes",
            encoding="utf-8",
            metadata={
                "z_pk": z_pk,
                "title": title,
                "created_at": created_at,
                "modified_at": modified_at,
                "source": self._cfg.source_name,
            },
        )
