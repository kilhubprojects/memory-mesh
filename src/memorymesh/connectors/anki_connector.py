"""Anki flashcard database connector for MemoryMesh.

Reads the local Anki SQLite collection (``collection.anki2``) and yields
one :class:`~memorymesh.core.models.ParsedDocument` per note.

Anki database schema (simplified)
----------------------------------
* ``col`` - single row; the ``decks`` column holds a JSON mapping
  from deck ID to deck information (including ``name``).
* ``notes`` - one row per flashcard note.  The ``flds`` column contains
  all fields separated by ``\\x1f``; the first field is the front, the
  second the back.
* ``cards`` - links notes to decks via ``nid`` (note ID) and ``did``
  (deck ID).

Features
--------
* **Safe copy** - the database is copied to a temporary directory before
  opening so Anki's write-ahead-lock is never touched.
* **HTML stripping** - Anki field content is HTML; it is stripped using
  the shared :func:`memorymesh.connectors._html.html_to_text` helper.
* **Auto-detection** - if ``db_path`` is ``None``, the default OS path
  ``~/Documents/Anki/User 1/collection.anki2`` is used.

Usage
-----
::

    connector = AnkiConnector(AnkiConfig(
        db_path=Path("~/Anki/User 1/collection.anki2"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._html import html_to_text
from memorymesh.core.models import ParsedDocument


class AnkiConfig(BaseModel):
    """Configuration for an Anki collection source.

    Args:
        db_path: Path to ``collection.anki2``.  ``None`` uses the default
            path ``~/Documents/Anki/User 1/collection.anki2``.
        source_name: Name used in the MemoryMesh source registry.
    """

    db_path: Path | None = None
    source_name: str = "anki"


class AnkiConnector:
    """Reads Anki flashcards and yields one ParsedDocument per note.

    Args:
        config: Database path and source settings.
    """

    def __init__(self, config: AnkiConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Copy the Anki database, query notes, and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            note, with ``file_type=".anki"`` and metadata containing
            ``deck``, ``note_id``, ``tags``, and ``modified``.
        """
        db_path = self._cfg.db_path
        if db_path is None:
            db_path = Path.home() / "Documents" / "Anki" / "User 1" / "collection.anki2"
        db_path = db_path.expanduser()
        if not db_path.exists():
            logger.warning(f"AnkiConnector: database not found: {db_path}")
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_db = Path(tmp) / "collection.anki2"
            try:
                shutil.copy2(db_path, tmp_db)
            except OSError as exc:
                logger.warning(f"AnkiConnector: cannot copy database: {exc}")
                return
            try:
                conn = sqlite3.connect(str(tmp_db))
                conn.row_factory = sqlite3.Row
                try:
                    yield from self._extract(conn)
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.warning(f"AnkiConnector: SQLite error: {exc}")

    def _load_deck_map(self, conn: sqlite3.Connection) -> dict[str, str]:
        """Parse the deck name mapping from the ``col`` table.

        Args:
            conn: Open SQLite connection to the copied database.

        Returns:
            Dict mapping deck-ID string to deck name string.
        """
        deck_map: dict[str, str] = {}
        try:
            row = conn.execute("SELECT decks FROM col LIMIT 1").fetchone()
        except sqlite3.OperationalError as exc:
            logger.warning(f"AnkiConnector: cannot read col table: {exc}")
            return deck_map

        if not row:
            return deck_map
        try:
            decks_data = json.loads(row[0])
            for did_str, info in decks_data.items():
                deck_map[did_str] = info.get("name", "Unknown")
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.debug(f"AnkiConnector: deck metadata unavailable: {exc}")
        return deck_map

    def _extract(self, conn: sqlite3.Connection) -> Iterator[ParsedDocument]:
        """Query notes and yield one ParsedDocument per note.

        Args:
            conn: Open SQLite connection to the copied database.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        deck_map = self._load_deck_map(conn)

        try:
            rows = conn.execute("""
                SELECT DISTINCT n.id, n.flds, n.tags, n.mod, c.did
                FROM notes n
                JOIN cards c ON c.nid = n.id
            """).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(f"AnkiConnector: cannot query notes: {exc}")
            return

        source = self._cfg.source_name
        total = 0
        for row in rows:
            note_id: int = row[0]
            flds: str = row[1] or ""
            tags_str: str = (row[2] or "").strip()
            mod: int = row[3]
            did: str = str(row[4])

            fields = flds.split("\x1f")
            front = html_to_text(fields[0]) if fields else ""
            back = html_to_text(fields[1]) if len(fields) > 1 else ""

            if not front:
                continue

            deck_name = deck_map.get(did, "Unknown Deck")
            tags = [t for t in tags_str.split() if t]
            safe_deck = deck_name.replace("/", "_").replace(" ", "_")

            yield ParsedDocument(
                path=Path(f"anki://{safe_deck}/{note_id}.anki"),
                text=f"Q: {front}\nA: {back}",
                file_type=".anki",
                encoding="utf-8",
                metadata={
                    "deck": deck_name,
                    "note_id": note_id,
                    "tags": tags,
                    "modified": mod,
                    "source": source,
                },
            )
            total += 1

        logger.info(f"AnkiConnector: yielded {total} note(s)")
