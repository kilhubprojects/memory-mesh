"""Zotero local database connector for MemoryMesh.

Reads the local Zotero SQLite database (``zotero.sqlite``) and yields
research items - journal articles, books, conference papers, theses, and
similar - as :class:`~memorymesh.core.models.ParsedDocument` objects.

Features
--------
* **Auto-detection** - defaults to ``~/Zotero/zotero.sqlite`` on all platforms.
* **Safe read** - copies the database to a temp path before opening so that a
  running Zotero instance is never locked out.
* **Notes** - optionally appends Zotero child notes to the item text.
* **Stdlib only** - uses only ``sqlite3``, ``shutil``, and ``tempfile``.

Usage
-----
::

    connector = ZoteroConnector(ZoteroConfig(
        include_notes=True,
        include_abstracts=True,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._html import html_to_text
from memorymesh.core.models import ParsedDocument

_SUPPORTED_TYPES: frozenset[str] = frozenset(
    {
        "journalArticle",
        "book",
        "bookSection",
        "conferencePaper",
        "thesis",
        "report",
        "preprint",
        "manuscript",
        "document",
    }
)


class ZoteroConfig(BaseModel):
    """Configuration for the Zotero local database connector.

    Args:
        db_path: Path to ``zotero.sqlite``.  ``None`` = auto-detect from
            ``~/Zotero/zotero.sqlite``.
        include_notes: Whether to append child notes to item text.
        include_abstracts: Whether to include abstracts in item text.
        source_name: Name used in the MemoryMesh source registry.
    """

    db_path: Path | None = None
    include_notes: bool = True
    include_abstracts: bool = True
    source_name: str = "zotero"


def _default_db_path() -> Path:
    """Return the default Zotero database path (``~/Zotero/zotero.sqlite``).

    Returns:
        Expanded absolute :class:`~pathlib.Path`.
    """
    return Path.home() / "Zotero" / "zotero.sqlite"


def _safe_html(raw: str) -> str:
    """Strip HTML tags from *raw* if it looks like HTML.

    Args:
        raw: Possibly-HTML string.

    Returns:
        Plain text string.
    """
    if "<" in raw and ">" in raw:
        return html_to_text(raw)
    return raw.strip()


def _safe_key(key: str) -> str:
    """Sanitise an item key for use in a synthetic path.

    Args:
        key: Zotero item key (typically 8 alphanumeric characters).

    Returns:
        Sanitised string.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", key)[:32]


_SQL_ITEMS = """
    SELECT i.itemID, i.key, it.typeName
    FROM items i
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    WHERE it.typeName IN ({placeholders})
    ORDER BY i.itemID
"""

_SQL_FIELDS = """
    SELECT f.fieldName, idv.value
    FROM itemData id
    JOIN fields f ON id.fieldID = f.fieldID
    JOIN itemDataValues idv ON id.valueID = idv.valueID
    WHERE id.itemID = ?
"""

_SQL_CREATORS = """
    SELECT c.firstName, c.lastName, ct.creatorType
    FROM itemCreators ic
    JOIN creators c ON ic.creatorID = c.creatorID
    JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
    WHERE ic.itemID = ?
    ORDER BY ic.orderIndex
"""

_SQL_NOTES = """
    SELECT note
    FROM itemNotes
    WHERE sourceItemID = ?
    AND note IS NOT NULL
    AND note != ''
"""


class ZoteroConnector:
    """Reads the local Zotero database and yields research items as ParsedDocuments.

    Args:
        config: Path and fetch settings.
    """

    def __init__(self, config: ZoteroConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Open the Zotero database and yield each supported item as a ParsedDocument.

        Opens a *copy* of the database to avoid conflicting with a running
        Zotero instance.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per item,
            with ``file_type=".zotero"`` and metadata containing ``title``,
            ``authors``, ``year``, ``doi``, ``item_type``, and ``item_key``.
        """
        db_path = self._cfg.db_path or _default_db_path()
        if not db_path.exists():
            logger.warning(
                f"ZoteroConnector: database not found at {db_path}. "
                "Set db_path in config or ensure Zotero is installed."
            )
            return

        logger.info(f"ZoteroConnector: reading {db_path}")

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            shutil.copy2(db_path, tmp_path)
        except OSError as exc:
            logger.warning(f"ZoteroConnector: cannot copy database: {exc}")
            tmp_path.unlink(missing_ok=True)
            return

        try:
            yield from self._query_items(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _query_items(self, db_path: Path) -> Iterator[ParsedDocument]:
        """Query all supported items from the copied database.

        Args:
            db_path: Path to the (copied) Zotero SQLite file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per item.
        """
        placeholders = ",".join("?" * len(_SUPPORTED_TYPES))
        sql_items = _SQL_ITEMS.format(placeholders=placeholders)

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError as exc:
            logger.warning(f"ZoteroConnector: cannot open database: {exc}")
            return

        yielded = 0
        try:
            rows = conn.execute(sql_items, list(_SUPPORTED_TYPES)).fetchall()
            logger.info(f"ZoteroConnector: {len(rows)} item(s) found")

            for row in rows:
                item_id = row["itemID"]
                item_key = row["key"]
                item_type = row["typeName"]

                try:
                    doc = self._build_document(conn, item_id, item_key, item_type)
                except Exception as exc:
                    logger.warning(f"ZoteroConnector: error processing item {item_key!r}: {exc}")
                    continue

                if doc is not None:
                    yield doc
                    yielded += 1

        except sqlite3.Error as exc:
            logger.warning(f"ZoteroConnector: query error: {exc}")
        finally:
            conn.close()

        logger.info(f"ZoteroConnector: yielded {yielded} item(s)")

    def _build_document(
        self,
        conn: sqlite3.Connection,
        item_id: int,
        item_key: str,
        item_type: str,
    ) -> ParsedDocument | None:
        """Assemble a ParsedDocument from a single Zotero item.

        Args:
            conn: Open (read-only) SQLite connection.
            item_id: Zotero internal item ID.
            item_key: Zotero item key (e.g. ``"ABCD1234"``).
            item_type: Item type name (e.g. ``"journalArticle"``).

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if
            the item has no title.
        """
        # Collect all data fields for this item
        fields: dict[str, str] = {}
        for frow in conn.execute(_SQL_FIELDS, (item_id,)).fetchall():
            fields[frow["fieldName"]] = frow["value"] or ""

        title = fields.get("title", "").strip()
        if not title:
            return None

        abstract = _safe_html(fields.get("abstractNote", "")).strip()
        year = fields.get("date", "")[:4].strip() if fields.get("date") else ""
        doi = fields.get("DOI", "").strip()
        publication = fields.get("publicationTitle", fields.get("publisher", "")).strip()

        # Collect creators
        creator_rows = conn.execute(_SQL_CREATORS, (item_id,)).fetchall()
        authors: list[str] = []
        for crow in creator_rows:
            first = (crow["firstName"] or "").strip()
            last = (crow["lastName"] or "").strip()
            full = f"{first} {last}".strip() if first else last
            if full and crow["creatorType"] == "author":
                authors.append(full)

        # Collect notes
        note_texts: list[str] = []
        if self._cfg.include_notes:
            for nrow in conn.execute(_SQL_NOTES, (item_id,)).fetchall():
                raw_note = nrow[0] or ""
                cleaned = _safe_html(raw_note).strip()
                if cleaned:
                    note_texts.append(cleaned)

        # Assemble text
        parts: list[str] = [f"Title: {title}"]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if year:
            parts.append(f"Year: {year}")
        if publication:
            parts.append(f"Publication: {publication}")
        if doi:
            parts.append(f"DOI: {doi}")
        if self._cfg.include_abstracts and abstract:
            parts.append("")
            parts.append(abstract)
        if note_texts:
            parts.append("")
            parts.append("Notes:")
            for note in note_texts:
                parts.append(note)

        text = "\n".join(parts)
        source = self._cfg.source_name
        synthetic_path = Path(f"zotero://{source}/{_safe_key(item_key)}.zotero")

        return ParsedDocument(
            path=synthetic_path,
            text=text,
            file_type=".zotero",
            encoding="utf-8",
            metadata={
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "item_type": item_type,
                "item_key": item_key,
                "source": source,
            },
        )
