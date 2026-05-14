"""Goodreads library export connector for MemoryMesh.

Parses the Goodreads library export CSV (My Books -> Import and Export ->
Export Library) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per book.

Export format
-------------
A CSV file (``goodreads_library_export.csv``) whose relevant columns
include: ``Book Id``, ``Title``, ``Author``, ``My Rating``,
``Exclusive Shelf``, ``Bookshelves``, ``Date Read``,
``Number of Pages``, ``Year Published``, ``My Review``.

Features
--------
* **Shelf filtering** - optionally restrict to specific shelf names
  (matched against ``Exclusive Shelf`` and ``Bookshelves``).
* **One document per book** - each book becomes its own searchable
  document including the personal review when present.

Usage
-----
::

    connector = GoodreadsConnector(GoodreadsConfig(
        export_path=Path("~/Downloads/goodreads_library_export.csv"),
        shelves=["read", "currently-reading"],
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class GoodreadsConfig(BaseModel):
    """Configuration for a Goodreads library export source.

    Args:
        export_path: Path to ``goodreads_library_export.csv``.
        shelves: Restrict to these shelf names.  An empty list indexes all
            books.  Values are compared case-insensitively against both
            ``Exclusive Shelf`` and the comma-separated ``Bookshelves``
            column.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    shelves: list[str] = []
    source_name: str = "goodreads"


class GoodreadsConnector:
    """Parses Goodreads CSV exports and yields one document per book.

    Args:
        config: Export path, shelf filter, and source settings.
    """

    def __init__(self, config: GoodreadsConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the library CSV and yield one ParsedDocument per book.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            book matching the shelf filter, with ``file_type=".goodreads"``
            and metadata containing ``book_id``, ``title``, ``author``,
            ``my_rating``, ``shelf``, ``date_read``, and
            ``year_published``.
        """
        path = self._cfg.export_path.expanduser()
        if not path.exists():
            logger.warning(f"GoodreadsConnector: file not found: {path}")
            return
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning(f"GoodreadsConnector: cannot read {path}: {exc}")
            return

        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            logger.warning(f"GoodreadsConnector: no header row in {path}")
            return

        source = self._cfg.source_name
        filter_shelves = {s.lower() for s in self._cfg.shelves}
        total = 0

        for row in reader:
            shelf = row.get("Exclusive Shelf", "").strip().lower()
            bookshelves = row.get("Bookshelves", "").strip().lower()

            if (
                filter_shelves
                and shelf not in filter_shelves
                and not any(s in bookshelves for s in filter_shelves)
            ):
                continue

            book_id = row.get("Book Id", "").strip()
            title = row.get("Title", "").strip()
            author = row.get("Author", "").strip()
            my_rating = row.get("My Rating", "0").strip()
            date_read = row.get("Date Read", "").strip()
            pages = row.get("Number of Pages", "").strip()
            my_review = row.get("My Review", "").strip()
            year_pub = row.get("Year Published", "").strip()

            text_parts = [
                f"Title: {title}",
                f"Author: {author}",
                f"Rating: {my_rating}/5",
                f"Shelf: {shelf}",
                f"Date Read: {date_read}",
                f"Pages: {pages}",
            ]
            if my_review:
                text_parts.append(f"\n{my_review}")

            if not book_id:
                book_id = str(total)

            yield ParsedDocument(
                path=Path(f"goodreads://{book_id}.goodreads"),
                text="\n".join(text_parts),
                file_type=".goodreads",
                encoding="utf-8",
                metadata={
                    "book_id": book_id,
                    "title": title,
                    "author": author,
                    "my_rating": my_rating,
                    "shelf": shelf,
                    "date_read": date_read,
                    "year_published": year_pub,
                    "source": source,
                },
            )
            total += 1

        logger.info(f"GoodreadsConnector: yielded {total} book(s)")
