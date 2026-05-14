"""Readwise highlights connector for MemoryMesh.

Fetches book highlights from the Readwise API (v2) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per source document
(book, article, tweet, etc.) containing all its highlights.

API reference: https://readwise.io/api/v2/

Features
--------
* **Incremental sync** - the latest highlight ``updated`` timestamp is
  persisted to a JSON checkpoint file.  On the next run only new/updated
  highlights are fetched via ``updated__gt``.
* **Source-type filter** - optionally restrict to ``"books"``, ``"articles"``,
  ``"tweets"``, or ``"podcasts"`` categories.
* **Pagination** - iterates the ``next`` URL returned by the API.
* **Privacy** - highlight text is never logged at INFO level.

Checkpoint file
---------------
Stored at ``checkpoint_path`` (default:
``~/.memorymesh/readwise_checkpoint.json``).  Contains:
``{"updated_after": "2024-01-15T10:00:00.000000Z"}``.

Usage
-----
::

    connector = ReadwiseConnector(ReadwiseConfig(
        api_key=SecretStr("..."),
        source_types=["books", "articles"],
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import urllib.parse
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://readwise.io/api/v2"


class ReadwiseConfig(BaseModel):
    """Configuration for a Readwise highlights source.

    Args:
        api_key: Readwise API token.
        source_types: Restrict to these Readwise categories.  An empty list
            means all categories are indexed.  Valid values:
            ``"books"``, ``"articles"``, ``"tweets"``, ``"podcasts"``.
        updated_after: ISO 8601 timestamp - only fetch highlights updated
            after this point.  Overridden by the checkpoint file when
            available.
        checkpoint_path: Path to the JSON checkpoint file.  ``None`` uses
            ``~/.memorymesh/readwise_checkpoint.json``.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    source_types: list[str] = []
    updated_after: str | None = None
    checkpoint_path: Path | None = None
    source_name: str = "readwise"


class ReadwiseConnector:
    """Fetches Readwise highlights and yields one document per source book.

    Args:
        config: Readwise API credentials and filter settings.
    """

    def __init__(self, config: ReadwiseConfig) -> None:
        self._cfg = config
        self._headers = {"Authorization": f"Token {config.api_key.get_secret_value()}"}
        checkpoint_path = config.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = Path.home() / ".memorymesh" / "readwise_checkpoint.json"
        self._checkpoint_path = checkpoint_path

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch highlights and yield one ParsedDocument per source book.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per book /
            article / tweet / podcast, with ``file_type=".readwise"`` and
            metadata containing ``book_id``, ``title``, ``author``,
            ``category``, ``source_url``, and ``highlight_count``.
        """
        updated_after = self._load_checkpoint() or self._cfg.updated_after

        # Collect all highlights grouped by book_id
        books: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        latest_ts: str | None = None

        for highlight in self._fetch_highlights(updated_after):
            book_id = highlight.get("book_id")
            if not isinstance(book_id, int):
                continue
            ts = highlight.get("updated", "")
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts
            books[book_id].append(highlight)

        logger.info(
            f"ReadwiseConnector: {sum(len(v) for v in books.values())}"
            f" highlight(s) across {len(books)} book(s)"
        )

        for book_id, highlights in books.items():
            book = self._fetch_book(book_id)
            if book is None:
                continue

            category = book.get("category", "")
            if self._cfg.source_types and category not in self._cfg.source_types:
                continue

            doc = self._build_document(book, highlights)
            if doc is not None:
                yield doc

        if latest_ts:
            self._save_checkpoint(latest_ts)

    def _fetch_highlights(self, updated_after: str | None) -> Iterator[dict[str, Any]]:
        """Paginate through all highlights from the Readwise API.

        Args:
            updated_after: Optional ISO 8601 timestamp to restrict results.

        Yields:
            Readwise highlight dict.
        """
        url = f"{_BASE}/highlights/?page_size=1000"
        if updated_after:
            url += f"&updated__gt={urllib.parse.quote(updated_after)}"

        while url:
            data = api_get(url, self._headers)
            if not isinstance(data, dict):
                break
            yield from data.get("results", [])
            next_url = data.get("next")
            url = next_url if isinstance(next_url, str) else ""

    def _fetch_book(self, book_id: int) -> dict[str, Any] | None:
        """Fetch a single book / source document by ID.

        Args:
            book_id: Readwise book ID.

        Returns:
            Book dict, or ``None`` on error.
        """
        return api_get(f"{_BASE}/books/{book_id}/", self._headers)

    def _build_document(
        self,
        book: dict[str, Any],
        highlights: list[dict[str, Any]],
    ) -> ParsedDocument | None:
        """Assemble a ParsedDocument from a book and its highlights.

        Args:
            book: Readwise book dict.
            highlights: All highlights belonging to this book.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if
            there is no usable content.
        """
        if not highlights:
            return None

        book_id = book.get("id", 0)
        title = book.get("title", "") or "Untitled"
        author = book.get("author", "") or ""
        category = book.get("category", "") or ""
        source_url = book.get("source_url", "") or ""

        highlight_texts = [(h.get("text", "") or "").strip() for h in highlights]
        highlight_texts = [t for t in highlight_texts if t]

        if not highlight_texts:
            return None

        header_lines = [f"# {title}"]
        if author:
            header_lines.append(f"Author: {author}")
        if category:
            header_lines.append(f"Category: {category}")
        header = "\n".join(header_lines)
        body = "\n---\n".join(highlight_texts)
        text = f"{header}\n\n{body}"

        return ParsedDocument(
            path=Path(f"readwise://{book_id}.readwise"),
            text=text,
            file_type=".readwise",
            encoding="utf-8",
            metadata={
                "book_id": book_id,
                "title": title,
                "author": author,
                "category": category,
                "source_url": source_url,
                "highlight_count": len(highlight_texts),
                "source": self._cfg.source_name,
            },
        )

    def _load_checkpoint(self) -> str | None:
        """Load the latest ``updated_after`` timestamp from the checkpoint.

        Returns:
            ISO 8601 timestamp string, or ``None`` if no checkpoint exists.
        """
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            return data.get("updated_after")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    def _save_checkpoint(self, updated_after: str) -> None:
        """Persist *updated_after* timestamp to the checkpoint file.

        Args:
            updated_after: ISO 8601 timestamp of the latest synced highlight.
        """
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path.write_text(
                json.dumps({"updated_after": updated_after}),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                f"ReadwiseConnector: cannot save checkpoint {self._checkpoint_path}: {exc}"
            )
