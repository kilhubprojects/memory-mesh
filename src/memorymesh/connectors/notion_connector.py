"""Notion API connector for MemoryMesh.

Fetches pages from Notion workspaces and yields each page as a
:class:`~memorymesh.core.models.ParsedDocument`.

API reference: https://developers.notion.com/reference

Features
--------
* **Flexible source** - index specific databases, explicit pages, or every
  page visible to the integration via the search API.
* **Block content** - paragraph, heading, list, code, quote, and callout
  blocks are converted to Markdown-ish plain text.
* **Pagination** - both page-listing and block-fetching calls are fully
  paginated.
* **Rate-limit aware** - sleeps 333 ms between requests (~3 req/s limit).
* **Privacy** - page content is never logged at INFO level.

Usage
-----
::

    connector = NotionConnector(NotionConfig(
        api_key=SecretStr("secret_abc..."),
        database_ids=["abc123"],
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._http import api_get, api_post
from memorymesh.core.models import ParsedDocument

_BASE = "https://api.notion.com/v1"
_VERSION = "2022-06-28"
_RATE_SLEEP = 0.333  # seconds between requests (Notion: ~3 req/s)


class NotionConfig(BaseModel):
    """Configuration for a Notion workspace source.

    Args:
        api_key: Notion integration token (``secret_...``).
        database_ids: Fetch all pages from these Notion database IDs.  An
            empty list skips database queries.
        page_ids: Fetch these specific page IDs directly.
        max_pages: Maximum total pages to yield per run.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    database_ids: list[str] = []
    page_ids: list[str] = []
    max_pages: int = 500
    source_name: str = "notion"


_TEXT_BLOCK_TYPES = frozenset(
    {
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "quote",
        "callout",
    }
)


def _rich_text(block_data: dict[str, Any]) -> str:
    """Concatenate ``plain_text`` from a Notion rich_text array.

    Args:
        block_data: The type-specific sub-dict of a Notion block.

    Returns:
        Plain text string (may be empty).
    """
    return "".join(item.get("plain_text", "") for item in block_data.get("rich_text", []))


def _block_to_text(block: dict[str, Any]) -> str:
    """Convert a single Notion block to a plain-text / Markdown line.

    Supported types: paragraph, heading_1/2/3, bulleted_list_item,
    numbered_list_item, code, quote, callout.  Unsupported types return an
    empty string.

    Args:
        block: A Notion block dict from the ``/blocks/{id}/children`` API.

    Returns:
        Text representation, or empty string for unsupported block types.
    """
    btype = block.get("type", "")
    bd = block.get(btype, {})

    if btype in _TEXT_BLOCK_TYPES:
        text = _rich_text(bd)
        if btype == "heading_1":
            return f"# {text}"
        if btype == "heading_2":
            return f"## {text}"
        if btype == "heading_3":
            return f"### {text}"
        if btype == "bulleted_list_item":
            return f"- {text}"
        if btype == "numbered_list_item":
            return f"1. {text}"
        if btype in ("quote", "callout"):
            return f"> {text}"
        return text

    if btype == "code":
        lang = bd.get("language", "")
        code = _rich_text(bd)
        return f"```{lang}\n{code}\n```"

    return ""


def _page_title(page: dict[str, Any]) -> str:
    """Extract the human-readable title from a Notion page object.

    Args:
        page: A Notion page dict (from search, database query, or GET page).

    Returns:
        Title string, or ``"Untitled"`` when no title property is found.
    """
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(item.get("plain_text", "") for item in prop.get("title", []))
    return "Untitled"


class NotionConnector:
    """Fetches Notion pages and yields them as ParsedDocuments.

    Args:
        config: Notion API credentials and source settings.
    """

    def __init__(self, config: NotionConfig) -> None:
        self._cfg = config
        self._headers = {
            **bearer_header(config.api_key.get_secret_value()),
            "Notion-Version": _VERSION,
        }

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Yield all configured Notion pages as ParsedDocuments.

        The order of precedence is: database pages -> explicit page IDs ->
        workspace search (only when both database_ids and page_ids are empty).

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per Notion
            page, with ``file_type=".notion"`` and metadata containing
            ``page_id``, ``title``, ``created_time``, ``last_edited_time``,
            and ``url``.
        """
        yielded = 0

        for db_id in self._cfg.database_ids:
            for page in self._query_database(db_id):
                if yielded >= self._cfg.max_pages:
                    return
                doc = self._page_to_document(page)
                if doc is not None:
                    yield doc
                    yielded += 1

        for page_id in self._cfg.page_ids:
            if yielded >= self._cfg.max_pages:
                return
            fetched_page: dict[str, Any] | None = self._get_page(page_id)
            if fetched_page is not None:
                doc = self._page_to_document(fetched_page)
                if doc is not None:
                    yield doc
                    yielded += 1

        if not self._cfg.database_ids and not self._cfg.page_ids:
            for page in self._search_pages():
                if yielded >= self._cfg.max_pages:
                    return
                doc = self._page_to_document(page)
                if doc is not None:
                    yield doc
                    yielded += 1

        logger.info(f"NotionConnector: yielded {yielded} page(s)")

    def _query_database(self, db_id: str) -> Iterator[dict[str, Any]]:
        """Paginate through all pages in a Notion database.

        Args:
            db_id: Notion database ID.

        Yields:
            Notion page dict for each page in the database.
        """
        url = f"{_BASE}/databases/{db_id}/query"
        payload: dict[str, Any] = {"page_size": 100}

        while True:
            time.sleep(_RATE_SLEEP)
            data = api_post(url, self._headers, payload)
            if not data or not isinstance(data, dict):
                break
            yield from data.get("results", [])
            if not data.get("has_more"):
                break
            payload["start_cursor"] = data.get("next_cursor", "")

    def _search_pages(self) -> Iterator[dict[str, Any]]:
        """Paginate through all pages accessible to the integration.

        Yields:
            Notion page dict from the search results.
        """
        url = f"{_BASE}/search"
        payload: dict[str, Any] = {
            "filter": {"value": "page", "property": "object"},
            "page_size": 100,
        }

        while True:
            time.sleep(_RATE_SLEEP)
            data = api_post(url, self._headers, payload)
            if not data or not isinstance(data, dict):
                break
            yield from data.get("results", [])
            if not data.get("has_more"):
                break
            payload["start_cursor"] = data.get("next_cursor", "")

    def _get_page(self, page_id: str) -> dict[str, Any] | None:
        """Fetch a single Notion page by ID.

        Args:
            page_id: Notion page UUID.

        Returns:
            Page dict, or ``None`` on error.
        """
        time.sleep(_RATE_SLEEP)
        return api_get(f"{_BASE}/pages/{page_id}", self._headers)

    def _fetch_blocks(self, page_id: str) -> list[str]:
        """Fetch all block children of a page and convert to text lines.

        Paginates automatically.  Unsupported block types are silently skipped.

        Args:
            page_id: Notion page UUID.

        Returns:
            List of text lines (one per block), empty strings removed.
        """
        lines: list[str] = []
        cursor: str | None = None

        while True:
            url = f"{_BASE}/blocks/{page_id}/children?page_size=100"
            if cursor:
                url += f"&start_cursor={cursor}"
            time.sleep(_RATE_SLEEP)
            data = api_get(url, self._headers)
            if not data or not isinstance(data, dict):
                break
            for block in data.get("results", []):
                line = _block_to_text(block)
                if line:
                    lines.append(line)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

        return lines

    def _page_to_document(self, page: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Notion page (with blocks) to a ParsedDocument.

        Args:
            page: Notion page dict.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if
            the page has no usable content.
        """
        page_id = page.get("id", "")
        if not page_id:
            return None

        title = _page_title(page)
        block_lines = self._fetch_blocks(page_id)
        body = "\n".join(block_lines)
        text = f"# {title}\n\n{body}".strip()

        if not text or text == f"# {title}":
            text = f"# {title}"

        return ParsedDocument(
            path=Path(f"notion://{page_id}.notion"),
            text=text,
            file_type=".notion",
            encoding="utf-8",
            metadata={
                "page_id": page_id,
                "title": title,
                "created_time": page.get("created_time", ""),
                "last_edited_time": page.get("last_edited_time", ""),
                "url": page.get("url", ""),
                "source": self._cfg.source_name,
            },
        )
