"""Pocket saved-articles connector for MemoryMesh.

Fetches saved articles from Pocket using the Pocket v3 API and yields
one :class:`~memorymesh.core.models.ParsedDocument` per saved item.

API reference
-------------
``https://getpocket.com/v3/get``

Authentication
--------------
Pocket uses a consumer-key + access-token pair.  Obtain credentials:

1. Create an app at https://getpocket.com/developer/apps/new
2. Follow the OAuth flow (or use a CLI tool) to obtain an access token
   for your personal account.

Request protocol
----------------
Pocket's read endpoint uses **POST** with a JSON body even for data
retrieval, which is non-standard.  The connector uses the shared
:func:`memorymesh.connectors._http.api_post` helper.

Features
--------
* **Pagination** - offsets through results in batches of 30 until the
  API returns an empty list.
* **Date filtering** - the ``since`` parameter restricts items to those
  added after the cutoff timestamp.
* **Tag extraction** - item tags are stored in metadata as a list.
* **Read status** - ``is_read: True`` when the item has been marked as
  read (``time_read > 0``).

Usage
-----
::

    connector = PocketConnector(PocketConfig(
        consumer_key=SecretStr("12345-abcdef..."),
        access_token=SecretStr("user_access_token"),
        state="all",
        days_past=180,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_post
from memorymesh.core.models import ParsedDocument

_API_URL = "https://getpocket.com/v3/get"
_POCKET_HEADERS = {
    "X-Accept": "application/json",
}
_PAGE_SIZE = 30


class PocketConfig(BaseModel):
    """Configuration for a Pocket saved-articles source.

    Args:
        consumer_key: Pocket application consumer key.
        access_token: User-level Pocket access token.
        state: Filter saved items: ``"unread"``, ``"archive"``, or
            ``"all"`` (default).
        days_past: Only include items added within this many days.  0 =
            no cutoff.
        max_items: Maximum number of items to fetch.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    consumer_key: SecretStr
    access_token: SecretStr
    state: Literal["unread", "archive", "all"] = "all"
    days_past: int = 365
    max_items: int = 1000
    source_name: str = "pocket"


class PocketConnector:
    """Fetches Pocket saved articles and yields ParsedDocuments.

    Args:
        config: Pocket credentials, state filter, and source settings.
    """

    def __init__(self, config: PocketConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Page through the Pocket API and yield one document per item.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            saved article, with ``file_type=".pocket"`` and metadata
            containing ``item_id``, ``title``, ``url``, ``tags``,
            ``word_count``, ``time_added``, and ``is_read``.
        """
        since_ts = self._since_ts()
        offset = 0
        yielded = 0
        limit = self._cfg.max_items

        while True:
            if limit > 0 and yielded >= limit:
                break

            body: dict[str, Any] = {
                "consumer_key": self._cfg.consumer_key.get_secret_value(),
                "access_token": self._cfg.access_token.get_secret_value(),
                "state": self._cfg.state,
                "sort": "newest",
                "detailType": "complete",
                "count": _PAGE_SIZE,
                "offset": offset,
            }
            if since_ts is not None:
                body["since"] = since_ts

            data = api_post(_API_URL, _POCKET_HEADERS, body)
            if not isinstance(data, dict):
                break

            items = data.get("list", {})
            if not items:
                break

            items_values: list[Any] = list(items.values()) if isinstance(items, dict) else []
            if not items_values:
                break

            for item in items_values:
                if limit > 0 and yielded >= limit:
                    break
                doc = self._build_doc(item)
                if doc is not None:
                    yield doc
                    yielded += 1

            offset += _PAGE_SIZE

        logger.info(f"PocketConnector: yielded {yielded} document(s)")

    def _since_ts(self) -> int | None:
        """Return the Unix epoch cutoff for ``since`` parameter.

        Returns:
            Integer epoch seconds, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return int((datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)).timestamp())

    def _build_doc(self, item: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Pocket item dict to a ParsedDocument.

        Args:
            item: Raw Pocket item JSON object.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if ``item_id`` is missing.
        """
        item_id = item.get("item_id")
        if not item_id:
            return None

        title = item.get("resolved_title") or item.get("given_title") or "Untitled"
        url = item.get("resolved_url") or item.get("given_url", "")
        excerpt = item.get("excerpt", "")
        time_added = int(item.get("time_added", 0) or 0)
        time_read = int(item.get("time_read", 0) or 0)
        word_count = int(item.get("word_count", 0) or 0)
        is_read = time_read > 0

        tags_raw = item.get("tags") or {}
        tags: list[str] = list(tags_raw.keys()) if isinstance(tags_raw, dict) else []

        return ParsedDocument(
            path=Path(f"pocket://{item_id}.pocket"),
            text=f"Title: {title}\nURL: {url}\nExcerpt: {excerpt}",
            file_type=".pocket",
            encoding="utf-8",
            metadata={
                "item_id": item_id,
                "title": title,
                "url": url,
                "tags": tags,
                "word_count": word_count,
                "time_added": time_added,
                "is_read": is_read,
                "source": self._cfg.source_name,
            },
        )
