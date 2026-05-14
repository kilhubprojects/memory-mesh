"""Feedly RSS reader connector for MemoryMesh.

Fetches saved/read articles from Feedly via the Feedly Developer API and
yields one :class:`~memorymesh.core.models.ParsedDocument` per article.

API reference
-------------
``https://cloud.feedly.com/v3``

Authentication
--------------
Bearer token passed as ``Authorization: Bearer {token}``.
Generate a developer token at https://feedly.com/v3/auth/dev.

Features
--------
* **Continuation pagination** - uses the ``continuation`` field from the
  API response to iterate through all entries.
* **Stream selection** - fetches from a configurable stream ID (defaults to
  the user's ``global.all`` stream, which includes all feeds).
* **HTML stripping** - article content and summaries are stripped to plain
  text.
* **Date filtering** - articles older than ``days_past`` are skipped.

Usage
-----
::

    connector = FeedlyConnector(FeedlyConfig(
        access_token=SecretStr("my-feedly-token"),
        days_past=30,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._html import html_to_text
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://cloud.feedly.com/v3"
_PAGE_SIZE = 50


class FeedlyConfig(BaseModel):
    """Configuration for a Feedly source.

    Args:
        access_token: Feedly developer access token.
        stream_id: Feedly stream ID to fetch.  Defaults to
            ``user/{user_id}/category/global.all``.  Set to a specific
            category or feed URL for narrower scope.
        days_past: Only include articles within this many days.  0 = no
            cutoff.
        unread_only: Only fetch unread articles.
        max_articles: Maximum total articles to fetch.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    access_token: SecretStr
    stream_id: str = ""
    days_past: int = 30
    unread_only: bool = False
    max_articles: int = 500
    source_name: str = "feedly"


class FeedlyConnector:
    """Fetches Feedly articles and yields one ParsedDocument per article.

    Args:
        config: Feedly token, stream settings, and source settings.
    """

    def __init__(self, config: FeedlyConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Feedly stream and yield article documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            article, with ``file_type=".feedly"`` and metadata containing
            ``entry_id``, ``title``, ``url``, ``feed_title``,
            ``published``, and ``tags``.
        """
        token = self._cfg.access_token.get_secret_value()
        headers = bearer_header(token)
        cutoff = self._cutoff()

        stream_id = self._cfg.stream_id or self._default_stream(headers)
        if not stream_id:
            logger.warning("FeedlyConnector: could not resolve stream ID")
            return

        continuation: str | None = None
        yielded = 0
        limit = self._cfg.max_articles

        while True:
            if limit > 0 and yielded >= limit:
                break

            params: dict[str, Any] = {
                "streamId": stream_id,
                "count": _PAGE_SIZE,
            }
            if self._cfg.unread_only:
                params["unreadOnly"] = "true"
            if cutoff:
                params["newerThan"] = int(cutoff.timestamp() * 1000)
            if continuation:
                params["continuation"] = continuation

            url = f"{_BASE}/streams/contents?{urlencode(params)}"
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            items: list[dict[str, Any]] = data.get("items", [])
            if not items:
                break

            for item in items:
                if limit > 0 and yielded >= limit:
                    break
                doc = self._build_doc(item, cutoff)
                if doc is not None:
                    yield doc
                    yielded += 1

            continuation = data.get("continuation")
            if not continuation:
                break

        logger.info(f"FeedlyConnector: yielded {yielded} article(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _default_stream(self, headers: dict[str, str]) -> str:
        """Resolve the default ``global.all`` stream ID for the user.

        Args:
            headers: Auth headers.

        Returns:
            Stream ID string or empty string on failure.
        """
        data = api_get(f"{_BASE}/profile", headers)
        if not isinstance(data, dict):
            return ""
        user_id = data.get("id", "")
        if not user_id:
            return ""
        stream_id = f"user/{user_id}/category/global.all"
        return quote(stream_id, safe="")

    def _build_doc(
        self,
        item: dict[str, Any],
        cutoff: datetime | None,
    ) -> ParsedDocument | None:
        """Convert a Feedly entry to a ParsedDocument.

        Args:
            item: Raw Feedly entry dict.
            cutoff: UTC datetime cutoff; skip items published before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the item should be skipped.
        """
        entry_id = item.get("id", "")
        if not entry_id:
            return None

        published_ms = item.get("published", 0)
        if cutoff and published_ms:
            try:
                dt = datetime.fromtimestamp(published_ms / 1000, tz=UTC)
                if dt < cutoff:
                    return None
            except (OSError, OverflowError, ValueError) as exc:
                logger.debug(f"FeedlyConnector: ignoring unparsable timestamp: {exc}")

        title = item.get("title", "Untitled")
        url = (item.get("alternate") or [{}])[0].get("href", "")
        feed_title = (item.get("origin") or {}).get("title", "")
        published = item.get("published", 0)

        content_html = (item.get("content") or {}).get("content", "") or (
            item.get("summary") or {}
        ).get("content", "")
        plain_text = html_to_text(content_html) if content_html else ""

        tags: list[str] = [
            t.get("label", "")
            for t in (item.get("tags") or [])
            if isinstance(t, dict) and t.get("label")
        ]

        text_parts = [
            f"Title: {title}",
            f"Feed: {feed_title}",
            f"URL: {url}",
        ]
        if plain_text:
            text_parts.append(f"\n{plain_text[:2000]}")

        id_hash = hashlib.md5(entry_id.encode()).hexdigest()[:12]

        return ParsedDocument(
            path=Path(f"feedly://{id_hash}.feedly"),
            text="\n".join(text_parts),
            file_type=".feedly",
            encoding="utf-8",
            metadata={
                "entry_id": entry_id,
                "title": title,
                "url": url,
                "feed_title": feed_title,
                "published": published,
                "tags": tags,
                "source": self._cfg.source_name,
            },
        )
