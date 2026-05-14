"""RSS/Atom feed connector for MemoryMesh.

Fetches one or more RSS or Atom feeds and yields each article as a
:class:`~memorymesh.core.models.ParsedDocument` so it can be searched like any
other personal knowledge.

Requires the ``feedparser`` package (optional dependency)::

    uv add 'memorymesh-mcp[rss]'
    # or directly:
    uv add feedparser

Features
--------
* **Multi-feed** - accepts a list of feed URLs in config.
* **Recency filter** - ``days_past`` limits articles to recent content.
* **HTML stripping** - article summaries and content are converted to plain
  text via the shared :func:`~memorymesh.connectors._html.html_to_text` utility.
* **Privacy** - article content is never logged at INFO level.

Usage
-----
::

    connector = RSSConnector(RSSConfig(
        feeds=["https://example.com/feed.xml"],
        days_past=30,
        max_items_per_feed=50,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import calendar
import hashlib
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._html import html_to_text
from memorymesh.core.models import ParsedDocument


class RSSConfig(BaseModel):
    """Configuration for one or more RSS/Atom feed sources.

    Args:
        feeds: List of feed URLs to fetch.
        max_items_per_feed: Maximum articles to yield per feed (0 = no limit).
        days_past: Only yield articles published within this many days (0 = all).
        source_name: Name used in the MemoryMesh source registry.
    """

    feeds: list[str]
    max_items_per_feed: int = 50
    days_past: int = 30
    source_name: str = "rss"


def _entry_hash(entry_id: str) -> str:
    """Return a short stable hash of *entry_id* for synthetic paths.

    Args:
        entry_id: Feed entry ``id`` attribute (URL or GUID).

    Returns:
        16-character hexadecimal string.
    """
    return hashlib.md5(entry_id.encode(), usedforsecurity=False).hexdigest()[:16]


def _feed_domain(feed_url: str) -> str:
    """Extract the hostname from *feed_url* for use in synthetic paths.

    Args:
        feed_url: RSS/Atom feed URL.

    Returns:
        Hostname string, or ``"unknown"`` if parsing fails.
    """
    try:
        return urllib.parse.urlparse(feed_url).netloc or "unknown"
    except Exception:
        return "unknown"


def _entry_text(entry: Any) -> str:
    """Extract plain text from a feedparser entry.

    Checks ``content``, ``summary``, and ``title`` in that preference order.
    HTML is stripped to plain text.

    Args:
        entry: A feedparser entry dict-like object.

    Returns:
        Plain text string (may be empty).
    """
    # ``content`` is a list of content objects; prefer text/html
    for content_item in getattr(entry, "content", []):
        value = getattr(content_item, "value", "") or ""
        if value:
            ctype = getattr(content_item, "type", "text/html")
            return html_to_text(value) if "html" in ctype else value.strip()

    summary = getattr(entry, "summary", "") or ""
    if summary:
        return html_to_text(summary)

    return ""


def _entry_published_ts(entry: Any) -> float | None:
    """Return the published timestamp of *entry* as a Unix float, or ``None``.

    Args:
        entry: A feedparser entry.

    Returns:
        Unix timestamp or ``None`` if no date is available.
    """
    # feedparser exposes ``published_parsed`` as a ``time.struct_time`` in UTC
    pt = getattr(entry, "published_parsed", None)
    if pt is None:
        pt = getattr(entry, "updated_parsed", None)
    if pt is None:
        return None
    try:
        return float(calendar.timegm(pt))
    except Exception:
        return None


class RSSConnector:
    """Fetches RSS/Atom feeds and yields articles as ParsedDocuments.

    Args:
        config: Feed URLs and fetch settings.
    """

    def __init__(self, config: RSSConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch all configured feeds and yield articles as ParsedDocuments.

        Requires the ``feedparser`` package.  If not installed, logs a warning
        and yields nothing.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per article,
            with ``file_type=".rss"`` and metadata containing ``url``,
            ``title``, ``published_ts``, and ``feed_url``.
        """
        try:
            import feedparser  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "RSSConnector: 'feedparser' package not installed. "
                "Run `uv add feedparser` to enable RSS sync."
            )
            return

        cutoff_ts: float | None = None
        if self._cfg.days_past > 0:
            cutoff_ts = time.time() - self._cfg.days_past * 86_400

        total = 0
        for feed_url in self._cfg.feeds:
            for doc in self._fetch_feed(feedparser, feed_url, cutoff_ts):
                yield doc
                total += 1

        logger.info(f"RSSConnector: yielded {total} article(s) from {len(self._cfg.feeds)} feed(s)")

    def _fetch_feed(
        self,
        feedparser: Any,
        feed_url: str,
        cutoff_ts: float | None,
    ) -> Iterator[ParsedDocument]:
        """Fetch and parse a single feed URL.

        Args:
            feedparser: The imported feedparser module.
            feed_url: URL of the RSS/Atom feed.
            cutoff_ts: Only yield entries published after this Unix timestamp.
                ``None`` means no cutoff.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per article.
        """
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:
            logger.warning(f"RSSConnector: failed to fetch {feed_url!r}: {exc}")
            return

        entries = getattr(parsed, "entries", []) or []
        domain = _feed_domain(feed_url)
        feed_title = getattr(getattr(parsed, "feed", None), "title", "") or domain

        logger.info(f"RSSConnector: {len(entries)} entries from {feed_title!r} ({domain})")

        limit = self._cfg.max_items_per_feed
        yielded = 0

        for entry in entries:
            if limit > 0 and yielded >= limit:
                break

            pub_ts = _entry_published_ts(entry)
            if cutoff_ts is not None and pub_ts is not None and pub_ts < cutoff_ts:
                continue

            doc = self._entry_to_document(entry, feed_url, domain)
            if doc is not None:
                yield doc
                yielded += 1

    def _entry_to_document(
        self,
        entry: Any,
        feed_url: str,
        domain: str,
    ) -> ParsedDocument | None:
        """Convert a feedparser entry to a ParsedDocument.

        Args:
            entry: feedparser entry object.
            feed_url: The feed URL this entry came from.
            domain: Hostname extracted from *feed_url*.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if
            the entry has no usable content.
        """
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        entry_id = getattr(entry, "id", link or title)
        body = _entry_text(entry)
        pub_ts = _entry_published_ts(entry)

        if not title.strip() and not body.strip():
            return None

        parts: list[str] = []
        if title:
            parts.append(f"Title: {title}")
        if link:
            parts.append(f"URL: {link}")
        if pub_ts is not None:
            parts.append(f"Published: {time.strftime('%Y-%m-%d', time.gmtime(pub_ts))}")
        if feed_url:
            parts.append(f"Feed: {feed_url}")
        if body:
            parts.append("")
            parts.append(body)
        text = "\n".join(parts)

        entry_hash = _entry_hash(entry_id or title)
        source = self._cfg.source_name
        synthetic_path = Path(f"rss://{domain}/{entry_hash}.rss")

        return ParsedDocument(
            path=synthetic_path,
            text=text,
            file_type=".rss",
            encoding="utf-8",
            metadata={
                "url": link,
                "title": title,
                "published_ts": pub_ts,
                "feed_url": feed_url,
                "source": source,
            },
        )
