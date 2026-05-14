"""HackerNews Algolia API connector for MemoryMesh.

Fetches a user's stories and comments from the HackerNews Algolia search
API (no API key required) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per item.

API reference
-------------
``https://hn.algolia.com/api/v1/``

Features
--------
* **No API key** - the Algolia HN API is fully public.
* **Pagination** - iterates via ``page`` and ``nbPages`` fields.
* **Date filtering** - only items within ``days_past`` are included.
* **HTML stripping** - comment body HTML is stripped via the shared
  :func:`memorymesh.connectors._html.html_to_text` helper.
* **Max items** - hard cap to avoid runaway fetches on prolific authors.

Usage
-----
::

    connector = HackerNewsConnector(HackerNewsConfig(
        username="pg",
        days_past=365,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._html import html_to_text
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_API_BASE = "https://hn.algolia.com/api/v1"


class HackerNewsConfig(BaseModel):
    """Configuration for a HackerNews user content source.

    Args:
        username: HackerNews username to fetch submissions and comments for.
        include_stories: Fetch story submissions when ``True``.
        include_comments: Fetch comments when ``True``.
        days_past: Only include items within this many days.  0 = no
            cutoff.
        max_items: Maximum combined stories + comments to fetch.  0 = no
            limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    username: str
    include_stories: bool = True
    include_comments: bool = True
    days_past: int = 365
    max_items: int = 500
    source_name: str = "hackernews"


class HackerNewsConnector:
    """Fetches HN stories and comments and yields ParsedDocuments.

    Args:
        config: Username, filters, and source settings.
    """

    def __init__(self, config: HackerNewsConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch stories and/or comments and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            story (``file_type=".hn"``, path
            ``hackernews://stories/{id}.hn``) and one per comment (path
            ``hackernews://comments/{id}.hn``).
        """
        cutoff = self._cutoff()
        total = 0

        if self._cfg.include_stories:
            for doc in self._fetch_items("story", cutoff):
                yield doc
                total += 1

        if self._cfg.include_comments:
            for doc in self._fetch_items("comment", cutoff):
                yield doc
                total += 1

        logger.info(f"HackerNewsConnector: yielded {total} document(s)")

    def _cutoff(self) -> float | None:
        """Return the Unix epoch cutoff for item date filtering.

        Returns:
            Float epoch seconds, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return (datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)).timestamp()

    def _fetch_items(
        self,
        kind: str,
        cutoff: float | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate the Algolia HN search API for one content type.

        Args:
            kind: ``"story"`` or ``"comment"``.
            cutoff: Minimum ``created_at_i`` Unix timestamp, or ``None``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        username = self._cfg.username
        tag = "story" if kind == "story" else "comment"
        page = 0
        yielded = 0
        limit = self._cfg.max_items

        while True:
            if limit > 0 and yielded >= limit:
                break

            params: dict[str, Any] = {
                "tags": f"{tag},author_{username}",
                "hitsPerPage": 50,
                "page": page,
            }
            url = f"{_API_BASE}/search?" + urllib.parse.urlencode(params)
            data = api_get(url, {})
            if not isinstance(data, dict):
                break

            hits = data.get("hits", [])
            nb_pages = int(data.get("nbPages", 1))

            if not hits:
                break

            for hit in hits:
                if limit > 0 and yielded >= limit:
                    break
                created_at_i = float(hit.get("created_at_i", 0) or 0)
                if cutoff is not None and created_at_i < cutoff:
                    continue
                doc = self._build_story(hit) if kind == "story" else self._build_comment(hit)
                if doc is not None:
                    yield doc
                    yielded += 1

            page += 1
            if page >= nb_pages:
                break

    def _build_story(self, hit: dict[str, Any]) -> ParsedDocument | None:
        """Convert an Algolia HN story hit to a ParsedDocument.

        Args:
            hit: Raw Algolia hit dict for a story.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the ``objectID`` field is missing.
        """
        object_id = hit.get("objectID")
        if not object_id:
            return None

        title = hit.get("title", "")
        url = hit.get("url", "")
        points = int(hit.get("points", 0) or 0)
        num_comments = int(hit.get("num_comments", 0) or 0)
        story_text = html_to_text(hit.get("story_text") or "")
        created_at = hit.get("created_at", "")

        text = f"HN: {title}\nURL: {url}\nPoints: {points}\nComments: {num_comments}"
        if story_text:
            text += f"\n\n{story_text}"

        return ParsedDocument(
            path=Path(f"hackernews://stories/{object_id}.hn"),
            text=text,
            file_type=".hn",
            encoding="utf-8",
            metadata={
                "object_id": object_id,
                "title": title,
                "url": url,
                "points": points,
                "created_at": created_at,
                "source": self._cfg.source_name,
            },
        )

    def _build_comment(self, hit: dict[str, Any]) -> ParsedDocument | None:
        """Convert an Algolia HN comment hit to a ParsedDocument.

        Args:
            hit: Raw Algolia hit dict for a comment.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the comment has no text or is missing ``objectID``.
        """
        object_id = hit.get("objectID")
        if not object_id:
            return None

        story_title = hit.get("story_title", "")
        story_url = hit.get("story_url", "")
        comment_text = html_to_text(hit.get("comment_text") or "")
        points = int(hit.get("points", 0) or 0)
        created_at = hit.get("created_at", "")

        if not comment_text:
            return None

        return ParsedDocument(
            path=Path(f"hackernews://comments/{object_id}.hn"),
            text=f"Comment on: {story_title}\n\n{comment_text}",
            file_type=".hn",
            encoding="utf-8",
            metadata={
                "object_id": object_id,
                "story_title": story_title,
                "story_url": story_url,
                "points": points,
                "created_at": created_at,
                "source": self._cfg.source_name,
            },
        )
