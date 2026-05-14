"""Reddit public JSON API connector for MemoryMesh.

Fetches a user's posts and comments from the Reddit public JSON API
without requiring OAuth - public profile data only.

API endpoints (no key required)
--------------------------------
* Posts:    ``https://www.reddit.com/user/{username}/submitted.json``
* Comments: ``https://www.reddit.com/user/{username}/comments.json``

Features
--------
* **Pagination** - iterates via ``data.after`` until exhausted or
  ``max_items`` is reached.
* **Date filtering** - only items within ``days_past`` are included.
* **Subreddit filtering** - optionally restrict to a list of subreddits.
* **Rate-limit aware** - 1 s sleep between paginated requests (Reddit
  asks for <= 1 req/s for unauthenticated calls).
* **User-Agent** - Reddit requires a descriptive User-Agent; the
  connector sends ``MemoryMesh/0.7 (personal data sync)``.

Usage
-----
::

    connector = RedditConnector(RedditConfig(
        username="my_reddit_username",
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_API_BASE = "https://www.reddit.com/user"
_USER_AGENT = "MemoryMesh/0.7 (personal data sync)"
_PAGE_SLEEP = 1.0  # seconds between paginated requests


class RedditConfig(BaseModel):
    """Configuration for a Reddit public profile source.

    Args:
        username: Reddit username to fetch data for (no ``u/`` prefix).
        include_posts: Fetch submitted posts when ``True``.
        include_comments: Fetch comments when ``True``.
        subreddits: Restrict to these subreddit names (case-insensitive).
            An empty list indexes all subreddits.
        days_past: Only include items within this many days.  0 = no
            cutoff.
        max_items: Maximum combined posts + comments to fetch.  0 = no
            limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    username: str
    include_posts: bool = True
    include_comments: bool = True
    subreddits: list[str] = []
    days_past: int = 365
    max_items: int = 500
    source_name: str = "reddit"


class RedditConnector:
    """Fetches Reddit posts and comments and yields ParsedDocuments.

    Args:
        config: Username, filters, and source settings.
    """

    def __init__(self, config: RedditConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch submitted posts and/or comments and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            post (``file_type=".reddit"``, path
            ``reddit://posts/{id}.reddit``) and one per comment (path
            ``reddit://comments/{id}.reddit``).
        """
        headers = {"User-Agent": _USER_AGENT}
        cutoff = self._cutoff()
        total = 0

        if self._cfg.include_posts:
            for doc in self._fetch_listing("submitted", headers, cutoff):
                yield doc
                total += 1

        if self._cfg.include_comments:
            for doc in self._fetch_listing("comments", headers, cutoff):
                yield doc
                total += 1

        logger.info(f"RedditConnector: yielded {total} document(s)")

    def _cutoff(self) -> float | None:
        """Return the Unix epoch cutoff for item date filtering.

        Returns:
            Float epoch seconds, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return (datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)).timestamp()

    def _fetch_listing(
        self,
        kind: str,
        headers: dict[str, str],
        cutoff: float | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate one Reddit listing endpoint and yield documents.

        Args:
            kind: Listing type - ``"submitted"`` for posts or
                ``"comments"`` for comments.
            headers: HTTP request headers (must include User-Agent).
            cutoff: Minimum ``created_utc`` Unix timestamp, or ``None``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        base_url = f"{_API_BASE}/{self._cfg.username}/{kind}.json"
        after: str | None = None
        yielded = 0
        limit = self._cfg.max_items
        subreddits = {s.lower() for s in self._cfg.subreddits}
        first_req = True

        while True:
            if limit > 0 and yielded >= limit:
                break

            params: dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after
            url = base_url + "?" + urllib.parse.urlencode(params)

            if not first_req:
                time.sleep(_PAGE_SLEEP)
            first_req = False

            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            listing = data.get("data", {})
            children = listing.get("children", [])
            if not children:
                break

            for child in children:
                if limit > 0 and yielded >= limit:
                    break
                item = child.get("data", {})
                created = float(item.get("created_utc", 0) or 0)
                if cutoff is not None and created < cutoff:
                    continue
                sub = (item.get("subreddit") or "").lower()
                if subreddits and sub not in subreddits:
                    continue
                doc = self._build_post(item) if kind == "submitted" else self._build_comment(item)
                if doc is not None:
                    yield doc
                    yielded += 1

            after = listing.get("after")
            if not after:
                break

    def _build_post(self, item: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Reddit post JSON object to a ParsedDocument.

        Args:
            item: Raw Reddit post ``data`` dict.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the post ID is missing.
        """
        post_id = item.get("id")
        if not post_id:
            return None

        title = item.get("title", "")
        selftext = item.get("selftext", "")
        subreddit = item.get("subreddit", "")
        score = int(item.get("score", 0) or 0)
        created_utc = float(item.get("created_utc", 0) or 0)
        url = item.get("url", "")

        return ParsedDocument(
            path=Path(f"reddit://posts/{post_id}.reddit"),
            text=f"r/{subreddit}\n{title}\n\n{selftext}",
            file_type=".reddit",
            encoding="utf-8",
            metadata={
                "post_id": post_id,
                "subreddit": subreddit,
                "title": title,
                "score": score,
                "created_utc": created_utc,
                "url": url,
                "source": self._cfg.source_name,
            },
        )

    def _build_comment(self, item: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Reddit comment JSON object to a ParsedDocument.

        Args:
            item: Raw Reddit comment ``data`` dict.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the comment ID is missing.
        """
        comment_id = item.get("id")
        if not comment_id:
            return None

        subreddit = item.get("subreddit", "")
        link_title = item.get("link_title", "")
        body = item.get("body", "")
        score = int(item.get("score", 0) or 0)
        created_utc = float(item.get("created_utc", 0) or 0)

        return ParsedDocument(
            path=Path(f"reddit://comments/{comment_id}.reddit"),
            text=f"r/{subreddit} -> {link_title}\n\n{body}",
            file_type=".reddit",
            encoding="utf-8",
            metadata={
                "comment_id": comment_id,
                "subreddit": subreddit,
                "post_title": link_title,
                "score": score,
                "created_utc": created_utc,
                "source": self._cfg.source_name,
            },
        )
