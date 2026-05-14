"""Bluesky social network connector for MemoryMesh.

Fetches posts from a Bluesky account via the AT Protocol (ATP) API and
yields one :class:`~memorymesh.core.models.ParsedDocument` per post.

API reference
-------------
``https://bsky.social/xrpc/com.atproto.server.createSession``
``https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed``

Authentication
--------------
ATP session authentication: POST to ``createSession`` with
``identifier`` (handle or DID) and ``password`` to receive an
``accessJwt`` bearer token.

Features
--------
* **Cursor pagination** - uses the ``cursor`` field from API responses.
* **Repost filtering** - repost items are skipped by default.
* **Date filtering** - posts older than ``days_past`` are skipped.

Usage
-----
::

    connector = BlueSkyConnector(BlueSkyConfig(
        handle="user.bsky.social",
        password=SecretStr("app-password"),
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._http import api_get, api_post
from memorymesh.core.models import ParsedDocument

_BASE = "https://bsky.social/xrpc"
_PAGE_SIZE = 50


class BlueSkyConfig(BaseModel):
    """Configuration for a Bluesky source.

    Args:
        handle: Bluesky handle, e.g. ``user.bsky.social``.
        password: Bluesky app password (not your main password - generate
            one in Settings -> App Passwords).
        skip_reposts: Whether to skip repost/rebleet items.
        days_past: Only include posts within this many days.  0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    handle: str
    password: SecretStr
    skip_reposts: bool = True
    days_past: int = 180
    source_name: str = "bluesky"


class BlueSkyConnector:
    """Fetches Bluesky posts and yields one ParsedDocument per post.

    Args:
        config: Bluesky credentials and source settings.
    """

    def __init__(self, config: BlueSkyConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Authenticate, then paginate the author feed and yield posts.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            post, with ``file_type=".bluesky"`` and metadata containing
            ``uri``, ``cid``, ``created_at``, ``reply_count``,
            ``repost_count``, and ``like_count``.
        """
        session = self._create_session()
        if not session:
            logger.warning("BlueSkyConnector: authentication failed")
            return

        token = session.get("accessJwt", "")
        did = session.get("did", "")
        headers = bearer_header(token)
        cutoff = self._cutoff()
        cursor: str | None = None
        total = 0
        stop = False

        while not stop:
            params: dict[str, Any] = {
                "actor": did,
                "limit": _PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor

            url = f"{_BASE}/app.bsky.feed.getAuthorFeed?{urlencode(params)}"
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            feed: list[dict[str, Any]] = data.get("feed", [])
            if not feed:
                break

            for item in feed:
                reason = item.get("reason")
                if self._cfg.skip_reposts and reason is not None:
                    continue

                post_view = item.get("post") or {}
                record = post_view.get("record") or {}
                created_at = record.get("createdAt", "")

                if cutoff and created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if dt < cutoff:
                            stop = True
                            break
                    except ValueError as exc:
                        logger.debug(f"BlueskyConnector: ignoring unparsable timestamp: {exc}")

                doc = self._build_doc(post_view, record)
                if doc is not None:
                    yield doc
                    total += 1

            cursor = data.get("cursor")
            if not cursor:
                break

        logger.info(f"BlueSkyConnector: yielded {total} post(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _create_session(self) -> dict[str, Any] | None:
        """POST to ``createSession`` and return the session dict.

        Returns:
            Session dict with ``accessJwt`` and ``did``, or ``None`` on
            failure.
        """
        payload = {
            "identifier": self._cfg.handle,
            "password": self._cfg.password.get_secret_value(),
        }
        data = api_post(f"{_BASE}/com.atproto.server.createSession", {}, payload)
        return data if isinstance(data, dict) else None

    def _build_doc(
        self,
        post_view: dict[str, Any],
        record: dict[str, Any],
    ) -> ParsedDocument | None:
        """Convert a Bluesky post view to a ParsedDocument.

        Args:
            post_view: Post view dict from the feed item.
            record: Record dict nested inside the post view.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the ``uri`` field is missing.
        """
        uri = post_view.get("uri")
        if not uri:
            return None

        cid = post_view.get("cid", "")
        text = record.get("text", "")
        created_at = record.get("createdAt", "")
        reply_count = post_view.get("replyCount", 0)
        repost_count = post_view.get("repostCount", 0)
        like_count = post_view.get("likeCount", 0)

        rkey = uri.rsplit("/", 1)[-1]

        return ParsedDocument(
            path=Path(f"bluesky://{rkey}.bluesky"),
            text=text,
            file_type=".bluesky",
            encoding="utf-8",
            metadata={
                "uri": uri,
                "cid": cid,
                "created_at": created_at,
                "reply_count": reply_count,
                "repost_count": repost_count,
                "like_count": like_count,
                "source": self._cfg.source_name,
            },
        )
