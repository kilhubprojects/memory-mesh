"""Mastodon connector for MemoryMesh.

Fetches the authenticated user's posts (toots) from any Mastodon-compatible
instance via the Mastodon API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per toot.

API reference
-------------
``{instance_url}/api/v1/accounts/{account_id}/statuses``

Authentication
--------------
Bearer token passed as ``Authorization: Bearer {token}``.
Generate one in your Mastodon instance under Settings -> Development -> New Application.

Features
--------
* **Link-header pagination** - reads ``max_id`` from the ``Link: <...>; rel="next"``
  response header to iterate backwards through the timeline.
* **Boost filtering** - retoots (``reblog != null``) are skipped by default.
* **HTML stripping** - post content is stripped to plain text.
* **Date filtering** - toots older than ``days_past`` are skipped.

Usage
-----
::

    connector = MastodonConnector(MastodonConfig(
        instance_url="https://mastodon.social",
        access_token=SecretStr("my-token"),
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._html import html_to_text
from memorymesh.connectors._http import api_get, api_get_with_headers
from memorymesh.core.models import ParsedDocument

_PAGE_SIZE = 40


class MastodonConfig(BaseModel):
    """Configuration for a Mastodon source.

    Args:
        instance_url: Mastodon instance URL, e.g.
            ``https://mastodon.social``.
        access_token: Mastodon user access token.
        skip_boosts: Whether to skip retoots (``reblog != null``).
        skip_replies: Whether to skip reply posts.
        days_past: Only include toots within this many days.  0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    instance_url: str
    access_token: SecretStr
    skip_boosts: bool = True
    skip_replies: bool = False
    days_past: int = 180
    source_name: str = "mastodon"


class MastodonConnector:
    """Fetches Mastodon toots and yields one ParsedDocument per toot.

    Args:
        config: Mastodon credentials and source settings.
    """

    def __init__(self, config: MastodonConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Mastodon status timeline and yield toot documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            toot, with ``file_type=".mastodon"`` and metadata containing
            ``id``, ``created_at``, ``language``, ``tags``, and
            ``visibility``.
        """
        headers = bearer_header(self._cfg.access_token.get_secret_value())
        base = self._cfg.instance_url.rstrip("/")

        account_id = self._get_account_id(headers, base)
        if not account_id:
            logger.warning("MastodonConnector: could not resolve account ID")
            return

        cutoff = self._cutoff()
        max_id: str | None = None
        total = 0
        stop = False

        while not stop:
            params: dict[str, Any] = {"limit": _PAGE_SIZE}
            if max_id:
                params["max_id"] = max_id

            url = f"{base}/api/v1/accounts/{account_id}/statuses?{urlencode(params)}"
            data, resp_headers = api_get_with_headers(url, headers)

            if not isinstance(data, list) or not data:
                break

            for status in data:
                if self._cfg.skip_boosts and status.get("reblog"):
                    continue
                if self._cfg.skip_replies and status.get("in_reply_to_id"):
                    continue

                created_at = status.get("created_at", "")
                if cutoff and created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if dt < cutoff:
                            stop = True
                            break
                    except ValueError as exc:
                        logger.debug(f"MastodonConnector: ignoring unparsable timestamp: {exc}")

                doc = self._build_doc(status)
                if doc is not None:
                    yield doc
                    total += 1

            max_id = self._next_max_id(resp_headers.get("link", ""))
            if not max_id:
                break

        logger.info(f"MastodonConnector: yielded {total} toot(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _get_account_id(self, headers: dict[str, str], base: str) -> str | None:
        """Fetch the authenticated account ID via ``verify_credentials``.

        Args:
            headers: Auth headers.
            base: Instance base URL.

        Returns:
            Account ID string, or ``None`` on failure.
        """
        data = api_get(f"{base}/api/v1/accounts/verify_credentials", headers)
        if isinstance(data, dict):
            return str(data.get("id", "")) or None
        return None

    @staticmethod
    def _next_max_id(link_header: str) -> str | None:
        """Parse the ``max_id`` cursor from a Mastodon Link response header.

        Args:
            link_header: Value of the ``Link`` HTTP response header.

        Returns:
            ``max_id`` string value, or ``None`` if not present.
        """
        match = re.search(r'max_id=(\d+)[^>]*>;\s*rel="next"', link_header)
        return match.group(1) if match else None

    def _build_doc(self, status: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Mastodon status to a ParsedDocument.

        Args:
            status: Raw Mastodon API status dict.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the status ID is missing.
        """
        status_id = status.get("id")
        if not status_id:
            return None

        created_at = status.get("created_at", "")
        language = status.get("language") or ""
        visibility = status.get("visibility", "")
        html_content = status.get("content", "")
        plain_text = html_to_text(html_content)

        tags: list[str] = [
            t.get("name", "") for t in (status.get("tags") or []) if isinstance(t, dict)
        ]

        text_parts = [plain_text]
        if tags:
            text_parts.append(f"Tags: {', '.join(tags)}")

        return ParsedDocument(
            path=Path(f"mastodon://{status_id}.mastodon"),
            text="\n".join(text_parts),
            file_type=".mastodon",
            encoding="utf-8",
            metadata={
                "id": str(status_id),
                "created_at": created_at,
                "language": language,
                "tags": tags,
                "visibility": visibility,
                "source": self._cfg.source_name,
            },
        )
