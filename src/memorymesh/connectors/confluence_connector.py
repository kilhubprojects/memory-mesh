"""Confluence wiki connector for MemoryMesh.

Fetches pages from Confluence Cloud via the REST API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per page.

API reference
-------------
``{base_url}/rest/api/content``

Authentication
--------------
HTTP Basic authentication with ``email:api_token`` (base64-encoded).
Generate an API token at https://id.atlassian.com/manage-profile/security/api-tokens.

Features
--------
* **Offset pagination** - iterates via ``start`` / ``limit`` until all
  pages in the requested spaces are fetched.
* **Space filtering** - optionally restrict to specific space keys.
* **HTML body stripping** - page bodies returned in ``storage`` format are
  stripped to plain text.
* **Date filtering** - pages not updated within ``days_past`` are skipped.

Usage
-----
::

    connector = ConfluenceConnector(ConfluenceConfig(
        base_url="https://myorg.atlassian.net",
        email="me@example.com",
        api_token=SecretStr("my-token"),
        space_keys=["ENG", "DOCS"],
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
from urllib.parse import urlencode, urljoin

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import basic_header
from memorymesh.connectors._html import html_to_text
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_PAGE_SIZE = 25


class ConfluenceConfig(BaseModel):
    """Configuration for a Confluence wiki source.

    Args:
        base_url: Confluence Cloud base URL, e.g.
            ``https://myorg.atlassian.net``.
        email: Atlassian account email address.
        api_token: Atlassian API token.
        space_keys: Restrict to these space keys.  Empty = all spaces.
        days_past: Only include pages updated within this many days.
            0 = no cutoff.
        max_pages: Maximum total pages to fetch.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    base_url: str
    email: str
    api_token: SecretStr
    space_keys: list[str] = []
    days_past: int = 180
    max_pages: int = 500
    source_name: str = "confluence"


class ConfluenceConnector:
    """Fetches Confluence pages and yields one ParsedDocument per page.

    Args:
        config: Confluence credentials, space filters, and source settings.
    """

    def __init__(self, config: ConfluenceConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Confluence content API and yield page documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            page, with ``file_type=".confluence"`` and metadata containing
            ``id``, ``title``, ``space_key``, ``space_name``, ``created_by``,
            ``created_at``, and ``updated_at``.
        """
        headers = {
            "Authorization": self._auth_header(),
            "Accept": "application/json",
        }
        cutoff = self._cutoff()
        start = 0
        total_yielded = 0
        limit = self._cfg.max_pages

        space_keys = self._cfg.space_keys or self._all_space_keys(headers)

        for space_key in space_keys:
            start = 0
            while True:
                if limit > 0 and total_yielded >= limit:
                    break
                params = urlencode(
                    {
                        "spaceKey": space_key,
                        "type": "page",
                        "status": "current",
                        "start": start,
                        "limit": _PAGE_SIZE,
                        "expand": "body.storage,history,space,version",
                    }
                )
                url = urljoin(
                    self._cfg.base_url.rstrip("/") + "/",
                    f"rest/api/content?{params}",
                )
                data = api_get(url, headers)
                if not isinstance(data, dict):
                    break

                results: list[dict[str, Any]] = data.get("results", [])
                if not results:
                    break

                for page in results:
                    if limit > 0 and total_yielded >= limit:
                        break
                    doc = self._build_doc(page, cutoff)
                    if doc is not None:
                        yield doc
                        total_yielded += 1

                size = data.get("size", 0)
                if size < _PAGE_SIZE:
                    break
                start += _PAGE_SIZE

        logger.info(f"ConfluenceConnector: yielded {total_yielded} page(s)")

    def _auth_header(self) -> str:
        """Build the HTTP Basic Authorization header value.

        Returns:
            ``Basic <base64(email:token)>`` string.
        """
        return basic_header(
            self._cfg.email,
            self._cfg.api_token.get_secret_value(),
        )["Authorization"]

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime for ``updatedDate`` filtering.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _all_space_keys(self, headers: dict[str, str]) -> list[str]:
        """Fetch all accessible space keys from the API.

        Args:
            headers: Auth headers for the request.

        Returns:
            List of space key strings.
        """
        url = urljoin(
            self._cfg.base_url.rstrip("/") + "/",
            "rest/api/space?limit=250",
        )
        data = api_get(url, headers)
        if not isinstance(data, dict):
            return []
        return [s.get("key", "") for s in data.get("results", []) if s.get("key")]

    def _build_doc(
        self,
        page: dict[str, Any],
        cutoff: datetime | None,
    ) -> ParsedDocument | None:
        """Convert a Confluence page API object to a ParsedDocument.

        Args:
            page: Raw Confluence page result dict.
            cutoff: UTC datetime cutoff; skip pages updated before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the page should be skipped.
        """
        page_id = page.get("id")
        if not page_id:
            return None

        title = page.get("title", "")
        space = page.get("space") or {}
        space_key = space.get("key", "")
        space_name = space.get("name", "")

        history = page.get("history") or {}
        created_by_obj = history.get("createdBy") or {}
        created_by = created_by_obj.get("displayName", "")
        created_at = history.get("createdDate", "")

        version = page.get("version") or {}
        updated_at = version.get("when", "")

        if cutoff and updated_at:
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if dt < cutoff:
                    return None
            except ValueError as exc:
                logger.debug(f"ConfluenceConnector: ignoring unparsable timestamp: {exc}")

        body_storage = (page.get("body") or {}).get("storage") or {}
        html_body = body_storage.get("value", "")
        plain_text = html_to_text(html_body) if html_body else ""

        text_parts = [
            f"Title: {title}",
            f"Space: {space_name} ({space_key})",
            f"Created by: {created_by}",
        ]
        if plain_text:
            text_parts.append(f"\n{plain_text}")

        return ParsedDocument(
            path=Path(f"confluence://{space_key}/{page_id}.confluence"),
            text="\n".join(text_parts),
            file_type=".confluence",
            encoding="utf-8",
            metadata={
                "id": page_id,
                "title": title,
                "space_key": space_key,
                "space_name": space_name,
                "created_by": created_by,
                "created_at": created_at,
                "updated_at": updated_at,
                "source": self._cfg.source_name,
            },
        )
