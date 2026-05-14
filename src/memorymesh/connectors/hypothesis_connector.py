"""Hypothes.is annotation connector for MemoryMesh.

Fetches web annotations made by the authenticated user via the
Hypothes.is API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per annotation.

API reference
-------------
``https://api.hypothes.is/api``

Authentication
--------------
Bearer token passed as ``Authorization: Bearer {token}``.
Generate one at https://hypothes.is/account/developer.

Features
--------
* **Offset pagination** - iterates via ``offset`` / ``limit`` until all
  annotations are fetched.
* **Date filtering** - annotations not updated within ``days_past`` are
  skipped.
* **Rich metadata** - includes the annotated URL, highlighted quote, and
  annotation text body.

Usage
-----
::

    connector = HypothesisConnector(HypothesisConfig(
        api_key=SecretStr("my-hypothesis-token"),
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
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://api.hypothes.is/api"
_PAGE_SIZE = 200


class HypothesisConfig(BaseModel):
    """Configuration for a Hypothes.is source.

    Args:
        api_key: Hypothes.is personal API token.
        days_past: Only include annotations updated within this many days.
            0 = no cutoff.
        max_annotations: Maximum total annotations to fetch.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    days_past: int = 365
    max_annotations: int = 5000
    source_name: str = "hypothesis"


class HypothesisConnector:
    """Fetches Hypothes.is annotations and yields one ParsedDocument each.

    Args:
        config: API key, date filter, and source settings.
    """

    def __init__(self, config: HypothesisConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Hypothes.is search API and yield annotation documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            annotation, with ``file_type=".hypothesis"`` and metadata
            containing ``id``, ``uri``, ``tags``, ``quote``, ``created``,
            and ``updated``.
        """
        token = self._cfg.api_key.get_secret_value()
        headers = bearer_header(token)
        cutoff = self._cutoff()
        offset = 0
        total_yielded = 0
        limit = self._cfg.max_annotations

        while True:
            if limit > 0 and total_yielded >= limit:
                break

            params: dict[str, Any] = {
                "user": "me",
                "sort": "updated",
                "order": "desc",
                "limit": _PAGE_SIZE,
                "offset": offset,
            }
            url = f"{_BASE}/search?{urlencode(params)}"
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            rows: list[dict[str, Any]] = data.get("rows", [])
            if not rows:
                break

            for row in rows:
                if limit > 0 and total_yielded >= limit:
                    break

                updated = row.get("updated", "")
                if cutoff and updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError as exc:
                        logger.debug(f"HypothesisConnector: ignoring unparsable timestamp: {exc}")

                doc = self._build_doc(row)
                if doc is not None:
                    yield doc
                    total_yielded += 1

            if len(rows) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        logger.info(f"HypothesisConnector: yielded {total_yielded} annotation(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _build_doc(self, row: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Hypothes.is annotation to a ParsedDocument.

        Args:
            row: Raw annotation dict from the API.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the annotation ID is missing.
        """
        ann_id = row.get("id")
        if not ann_id:
            return None

        uri = row.get("uri", "")
        created = row.get("created", "")
        updated = row.get("updated", "")
        tags: list[str] = row.get("tags") or []

        quote_text = ""
        target = row.get("target") or []
        for t in target:
            if isinstance(t, dict):
                for sel in t.get("selector") or []:
                    if isinstance(sel, dict) and sel.get("type") == "TextQuoteSelector":
                        quote_text = sel.get("exact", "")
                        break
            if quote_text:
                break

        body_text = ""
        for body in row.get("body") or []:
            if isinstance(body, dict) and body.get("type") == "TextualBody":
                body_text = body.get("value", "")
                break

        text_parts = [f"URL: {uri}"]
        if quote_text:
            text_parts.append(f"Quote: {quote_text}")
        if body_text:
            text_parts.append(f"Note: {body_text}")
        if tags:
            text_parts.append(f"Tags: {', '.join(tags)}")

        return ParsedDocument(
            path=Path(f"hypothesis://{ann_id}.hypothesis"),
            text="\n".join(text_parts),
            file_type=".hypothesis",
            encoding="utf-8",
            metadata={
                "id": ann_id,
                "uri": uri,
                "tags": tags,
                "quote": quote_text,
                "created": created,
                "updated": updated,
                "source": self._cfg.source_name,
            },
        )
