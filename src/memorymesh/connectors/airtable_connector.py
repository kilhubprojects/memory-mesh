"""Airtable connector for MemoryMesh.

Fetches records from Airtable bases via the Airtable REST API and yields
one :class:`~memorymesh.core.models.ParsedDocument` per table.

API reference
-------------
``https://api.airtable.com/v0``
``https://api.airtable.com/v0/meta/bases/{base_id}/tables``

Authentication
--------------
Bearer token (personal access token) passed as ``Authorization: Bearer {token}``.
Generate one at https://airtable.com/account.

Features
--------
* **Metadata API** - lists all tables in a base automatically.
* **Offset pagination** - iterates via ``offset`` until all records are
  fetched.
* **Field serialization** - all field values are converted to strings and
  included in the document text.
* **Date filtering** - records whose ``Created Time`` or
  ``Last Modified Time`` falls before the cutoff are skipped.

Usage
-----
::

    connector = AirtableConnector(AirtableConfig(
        access_token=SecretStr("patXXX..."),
        base_id="appXXX...",
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

_BASE_URL = "https://api.airtable.com/v0"
_META_URL = "https://api.airtable.com/v0/meta/bases"
_PAGE_SIZE = 100


class AirtableConfig(BaseModel):
    """Configuration for an Airtable source.

    Args:
        access_token: Airtable personal access token.
        base_id: Airtable base ID (starts with ``app``).
        table_names: Restrict to these table names.  Empty = all tables.
        days_past: Only include records modified within this many days.
            0 = no cutoff.
        max_records: Maximum records per table.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    access_token: SecretStr
    base_id: str
    table_names: list[str] = []
    days_past: int = 180
    max_records: int = 1000
    source_name: str = "airtable"


class AirtableConnector:
    """Fetches Airtable records and yields one ParsedDocument per record.

    Args:
        config: Airtable token, base ID, and source settings.
    """

    def __init__(self, config: AirtableConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """List tables, then paginate records and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            record, with ``file_type=".airtable"`` and metadata containing
            ``record_id``, ``table``, ``base_id``, and ``fields``.
        """
        token = self._cfg.access_token.get_secret_value()
        headers = bearer_header(token)
        cutoff = self._cutoff()
        total = 0

        tables = self._list_tables(headers)
        filter_names = {n.lower() for n in self._cfg.table_names}

        for table in tables:
            if not isinstance(table, dict):
                continue
            table_name = table.get("name", "")
            if filter_names and table_name.lower() not in filter_names:
                continue
            table_id = str(table.get("id") or table_name)

            for doc in self._fetch_table(headers, table_name, table_id, cutoff):
                yield doc
                total += 1

        logger.info(f"AirtableConnector: yielded {total} record(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _list_tables(self, headers: dict[str, str]) -> list[dict[str, Any]]:
        """Fetch table metadata for the configured base.

        Args:
            headers: Auth headers.

        Returns:
            List of table dicts with ``id`` and ``name``.
        """
        url = f"{_META_URL}/{self._cfg.base_id}/tables"
        data = api_get(url, headers)
        if not isinstance(data, dict):
            return []
        return data.get("tables", [])

    def _fetch_table(
        self,
        headers: dict[str, str],
        table_name: str,
        table_id: str,
        cutoff: datetime | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate records for one table and yield documents.

        Args:
            headers: Auth headers.
            table_name: Human-readable table name.
            table_id: Airtable table ID or name used in the URL.
            cutoff: UTC datetime cutoff.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per record.
        """
        base_id = self._cfg.base_id
        offset: str | None = None
        yielded = 0
        limit = self._cfg.max_records

        while True:
            if limit > 0 and yielded >= limit:
                break

            params: dict[str, Any] = {"pageSize": _PAGE_SIZE}
            if offset:
                params["offset"] = offset

            url = f"{_BASE_URL}/{base_id}/{table_id}?{urlencode(params)}"
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            records: list[dict[str, Any]] = data.get("records", [])
            if not records:
                break

            for rec in records:
                if limit > 0 and yielded >= limit:
                    break
                doc = self._build_doc(rec, table_name, cutoff)
                if doc is not None:
                    yield doc
                    yielded += 1

            offset = data.get("offset")
            if not offset:
                break

    def _build_doc(
        self,
        record: dict[str, Any],
        table_name: str,
        cutoff: datetime | None,
    ) -> ParsedDocument | None:
        """Convert an Airtable record to a ParsedDocument.

        Args:
            record: Raw Airtable record dict.
            table_name: Name of the parent table.
            cutoff: UTC datetime cutoff; skip records modified before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the record should be skipped.
        """
        record_id = record.get("id")
        if not record_id:
            return None

        modified = record.get("createdTime", "")
        fields: dict[str, Any] = record.get("fields") or {}

        last_modified = fields.get("Last Modified Time") or modified
        if cutoff and last_modified:
            try:
                dt = datetime.fromisoformat(str(last_modified).replace("Z", "+00:00"))
                if dt < cutoff:
                    return None
            except ValueError as exc:
                logger.debug(f"AirtableConnector: ignoring unparsable timestamp: {exc}")

        field_lines: list[str] = [f"{k}: {v}" for k, v in fields.items() if v is not None]
        text_parts = [
            f"Table: {table_name}",
            f"Record: {record_id}",
            *field_lines,
        ]

        return ParsedDocument(
            path=Path(f"airtable://{self._cfg.base_id}/{record_id}.airtable"),
            text="\n".join(text_parts),
            file_type=".airtable",
            encoding="utf-8",
            metadata={
                "record_id": record_id,
                "table": table_name,
                "base_id": self._cfg.base_id,
                "fields": {k: str(v) for k, v in fields.items()},
                "source": self._cfg.source_name,
            },
        )
