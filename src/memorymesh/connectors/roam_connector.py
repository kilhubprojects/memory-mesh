"""Roam Research graph connector for MemoryMesh.

Parses a Roam Research JSON export and yields one
:class:`~memorymesh.core.models.ParsedDocument` per page.

Export format
-------------
Roam exports a single JSON file containing an array of page objects, each
with a ``title`` string and a ``children`` list of nested block objects.
Blocks may themselves have ``children``.

Features
--------
* **Recursive block flattening** - the nested block tree is flattened into
  a single plain-text string.
* **Roam syntax stripping** - ``{{[[...]]}}`` template refs, ``[[...]]``
  page refs, and ``#[[...]]`` / ``#tag`` hashtag syntax are cleaned to
  bare text.
* **Date filtering** - pages with ``create-time`` or ``edit-time`` outside
  the ``days_past`` window are skipped.
* **Empty page skipping** - pages with no block content are skipped.

Usage
-----
::

    connector = RoamConnector(RoamConfig(
        export_path=Path("roam-export.json"),
        days_past=365,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

# Patterns applied in order to strip Roam-specific syntax
_ROAM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\{\{(\[\[.*?\]\])\}\}"),  # {{[[template]]}}
    re.compile(r"\[\[([^\]]+)\]\]"),  # [[page ref]]
    re.compile(r"#\[\[([^\]]+)\]\]"),  # #[[tag]]
    re.compile(r"#(\S+)"),  # #tag
]


class RoamConfig(BaseModel):
    """Configuration for a Roam Research export source.

    Args:
        export_path: Path to the Roam JSON export file.
        days_past: Only include pages edited within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    days_past: int = 365
    source_name: str = "roam"


class RoamConnector:
    """Parses a Roam JSON export and yields one ParsedDocument per page.

    Args:
        config: Export path, date filter, and source settings.
    """

    def __init__(self, config: RoamConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the JSON export and yield page documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            non-empty page, with ``file_type=".roam"`` and metadata
            containing ``title``, ``create_time``, and ``edit_time``.
        """
        try:
            raw = self._cfg.export_path.read_text(encoding="utf-8")
            pages: list[Any] = json.loads(raw)
        except Exception as exc:
            logger.warning(f"RoamConnector: failed to read export: {exc}")
            return

        if not isinstance(pages, list):
            logger.warning("RoamConnector: export is not a JSON array")
            return

        cutoff = self._cutoff()
        total = 0

        for page in pages:
            if not isinstance(page, dict):
                continue
            doc = self._build_doc(page, cutoff)
            if doc is not None:
                yield doc
                total += 1

        logger.info(f"RoamConnector: yielded {total} page(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _flatten_blocks(self, blocks: list[Any], depth: int = 0) -> list[str]:
        """Recursively flatten nested Roam blocks into text lines.

        Args:
            blocks: List of block dicts (may have nested ``children``).
            depth: Current indentation depth (used for indent prefix).

        Returns:
            List of plain text strings, one per block.
        """
        lines: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            string = block.get("string", "")
            if string:
                cleaned = self._strip_roam_syntax(string)
                if cleaned:
                    indent = "  " * depth
                    lines.append(f"{indent}{cleaned}")
            children = block.get("children")
            if isinstance(children, list):
                lines.extend(self._flatten_blocks(children, depth + 1))
        return lines

    def _strip_roam_syntax(self, text: str) -> str:
        """Remove Roam-specific markup from a string.

        Args:
            text: Raw block string.

        Returns:
            Cleaned plain text.
        """
        for pattern in _ROAM_PATTERNS:
            text = pattern.sub(r"\1" if pattern.groups else "", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def _build_doc(self, page: dict[str, Any], cutoff: datetime | None) -> ParsedDocument | None:
        """Convert a Roam page dict to a ParsedDocument.

        Args:
            page: Raw Roam page dict.
            cutoff: UTC datetime cutoff; skip pages edited before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the page should be skipped.
        """
        title = page.get("title", "")
        if not title:
            return None

        create_time: int = int(page.get("create-time", 0) or 0)
        edit_time: int = int(page.get("edit-time", 0) or 0)
        ts = edit_time or create_time

        if cutoff and ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
            if dt < cutoff:
                return None

        children = page.get("children")
        block_lines: list[str] = []
        if isinstance(children, list):
            block_lines = self._flatten_blocks(children)

        if not block_lines:
            return None

        text = f"# {title}\n" + "\n".join(block_lines)
        safe_title = re.sub(r"[^\w\-]", "_", title)[:80]

        return ParsedDocument(
            path=Path(f"roam://{safe_title}.roam"),
            text=text,
            file_type=".roam",
            encoding="utf-8",
            metadata={
                "title": title,
                "create_time": create_time,
                "edit_time": edit_time,
                "source": self._cfg.source_name,
            },
        )
