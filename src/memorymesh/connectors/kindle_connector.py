"""Kindle highlights connector for MemoryMesh.

Reads ``My Clippings.txt`` from a Kindle device or backup and yields each
highlight as a :class:`~memorymesh.core.models.ParsedDocument`.

Format of ``My Clippings.txt``
------------------------------
Entries are separated by a line containing only ``==========``.  Each entry
has the structure::

    Book Title (Author Name)
    [blank line]
    Your Highlight on page X | location Y-Z | Added on Day, Mon D, YYYY HH:MM:SS
    [blank line]
    Highlight text here

Notes and bookmarks follow the same separator pattern but have different
metadata lines; they are parsed and included as well.

Features
--------
* **Stdlib only** - pure text parsing, no third-party dependencies.
* **Privacy** - highlight text is never logged at INFO level.

Usage
-----
::

    connector = KindleConnector(KindleConfig(
        clippings_path=Path("/media/Kindle/documents/My Clippings.txt"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_SEPARATOR = "=========="


class KindleConfig(BaseModel):
    """Configuration for a Kindle highlights source.

    Args:
        clippings_path: Path to ``My Clippings.txt``.
        source_name: Name used in the MemoryMesh source registry.
    """

    clippings_path: Path
    source_name: str = "kindle"


_RE_TITLE_AUTHOR = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_RE_META = re.compile(
    r"Your\s+(?:Highlight|Note|Bookmark)"
    r"(?:\s+on\s+(?:page\s+[\d\-]+\s*\|?\s*)?)?(?:location\s+([\d\-]+))?"
    r".*?Added\s+on\s+(.+)$",
    re.IGNORECASE,
)


def _safe_path_segment(text: str) -> str:
    """Sanitise *text* for use in a synthetic path segment.

    Args:
        text: Raw title or location string.

    Returns:
        String with non-alphanumeric characters replaced by underscores,
        truncated to 60 characters.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)[:60]


def _parse_block(block: str, source_name: str) -> ParsedDocument | None:
    """Parse a single clipping block into a ParsedDocument.

    Args:
        block: Raw text of one clipping entry (between separators).
        source_name: Source name for the synthetic path.

    Returns:
        :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if the
        block cannot be parsed or contains no text.
    """
    lines = [ln.rstrip() for ln in block.strip().splitlines()]
    if len(lines) < 3:
        return None

    # Line 0: Title (Author)
    title_line = lines[0].strip()
    title = title_line
    author = ""
    m = _RE_TITLE_AUTHOR.match(title_line)
    if m:
        title = m.group(1).strip()
        author = m.group(2).strip()

    # Line 1: blank, Line 2: metadata
    meta_line = lines[2].strip() if len(lines) > 2 else ""
    location = ""
    date_added = ""
    m2 = _RE_META.search(meta_line)
    if m2:
        location = (m2.group(1) or "").strip()
        date_added = (m2.group(2) or "").strip()

    # Lines 4+: highlight text (skip blank line 3)
    text_lines = [ln for ln in lines[4:] if ln] if len(lines) > 4 else []
    highlight_text = "\n".join(text_lines).strip()

    if not highlight_text:
        return None

    # Assemble document text
    parts: list[str] = []
    parts.append(f"Book: {title}")
    if author:
        parts.append(f"Author: {author}")
    if location:
        parts.append(f"Location: {location}")
    if date_added:
        parts.append(f"Date: {date_added}")
    parts.append("")
    parts.append(highlight_text)
    text = "\n".join(parts)

    safe_title = _safe_path_segment(title)
    safe_loc = _safe_path_segment(location) if location else "0"
    synthetic_path = Path(f"kindle://{source_name}/{safe_title}/{safe_loc}.kindle")

    return ParsedDocument(
        path=synthetic_path,
        text=text,
        file_type=".kindle",
        encoding="utf-8",
        metadata={
            "title": title,
            "author": author,
            "location": location,
            "date_added": date_added,
            "source": source_name,
        },
    )


class KindleConnector:
    """Reads ``My Clippings.txt`` and yields highlights as ParsedDocuments.

    Args:
        config: Path to ``My Clippings.txt`` and source settings.
    """

    def __init__(self, config: KindleConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the clippings file and yield one ParsedDocument per highlight.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            highlight/note, with ``file_type=".kindle"`` and metadata
            containing ``title``, ``author``, ``location``, and ``date_added``.
        """
        path = self._cfg.clippings_path
        if not path.exists():
            logger.warning(f"KindleConnector: file not found: {path}")
            return

        logger.info(f"KindleConnector: reading {path}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"KindleConnector: cannot read file: {exc}")
            return

        blocks = raw.split(_SEPARATOR)
        yielded = 0
        skipped = 0

        for block in blocks:
            if not block.strip():
                continue
            try:
                doc = _parse_block(block, self._cfg.source_name)
            except Exception as exc:
                logger.warning(f"KindleConnector: parse error: {exc}")
                skipped += 1
                continue

            if doc is None:
                skipped += 1
                continue

            yield doc
            yielded += 1

        logger.info(f"KindleConnector: yielded={yielded} skipped={skipped}")


# Backward-compatible alias
KindleHighlightsConnector = KindleConnector
