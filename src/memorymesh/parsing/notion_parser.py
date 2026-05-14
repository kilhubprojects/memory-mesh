"""Notion HTML export parser.

Notion exports pages as single ``.html`` files with a predictable structure:
* ``<h1>`` or ``<title>`` holds the page title.
* Database property blocks carry ``data-type`` attributes.
* The page UUID is embedded in the filename (``<title> <uuid>.html``).

Only the stdlib ``html.parser`` is used — no external dependencies.

The parser is registered for ``.html`` and ``.htm`` files when a source is
configured with ``source.type: notion``.  For general HTML sources the standard
:class:`~memorymesh.parsing.text.TextParser` is used instead (stripping is not
performed; raw HTML is indexed).
"""

from __future__ import annotations

import re
import uuid as _uuid_module
from html.parser import HTMLParser
from pathlib import Path

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

# UUID pattern found in Notion export filenames: ``Page Title a1b2c3...html``
_UUID_FILENAME_RE = re.compile(
    r"([0-9a-f]{8}(?:[0-9a-f]{4}){3}[0-9a-f]{12})(?:\s|\.|$)",
    re.IGNORECASE,
)


class _NotionHTMLParser(HTMLParser):
    """SAX-style parser that collects clean text and metadata from Notion HTML.

    Extracts:
    * ``<title>`` tag content.
    * ``<h1>`` text (the visible page title).
    * All ``data-type`` attribute values (database property types).
    * All visible body text, stripped of markup.

    Tags whose content is typically non-text (``style``, ``script``, ``head``)
    are suppressed entirely.
    """

    _SKIP_TAGS: frozenset[str] = frozenset({"style", "script", "head", "meta", "link"})

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.title: str = ""
        self.h1_title: str = ""
        self.db_properties: list[dict[str, str]] = []

        self._skip_depth: int = 0
        self._in_title: bool = False
        self._in_h1: bool = False
        self._current_property: dict[str, str] | None = None
        self._last_tag: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Process an opening HTML tag, tracking structure and Notion metadata."""
        tag = tag.lower()
        self._last_tag = tag

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "h1":
            self._in_h1 = True
            return

        # Detect Notion database property blocks.
        attr_dict = dict(attrs)
        data_type = attr_dict.get("data-type")
        if data_type:
            self._current_property = {"type": data_type, "value": ""}

    def handle_endtag(self, tag: str) -> None:
        """Process a closing HTML tag, finalising Notion database property blocks."""
        tag = tag.lower()

        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == "title":
            self._in_title = False
            return

        if tag == "h1":
            self._in_h1 = False
            return

        if self._current_property is not None and tag in ("td", "span", "div", "p"):
            value = self._current_property.get("value", "").strip()
            if value:
                self.db_properties.append(
                    {
                        "type": self._current_property["type"],
                        "value": value,
                    }
                )
            self._current_property = None

    def handle_data(self, data: str) -> None:
        """Accumulate visible text, routing it to title, heading, or body buckets."""
        if self._skip_depth > 0:
            return

        stripped = data.strip()
        if not stripped:
            return

        if self._in_title:
            self.title = stripped
            return

        if self._in_h1 and not self.h1_title:
            self.h1_title = stripped

        if self._current_property is not None:
            self._current_property["value"] = self._current_property.get("value", "") + data
        else:
            self.text_parts.append(stripped)

    @property
    def clean_text(self) -> str:
        """Return all collected text parts joined by newlines."""
        return "\n".join(self.text_parts)


def _extract_notion_id(filename: str) -> str | None:
    """Extract the Notion page UUID embedded in a Notion export filename.

    Notion exports pages as ``<Title> <uuid>.html``.  The UUID may be the
    full 36-char form or the 32-char compact form.

    Args:
        filename: The filename without directory (e.g. ``My Page 1a2b3c....html``).

    Returns:
        The UUID string if found, or ``None``.
    """
    m = _UUID_FILENAME_RE.search(filename)
    if m:
        raw = m.group(1).replace("-", "")
        try:
            return str(_uuid_module.UUID(hex=raw))
        except ValueError:
            return raw
    return None


class NotionParser(Parser):
    """Parser for Notion HTML export files.

    Extracts visible text and database property metadata from single-file
    Notion exports.  Only ``html.parser`` from the stdlib is used.

    Metadata returned in :attr:`~memorymesh.core.models.ParsedDocument.metadata`:

    * ``notion_id`` — page UUID extracted from the filename.
    * ``page_title`` — ``<h1>`` content (preferred) or ``<title>`` content.
    * ``database_name`` — name of the immediate parent directory (represents
      the Notion database when pages are exported inside a database folder).
    * ``db_properties`` — list of ``{"type": …, "value": …}`` dicts from
      ``data-type`` attribute blocks.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: Notion HTML exports."""
        return frozenset({".html", ".htm"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a Notion HTML export file.

        Args:
            path: Absolute path to the ``.html`` file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with extracted
            plain text and Notion metadata.
        """
        try:
            raw_html = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"NotionParser: cannot read {path}: {exc}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".html",
                metadata={"error": str(exc)},
            )

        html_parser = _NotionHTMLParser()
        try:
            html_parser.feed(raw_html)
        except Exception as exc:
            logger.warning(f"NotionParser: HTML parse error in {path.name!r}: {exc}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".html",
                metadata={"error": str(exc)},
            )

        page_title = html_parser.h1_title or html_parser.title or path.stem
        notion_id = _extract_notion_id(path.name)
        database_name = path.parent.name

        meta: dict[str, object] = {
            "notion_id": notion_id,
            "page_title": page_title,
            "database_name": database_name,
            "db_properties": html_parser.db_properties,
        }

        body = html_parser.clean_text
        if not body.strip():
            logger.warning(
                f"NotionParser: no text extracted from {path.name!r} — "
                "check that the file is a valid Notion HTML export"
            )

        logger.debug(
            f"NotionParser: {path.name!r} title={page_title!r} "
            f"props={len(html_parser.db_properties)}"
        )

        return ParsedDocument(
            path=path,
            text=body,
            file_type=".html",
            encoding="utf-8",
            metadata=meta,
        )
