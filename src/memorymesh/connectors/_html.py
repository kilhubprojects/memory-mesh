"""Shared HTML-to-text conversion utilities for MemoryMesh connectors.

Extracted so IMAP, RSS, and any future connectors can strip HTML without
duplicating the parser logic.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _HTMLStripper(HTMLParser):
    """Minimal HTMLParser subclass that strips tags and decodes entities."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Accumulate visible text nodes."""
        self._parts.append(data)

    def get_text(self) -> str:
        """Return all accumulated text joined by spaces."""
        return " ".join(self._parts)


def html_to_text(html_content: str) -> str:
    """Strip HTML tags and decode entities from *html_content*.

    Falls back to a regex pass if the HTML parser raises.

    Args:
        html_content: Raw HTML string.

    Returns:
        Plain text with tags removed and whitespace normalised.
    """
    stripper = _HTMLStripper()
    try:
        stripper.feed(html_content)
        text = stripper.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_content)
        text = html.unescape(text)
    return re.sub(r"\s{2,}", " ", text).strip()
