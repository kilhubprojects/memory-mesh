"""Markdown parser.

Uses the same encoding-resilient reader as :mod:`~memorymesh.parsing.text`
but tags the result with ``file_type=".md"`` so the chunking layer can select
the heading-aware splitter.
"""

from __future__ import annotations

from pathlib import Path

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser
from memorymesh.parsing.text import _read_with_fallback


class MarkdownParser(Parser):
    """Parser for Markdown files (.md, .mdx, .markdown)."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: Markdown files."""
        return frozenset({".md", ".mdx", ".markdown"})

    def parse(self, path: Path) -> ParsedDocument:
        """Read and return Markdown file content.

        Args:
            path: Absolute path to the Markdown file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with raw Markdown text.
        """
        text, encoding, error = _read_with_fallback(path)
        meta: dict[str, object] = {}
        if error:
            meta["error"] = error
        return ParsedDocument(
            path=path,
            text=text,
            file_type=".md",
            encoding=encoding,
            metadata=meta,
        )
