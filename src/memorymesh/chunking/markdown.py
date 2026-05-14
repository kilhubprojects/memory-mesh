"""Heading-based Markdown chunker.

Splits a Markdown document at ATX headings (``#``, ``##``, …).  Each chunk
contains one section: the heading line + the body text below it, up to the
next same-or-higher-level heading.  Sections exceeding *max_chunk_size*
characters are further split by the recursive chunker.

The ``heading_path`` metadata field is populated with the full ancestor heading
stack so search hits can reference their exact location in the document.
"""

from __future__ import annotations

import os
import re

from memorymesh.chunking.base import Chunker
from memorymesh.chunking.recursive import RecursiveChunker
from memorymesh.core.models import Chunk, ChunkMetadata, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


class MarkdownChunker(Chunker):
    """Split Markdown by ATX headings with recursive overflow splitting.

    Args:
        max_chunk_size: Sections longer than this are split by
            :class:`~memorymesh.chunking.recursive.RecursiveChunker`.
    """

    def __init__(self, max_chunk_size: int = 800) -> None:
        self._max_chunk_size = max_chunk_size
        self._fallback = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=50)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split *doc* at Markdown headings.

        Args:
            doc: Parsed Markdown document.

        Returns:
            List of :class:`~memorymesh.core.models.Chunk` objects.
        """
        text = doc.text
        if not text:
            return [_make_chunk(doc, 0, "", 0, 0, [])]

        sections = _extract_sections(text)
        chunks: list[Chunk] = []
        idx = 0

        for section_text, start_char, heading_path in sections:
            if len(section_text) <= self._max_chunk_size:
                end_char = start_char + len(section_text)
                chunks.append(
                    _make_chunk(doc, idx, section_text, start_char, end_char, heading_path)
                )
                idx += 1
            else:
                # overflow: delegate to recursive chunker with a synthetic doc
                synthetic = ParsedDocument(
                    path=doc.path,
                    text=section_text,
                    file_type=doc.file_type,
                    encoding=doc.encoding,
                )
                sub_chunks = self._fallback.chunk(synthetic)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            path=doc.path,
                            chunk_index=idx,
                            text=sub.text,
                            start_char=start_char + sub.start_char,
                            end_char=start_char + sub.end_char,
                            file_type=doc.file_type,
                            mtime=sub.mtime,
                            source_root="",
                            metadata=ChunkMetadata(heading_path=heading_path),
                        )
                    )
                    idx += 1

        return chunks if chunks else [_make_chunk(doc, 0, text, 0, len(text), [])]


def _extract_sections(
    text: str,
) -> list[tuple[str, int, list[str]]]:
    """Return ``(section_text, start_char, heading_path)`` for each heading section.

    Text before the first heading becomes a section with an empty heading path.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(text, 0, [])]

    sections: list[tuple[str, int, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    # Preamble before first heading
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append((preamble, 0, []))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Trim stack to current level
        heading_stack = [(lvl, ttl) for lvl, ttl in heading_stack if lvl < level]
        heading_stack.append((level, title))
        heading_path = [ttl for _, ttl in heading_stack]

        section_start = match.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()

        if section_text:
            sections.append((section_text, section_start, heading_path))

    return sections


def _make_chunk(
    doc: ParsedDocument,
    idx: int,
    text: str,
    start: int,
    end: int,
    heading_path: list[str],
) -> Chunk:
    mtime = float(os.path.getmtime(doc.path)) if doc.path.exists() else 0.0
    return Chunk(
        path=doc.path,
        chunk_index=idx,
        text=text,
        start_char=start,
        end_char=end,
        file_type=doc.file_type,
        mtime=mtime,
        source_root="",
        metadata=ChunkMetadata(heading_path=heading_path if heading_path else None),
    )
