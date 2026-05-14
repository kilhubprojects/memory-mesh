"""Recursive character-level text splitter.

Splits text at natural boundaries (paragraphs → sentences → words → characters)
to keep chunks under *chunk_size* characters while preserving a *chunk_overlap*
sliding window for context continuity.
"""

from __future__ import annotations

import os

from memorymesh.chunking.base import Chunker
from memorymesh.core.models import Chunk, ChunkMetadata, ParsedDocument

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    """Generic recursive text splitter.

    Args:
        chunk_size: Target maximum character length per chunk.
        chunk_overlap: Number of characters to repeat at the start of the next
            chunk for context continuity.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split *doc* using recursive character splitting.

        Args:
            doc: Parsed document to split.

        Returns:
            List of :class:`~memorymesh.core.models.Chunk` objects.
        """
        text = doc.text
        if not text:
            return [_make_chunk(doc, 0, "", 0, 0)]

        raw_chunks = _split(text, self._chunk_size, self._chunk_overlap, _SEPARATORS)
        chunks: list[Chunk] = []
        for idx, (chunk_text, start, end) in enumerate(raw_chunks):
            chunks.append(_make_chunk(doc, idx, chunk_text, start, end))
        return chunks


def _split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[tuple[str, int, int]]:
    """Recursively split *text* and return ``(chunk_text, start_char, end_char)`` tuples."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    results: list[tuple[str, int, int]] = []
    _split_recursive(text, 0, chunk_size, chunk_overlap, separators, results)
    return results


def _split_recursive(
    text: str,
    offset: int,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    out: list[tuple[str, int, int]],
) -> None:
    if len(text) <= chunk_size:
        if text.strip():
            out.append((text, offset, offset + len(text)))
        return

    sep = ""
    for candidate in separators:
        if candidate in text:
            sep = candidate
            break

    if not sep:
        # Hard cut
        out.append((text[:chunk_size], offset, offset + chunk_size))
        overlap_start = max(0, chunk_size - chunk_overlap)
        _split_recursive(
            text[overlap_start:], offset + overlap_start, chunk_size, chunk_overlap, separators, out
        )
        return

    parts = text.split(sep)
    current = ""
    current_start = 0

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            if not current:
                current_start = offset
            current = candidate
        else:
            if current.strip():
                out.append((current, current_start, current_start + len(current)))
                overlap = current[-chunk_overlap:] if chunk_overlap else ""
                current = overlap + (sep if overlap else "") + part
                current_start = offset + text.index(part)
            else:
                current = part
                current_start = offset + text.index(part)

    if current.strip():
        out.append((current, current_start, current_start + len(current)))


def _make_chunk(
    doc: ParsedDocument,
    idx: int,
    text: str,
    start: int,
    end: int,
) -> Chunk:
    return Chunk(
        path=doc.path,
        chunk_index=idx,
        text=text,
        start_char=start,
        end_char=end,
        file_type=doc.file_type,
        mtime=float(os.path.getmtime(doc.path)) if doc.path.exists() else 0.0,
        source_root="",
        metadata=ChunkMetadata(),
    )
