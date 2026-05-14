"""Search result domain models for MemoryMesh."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A single result from the search engine.

    Args:
        path: Absolute path to the source file.
        chunk_index: Position of this chunk within the file.
        score: Relevance score (higher is better; range depends on fusion mode).
        preview: Up to 200 characters of chunk text.
        file_type: Normalised extension.
        mtime: Modification timestamp at index time.
        source_root: Name of the owning source.
        start_char: Character offset of the first character of this chunk in the
            source file.  Populated from Chroma metadata; defaults to 0.
        end_char: Character offset one past the last character of this chunk.
            Populated from Chroma metadata; defaults to 0.
        extended_preview: Wider context window around the chunk, populated by
            :func:`~memorymesh.search.context.expand_context` when
            ``search.parent_window_chars > 0``.  ``None`` when disabled.
    """

    path: str
    chunk_index: int
    score: float
    preview: str
    file_type: str
    mtime: float
    source_root: str
    start_char: int = 0
    end_char: int = 0
    extended_preview: str | None = None


class SearchResponse(BaseModel):
    """Full response from :func:`search_memory` MCP tool.

    Args:
        hits: Ranked list of search results.
        mode: Search mode used — ``hybrid``, ``dense``, or ``sparse``.
        duration_ms: Total query latency in milliseconds.
    """

    hits: list[SearchHit]
    mode: Literal["hybrid", "dense", "sparse"]
    duration_ms: float


class SourceStats(BaseModel):
    """Per-source statistics returned by :func:`list_sources` MCP tool."""

    name: str
    path: str
    recursive: bool
    n_files_indexed: int
    n_files_pending: int
    n_files_errored: int
    last_scan_at: float | None
    total_chunks: int
    disk_size_bytes: int


class SourcesReport(BaseModel):
    """Full response from the :func:`list_sources` MCP tool."""

    sources: list[SourceStats]


class DocumentResponse(BaseModel):
    """Full response from the :func:`get_document` MCP tool.

    Args:
        path: Absolute path of the document.
        content: File contents (possibly truncated).
        file_type: Normalised extension.
        size_bytes: File size in bytes.
        mtime: Modification timestamp.
        truncated: ``True`` when ``content`` was cut at ``max_bytes``.
    """

    path: str
    content: str
    file_type: str
    size_bytes: int
    mtime: float
    truncated: bool


class IndexResponse(BaseModel):
    """Full response from the :func:`index_now` MCP tool."""

    n_files_processed: int
    n_chunks: int
    duration_ms: float
    errors: list[str] = Field(default_factory=list)
