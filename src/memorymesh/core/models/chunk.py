"""Chunk and memory-tier domain models for MemoryMesh."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memorymesh.core.models.config import AgentPermission, FileStatusLiteral, MemoryTier


class ChunkMetadata(BaseModel):
    """Optional structural metadata attached to a chunk.

    Fields are populated by the relevant :class:`~memorymesh.chunking.base.Chunker`
    and stored alongside the chunk in ChromaDB so search hits can reference the
    exact code symbol or heading they came from.

    Extra fields (backlinks, chunk_type) are used by specialised parsers and
    the multi-vector indexing pipeline respectively.
    """

    heading_path: list[str] | None = None
    function_name: str | None = None
    class_name: str | None = None
    language: str | None = None
    # Obsidian wikilink backlinks extracted from [[link]] patterns.
    backlinks: list[str] | None = None
    # "chunk" for normal content; "summary" for Ollama-generated doc summaries.
    chunk_type: str = "chunk"


class ParsedDocument(BaseModel):
    """Output of a :class:`~memorymesh.parsing.base.Parser`.

    Args:
        path: Absolute path to the source file.
        text: Extracted plain text (may be empty for image-only PDFs).
        file_type: Normalised extension (e.g. ``".pdf"``).
        encoding: Detected encoding used to read the file.
        metadata: Free-form dict for parser-specific extras (page count, etc.).
    """

    path: Path
    text: str
    file_type: str
    encoding: str = "utf-8"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single text chunk produced by a :class:`~memorymesh.chunking.base.Chunker`.

    Args:
        path: Absolute path to the source file.
        chunk_index: Zero-based position in the file's chunk sequence.
        text: Raw chunk content.
        start_char: Character offset of the first character in the source text.
        end_char: Character offset one past the last character.
        file_type: Normalised extension of the source file.
        mtime: Modification timestamp of the source file (``os.path.getmtime``).
        source_root: Name of the :class:`SourceConfig` this file belongs to.
        metadata: Optional structural metadata (heading, function name, …).
    """

    model_config = ConfigDict(frozen=False)

    path: Path
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    file_type: str = ""
    mtime: float = 0.0
    source_root: str = ""
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

    @property
    def id(self) -> str:
        """Stable unique identifier: ``<path>:<chunk_index>``."""
        return f"{self.path}:{self.chunk_index}"


class ChunkWithEmbedding(Chunk):
    """A :class:`Chunk` augmented with its dense embedding vector.

    Produced by passing chunks through an
    :class:`~memorymesh.embeddings.base.EmbeddingProvider`.
    """

    embedding: list[float]


class FileRecord(BaseModel):
    """Row in the ``files`` table of the metadata SQLite database.

    Args:
        path: Absolute path string (primary key).
        source_name: Name of the owning :class:`SourceConfig`.
        sha256: SHA-256 hex digest of the file contents at index time.
        mtime: ``os.path.getmtime`` value at index time.
        size_bytes: File size in bytes.
        file_type: Normalised extension.
        n_chunks: Number of chunks produced.
        status: One of ``indexed``, ``parse_error``, ``unsupported``,
            ``deleted``, ``pending_reindex``.
        error_message: Human-readable description of the last failure, if any.
        indexed_at: Unix timestamp of the last successful (or failed) index.
        embedding_model_id: The ``model_id`` of the embedding provider used.
    """

    path: str
    source_name: str
    sha256: str
    mtime: float
    size_bytes: int
    file_type: str
    n_chunks: int
    status: FileStatusLiteral
    error_message: str | None = None
    indexed_at: float = Field(default_factory=time.time)
    embedding_model_id: str = ""


class IndexResult(BaseModel):
    """Result of indexing a single file, returned by :class:`FileIndexer`.

    Args:
        path: The file that was processed.
        status: Outcome — ``indexed``, ``skipped``, ``parse_error``, ``unsupported``.
        n_chunks: Number of chunks upserted (0 for non-indexed statuses).
        duration_ms: Wall time for the full pipeline in milliseconds.
        error: Error message when ``status`` is not ``indexed`` or ``skipped``.
    """

    path: Path
    status: Literal["indexed", "skipped", "parse_error", "unsupported"]
    n_chunks: int = 0
    duration_ms: float = 0.0
    error: str | None = None


class EpisodicEvent(BaseModel):
    """A timestamped event record stored in the ``episodic_events`` table.

    Args:
        event_id: Auto-generated UUID (assigned by the store).
        timestamp: Unix timestamp of the event.
        event_type: Category — ``"retrieval"``, ``"index"``, ``"pin"``,
            ``"forget"``, ``"user_note"``.
        source: File path or source name associated with the event.
        chunk_ids: List of chunk IDs involved (e.g. chunks retrieved).
        client_id: Agent that triggered the event (empty = anonymous).
        metadata: Free-form extra fields.
    """

    event_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    event_type: str = "retrieval"
    source: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    client_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A named entity extracted from indexed content (opt-in via Ollama).

    Args:
        name: Canonical entity name (lowercased).
        entity_type: One of ``person``, ``project``, ``concept``, ``location``.
        mention_count: Total number of chunks mentioning this entity.
        first_seen: Unix timestamp of the first indexed mention.
        last_seen: Unix timestamp of the most recent indexed mention.
    """

    name: str
    entity_type: str
    mention_count: int = 1
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)


class ChunkTierRecord(BaseModel):
    """Row in the ``chunk_tiers`` table.

    Args:
        chunk_id: ``<path>:<chunk_index>`` stable identifier.
        tier: Current memory tier.
        last_accessed: Unix timestamp of the most recent retrieval.
        access_count: Total number of times this chunk was retrieved.
        pinned: Whether this chunk was manually pinned to hot tier.
    """

    chunk_id: str
    tier: MemoryTier = MemoryTier.warm
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False


class ClientIdentity(BaseModel):
    """Resolved identity of an MCP client for a single request.

    Args:
        client_id: Identifier string extracted from the request.
        name: Human-readable label (from AgentConfig or auto-generated).
        permission: Effective permission level.
        allowed_sources: Empty list means "all sources allowed".
        rate_limit_per_min: Effective rate limit (0 = unlimited).
        is_anonymous: True when no valid client_id was provided.
    """

    client_id: str = "anonymous"
    name: str = "anonymous"
    permission: AgentPermission = AgentPermission.read
    allowed_sources: list[str] = Field(default_factory=list)
    rate_limit_per_min: int = 120
    is_anonymous: bool = True
