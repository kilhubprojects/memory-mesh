"""FastMCP application factory and AppContext definition.

``AppContext`` is a frozen dataclass that holds every dependency the MCP tools
need.  It is constructed once at startup and injected into each tool via
closure inside the ``register(mcp, ctx)`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.auth.acl import ACLEnforcer
    from memorymesh.auth.identity import IdentityResolver
    from memorymesh.auth.rate_limiter import RateLimiter
    from memorymesh.auth.revocation import RevocationList
    from memorymesh.core.models import MemoryMeshConfig
    from memorymesh.embeddings.base import EmbeddingProvider
    from memorymesh.embeddings.multimodal.clip_provider import CLIPProvider
    from memorymesh.embeddings.multimodal.whisper_provider import WhisperProvider
    from memorymesh.indexer.file_indexer import FileIndexer
    from memorymesh.llm.ollama_client import OllamaClient
    from memorymesh.observability.audit import AuditLogger
    from memorymesh.search.engine import SearchEngine
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.tiered import TieredMemoryManager
    from memorymesh.storage.vector_store import VectorStore


@dataclass(frozen=True)
class AppContext:
    """All runtime dependencies for the MCP server.

    Frozen so tools cannot accidentally mutate shared state.

    Args:
        config: Root MemoryMesh configuration.
        vector_store: ChromaDB wrapper for dense vectors.
        metadata_store: SQLite metadata store.
        bm25: Sparse BM25 index.
        provider: Embedding provider.
        indexer: File indexer / pipeline orchestrator.
        engine: Hybrid search engine.
        audit_logger: Append-only JSONL audit logger.
        ollama_client: Optional Ollama LLM client.  ``None`` when
            ``ollama.enabled: false`` in the config.
        tiered_memory: Optional tiered memory manager for hot/warm/cold tier
            support (Wave 3).  ``None`` when memory tier config is absent.
        identity_resolver: Auth identity resolver (Wave 4).  Always present;
            returns permissive defaults when ``auth.enabled: false``.
        acl_enforcer: ACL enforcer (Wave 4).  Always present; no-op when auth
            is disabled.
        rate_limiter: Token-bucket rate limiter (Wave 4).  Always present;
            no-op (unlimited) when auth is disabled.
        revocation_list: SQLite-backed revocation deny list (Wave 4).
            ``None`` when auth storage is unavailable.
    """

    config: MemoryMeshConfig
    vector_store: VectorStore
    metadata_store: MetadataStore
    bm25: BM25Index
    provider: EmbeddingProvider
    indexer: FileIndexer
    engine: SearchEngine
    audit_logger: AuditLogger
    ollama_client: OllamaClient | None = field(default=None)
    tiered_memory: TieredMemoryManager | None = field(default=None)
    # Wave 4 auth — always instantiated (permissive when auth.enabled: false)
    identity_resolver: IdentityResolver | None = field(default=None)
    acl_enforcer: ACLEnforcer | None = field(default=None)
    rate_limiter: RateLimiter | None = field(default=None)
    revocation_list: RevocationList | None = field(default=None)
    # Multimodal providers — None when not configured or optional deps absent
    clip_provider: CLIPProvider | None = field(default=None)
    whisper_provider: WhisperProvider | None = field(default=None)


def build_mcp(ctx: AppContext) -> FastMCP:
    """Construct and configure the FastMCP instance with all tools registered.

    Args:
        ctx: Fully initialised :class:`AppContext` with all dependencies.

    Returns:
        A :class:`~mcp.server.fastmcp.FastMCP` instance ready to serve.
    """
    mcp: FastMCP = FastMCP(
        name="memorymesh",
        instructions=(
            "MemoryMesh gives you read access to the user's personal knowledge "
            "base — local files indexed from configured source directories.  "
            "Use search_memory to find relevant passages, list_sources to "
            "explore what has been indexed, get_document to read a full file, "
            "index_now to trigger an immediate re-index, and ask_memory to "
            "get an LLM-generated answer from your indexed content (requires "
            "Ollama to be installed and running locally).  "
            "Use pin_memory / unpin_memory to control hot-tier pinning, "
            "forget_memory to suppress stale content, and query_timeline to "
            "explore episodic activity history."
        ),
    )

    from memorymesh.server.tools import (
        ask_memory,
        forget_memory,
        forget_source,
        get_document,
        get_entity,
        graph_memory,
        index_now,
        list_sources,
        pin_memory,
        query_timeline,
        related_documents,
        search_by_date,
        search_memory,
        summarize_source,
        sync_source,
    )

    search_memory.register(mcp, ctx)
    list_sources.register(mcp, ctx)
    get_document.register(mcp, ctx)
    index_now.register(mcp, ctx)
    ask_memory.register(mcp, ctx)
    pin_memory.register(mcp, ctx)
    forget_memory.register(mcp, ctx)
    query_timeline.register(mcp, ctx)
    sync_source.register(mcp, ctx)
    get_entity.register(mcp, ctx)
    related_documents.register(mcp, ctx)
    search_by_date.register(mcp, ctx)
    forget_source.register(mcp, ctx)
    summarize_source.register(mcp, ctx)
    graph_memory.register(mcp, ctx)

    return mcp
