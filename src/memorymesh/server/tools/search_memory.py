"""MCP tool: search_memory.

Performs hybrid (dense + sparse) or pure dense/sparse semantic search over
all indexed documents and returns ranked passages with context.

Wave 3 additions:
- Records chunk access events in the tiered memory manager (hot/warm/cold).
- Auto-records retrieval events to the episodic memory timeline.
- Applies forgetting-policy score decay to cold-tier chunks when enabled.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``search_memory`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def search_memory(
        query: Annotated[str, "The natural-language query to search for."],
        top_k: Annotated[int, "Maximum number of results to return (1-50)."] = 10,
        mode: Annotated[
            str,
            "Search mode: 'hybrid' (default), 'dense' (semantic only), or 'sparse' (BM25 only).",
        ] = "hybrid",
        source: Annotated[
            str | None,
            "Restrict results to a specific source name. Omit to search all sources.",
        ] = None,
        modality: Annotated[
            str,
            "Modality filter: 'all' (default), 'text' (exclude images), 'image' (images only).",
        ] = "all",
    ) -> dict:
        """Search the personal knowledge base for passages relevant to *query*.

        Returns a ranked list of matching text passages with file paths,
        relevance scores, and short previews.  When ``search.parent_window_chars``
        is configured, each hit also includes an ``extended_preview`` with wider
        context around the matched chunk.

        Args:
            query: Natural-language question or phrase.
            top_k: Number of results (clamped to 1-50).
            mode: ``hybrid`` | ``dense`` | ``sparse``.
            source: Optional source name filter.
            modality: ``all`` | ``text`` | ``image``.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read", source=source)) is not None:
            return err

        t0 = time.perf_counter()
        top_k = max(1, min(50, top_k))
        filter_: dict | None = {"source_root": source} if source else None

        response = ctx.engine.search(
            query, top_k=top_k, mode=mode, filter_=filter_, modality=modality
        )

        if ctx.tiered_memory is not None and response.hits:
            chunk_ids = [f"{h.path}:{h.chunk_index}" for h in response.hits]

            # Record accesses (promotes recently retrieved chunks toward hot).
            for cid in chunk_ids:
                ctx.tiered_memory.record_access(cid)

            # Apply forgetting-policy score decay to cold-tier chunks.
            scored = [(f"{h.path}:{h.chunk_index}", h.score) for h in response.hits]
            decayed = ctx.tiered_memory.apply_decay(scored)
            score_map = {cid: s for cid, s in decayed}
            for hit in response.hits:
                cid = f"{hit.path}:{hit.chunk_index}"
                hit.score = score_map.get(cid, hit.score)

        if (
            ctx.config.memory.episodic.enabled
            and ctx.config.memory.episodic.auto_record
            and response.hits
        ):
            from memorymesh.core.models import EpisodicEvent

            chunk_ids_for_event = [f"{h.path}:{h.chunk_index}" for h in response.hits]
            event = EpisodicEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type="retrieval",
                source=source or "",
                chunk_ids=chunk_ids_for_event,
                client_id="",
                metadata={"query_hash": str(hash(query)), "mode": mode},
            )
            with contextlib.suppress(Exception):
                ctx.metadata_store.upsert_episodic_event(event)

        window = ctx.config.search.parent_window_chars
        if window > 0:
            from memorymesh.search.context import expand_context

            for hit in response.hits:
                extended = expand_context(hit.path, hit.start_char, hit.end_char, window)
                hit.extended_preview = extended if extended else None

        latency_ms = (time.perf_counter() - t0) * 1000
        ctx.audit_logger.log_query(
            tool="search_memory",
            query=query,
            n_results=len(response.hits),
            latency_ms=latency_ms,
        )

        return {
            "mode": response.mode,
            "duration_ms": round(response.duration_ms, 2),
            "total_hits": len(response.hits),
            "hits": [
                {
                    "path": h.path,
                    "chunk_index": h.chunk_index,
                    "score": h.score,
                    "preview": h.preview,
                    "extended_preview": h.extended_preview,
                    "file_type": h.file_type,
                    "source": h.source_root,
                }
                for h in response.hits
            ],
        }
