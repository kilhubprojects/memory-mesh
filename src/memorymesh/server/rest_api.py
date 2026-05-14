"""FastAPI REST API for MemoryMesh.

Mounts at ``/api`` alongside the MCP server.  Provides a conventional
HTTP interface for clients that cannot use the MCP protocol directly
(browser extensions, shell scripts, external dashboards).

Endpoints
---------
GET  /api/health               - liveness probe (no auth required)
GET  /api/sources              - list indexed sources (requires read)
GET  /api/search               - hybrid/dense/sparse search (requires read)
GET  /api/document/{path}      - retrieve document chunks (requires read)
POST /api/index                - trigger indexing (requires index)
GET  /api/entities             - list extracted named entities (requires read)
GET  /api/graph                - entity co-occurrence graph (requires read)
GET  /api/tiers                - memory tier distribution (requires read)
GET  /api/timeline             - episodic event timeline (requires read)
POST /api/memory/forget        - suppress a chunk from search (requires delete)
GET  /api/openapi.json         - OpenAPI schema (JSON, no auth required)

Authentication
--------------
When ``auth.enabled: true``, all endpoints except ``/health`` require an
``Authorization: Bearer <token>`` header.  The token value is resolved
against the configured ``agents`` list.  Unknown tokens receive the
``default_permission`` level defined in ``auth`` config.

Unauthenticated and authorised-but-denied requests return HTTP 403.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext

from fastapi import Request


def build_rest_api(app_ctx: AppContext) -> Any:
    """Create and return a FastAPI application wired to *app_ctx*.

    The returned app is intended to be mounted at ``/api`` inside the main
    Starlette application tree.

    Args:
        app_ctx: Fully initialised application context.

    Returns:
        A ``fastapi.FastAPI`` instance with all REST routes registered.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    from fastapi import FastAPI, HTTPException, Query

    api = FastAPI(
        title="MemoryMesh REST API",
        version="0.8.0",
        description="HTTP interface for the MemoryMesh personal knowledge base.",
        docs_url="/docs",
        redoc_url=None,
    )

    def _auth(request: Request, action: str, *, source: str | None = None) -> None:
        """Run auth guard for a REST endpoint; raises HTTPException on denial."""
        from memorymesh.server.auth_guard import check_access_http

        check_access_http(
            app_ctx,
            cast(Literal["read", "index", "delete", "admin"], action),
            source=source,
            authorization=request.headers.get("Authorization"),
        )

    @api.get("/health", summary="Liveness probe")
    async def health() -> dict:
        """Return ``{"status": "ok"}`` with uptime and version.

        No authentication required - health checks must always succeed.
        """
        from memorymesh import __version__

        return {"status": "ok", "version": __version__, "ts": time.time()}

    @api.get("/sources", summary="List indexed sources")
    async def list_sources(request: Request) -> dict:
        """Return all configured sources with indexing statistics.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        try:
            records = app_ctx.metadata_store.list_files()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        sources: dict[str, dict] = {}
        for src in app_ctx.config.sources:
            name = src.name or str(src.path)
            src_records = [r for r in records if r.source_name == name]
            sources[name] = {
                "path": str(src.path),
                "n_indexed": sum(1 for r in src_records if r.status == "indexed"),
                "n_errors": sum(1 for r in src_records if r.status == "parse_error"),
                "n_chunks": sum(r.n_chunks for r in src_records if r.status == "indexed"),
            }
        return {"sources": sources}

    @api.get("/search", summary="Semantic / hybrid search")
    async def search(
        request: Request,
        q: str = Query(..., description="Natural-language query."),
        top_k: int = Query(10, ge=1, le=50, description="Max results."),
        mode: str = Query("hybrid", description="'hybrid', 'dense', or 'sparse'."),
        modality: str = Query("all", description="'all', 'text', or 'image'."),
        source: str | None = Query(None, description="Restrict to one source."),
    ) -> dict:
        """Run a search query and return ranked hits.

        Requires ``read`` permission.  When a ``source`` filter is specified,
        the auth layer also verifies source-level ACL access.
        """
        _auth(request, "read", source=source)
        try:
            resp = app_ctx.engine.search(
                q, top_k=top_k, mode=mode, modality=modality, source=source
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "mode": resp.mode,
            "duration_ms": resp.duration_ms,
            "total_hits": len(resp.hits),
            "hits": [
                {
                    "path": h.path,
                    "chunk_index": h.chunk_index,
                    "score": h.score,
                    "preview": h.preview,
                    "file_type": h.file_type,
                    "source": h.source_root,
                }
                for h in resp.hits
            ],
        }

    @api.get("/document/{file_path:path}", summary="Get document chunks")
    async def get_document(file_path: str, request: Request) -> dict:
        """Return all indexed chunks for the given file path.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        try:
            records = app_ctx.metadata_store.get_file(file_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if records is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "path": file_path,
            "status": records.status,
            "n_chunks": records.n_chunks,
            "source": records.source_name,
        }

    @api.post("/index", summary="Trigger indexing", status_code=202)
    async def trigger_index(request: Request, body: dict | None = None) -> dict:
        """Queue an immediate re-index of all sources.  Returns 202 Accepted.

        Requires ``index`` permission.
        """
        _auth(request, "index")
        try:
            for src in app_ctx.config.sources:
                src_name = src.name or str(src.path)
                if src.path.exists():
                    app_ctx.indexer.index_directory(
                        src.path,
                        source_name=src_name,
                        recursive=src.recursive,
                        extensions=src.extensions or None,
                    )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "accepted", "message": "Indexing started."}

    @api.get("/entities", summary="List extracted entities")
    async def entities(
        request: Request,
        entity_type: str | None = Query(None),
        min_mentions: int = Query(1, ge=1),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict:
        """Return named entities ranked by mention count.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        try:
            ents = app_ctx.metadata_store.list_entities(
                entity_type=entity_type,
                min_mentions=min_mentions,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "total": len(ents),
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "mentions": e.mention_count,
                }
                for e in ents
            ],
        }

    @api.get("/graph", summary="Entity co-occurrence graph")
    async def graph(
        request: Request,
        min_mentions: int = Query(2, ge=1),
        entity_type: str | None = Query(None),
    ) -> dict:
        """Return the knowledge graph as nodes + edges.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        from memorymesh.server.dashboard import _build_knowledge_graph

        return _build_knowledge_graph(app_ctx, min_mentions=min_mentions, entity_type=entity_type)

    @api.get("/tiers", summary="Memory tier distribution")
    async def tiers(request: Request) -> dict:
        """Return hot/warm/cold chunk counts.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        if app_ctx.tiered_memory is None:
            return {"enabled": False}

        from memorymesh.core.models import MemoryTier

        counts: dict[str, int] = {}
        for tier in MemoryTier:
            try:
                entries = app_ctx.metadata_store.list_chunks_by_tier(tier, limit=None)
                counts[tier.value] = len(entries)
            except Exception:
                counts[tier.value] = 0
        return {"enabled": True, "tiers": counts}

    @api.get("/timeline", summary="Episodic event timeline")
    async def timeline(
        request: Request,
        days: float = Query(7.0, ge=0.1),
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict:
        """Return episodic events from the last *days* days.

        Requires ``read`` permission.
        """
        _auth(request, "read")
        if not app_ctx.config.memory.episodic.enabled:
            return {"enabled": False, "events": []}

        since_ts = time.time() - days * 86_400
        try:
            events = app_ctx.metadata_store.list_episodic_events(since=since_ts, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "enabled": True,
            "events": [
                {
                    "event_id": ev.event_id,
                    "ts": ev.timestamp,
                    "type": ev.event_type,
                    "source": ev.source,
                    "n_chunks": len(ev.chunk_ids),
                }
                for ev in events
            ],
        }

    @api.post("/memory/forget", summary="Suppress a chunk from search")
    async def forget(request: Request, body: dict) -> dict:
        """Add a chunk to the suppression list so it is hidden from all future
        search results.

        This is a **semantic suppression** operation (the chunk remains in the
        index but is filtered from :class:`~memorymesh.search.engine.SearchEngine`
        results).  It is different from tier-based forgetting:

        - ``POST /memory/forget`` - chunk hidden from search immediately
          (``forgotten_chunks`` table).  Use when you never want to see this
          chunk again.
        - MCP ``forget_memory`` tool - demotes chunk to cold tier so its
          relevance score decays over time.  Use when you want the chunk to
          gradually fade, not vanish instantly.

        Request body: ``{"chunk_id": "<path>:<chunk_index>"}``

        Requires ``delete`` permission.
        """
        _auth(request, "delete")
        chunk_id = body.get("chunk_id", "")
        if not chunk_id:
            raise HTTPException(status_code=422, detail="'chunk_id' is required.")
        try:
            app_ctx.metadata_store.forget_chunk(chunk_id)
        except AttributeError as exc:
            raise HTTPException(status_code=501, detail="forget_chunk not implemented") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "forgotten", "chunk_id": chunk_id}

    return api
