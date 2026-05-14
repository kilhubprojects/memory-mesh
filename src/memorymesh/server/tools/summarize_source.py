"""MCP tool: summarize_source.

Returns indexing statistics and a content overview for one or all sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``summarize_source`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def summarize_source(
        source: Annotated[
            str | None,
            "Source name to summarize.  Omit to summarize all sources.",
        ] = None,
    ) -> dict:
        """Return indexing statistics for one or all named sources.

        Reports the number of indexed documents, total chunks, file type
        breakdown, and most recent indexed-at timestamp for each source.

        Args:
            source: Source name, or ``null`` for all sources.
        """
        from datetime import UTC, datetime

        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        records = ctx.metadata_store.list_files(status="indexed")

        if source:
            records = [r for r in records if r.source_name == source]
            if not records:
                return {
                    "status": "not_found",
                    "source": source,
                    "message": f"No indexed documents found for source '{source}'.",
                }

        groups: dict[str, list] = {}
        for rec in records:
            groups.setdefault(rec.source_name or "unknown", []).append(rec)

        def _iso(ts: float | None) -> str:
            if not ts:
                return ""
            return datetime.fromtimestamp(ts, tz=UTC).isoformat()

        summaries = []
        for src_name, recs in sorted(groups.items()):
            n_indexed = len(recs)
            total_chunks = sum(r.n_chunks for r in recs)
            latest_ts = max((r.indexed_at or 0) for r in recs)

            type_counts: dict[str, int] = {}
            for r in recs:
                type_counts[r.file_type] = type_counts.get(r.file_type, 0) + 1

            summaries.append(
                {
                    "source": src_name,
                    "documents": n_indexed,
                    "chunks": total_chunks,
                    "file_types": type_counts,
                    "last_indexed": _iso(latest_ts if latest_ts > 0 else None),
                }
            )

        return {
            "status": "ok",
            "sources": summaries,
            "total_documents": sum(s["documents"] for s in summaries),
            "total_chunks": sum(s["chunks"] for s in summaries),
        }
