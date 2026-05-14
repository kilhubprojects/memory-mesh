"""MCP tool: forget_source.

Removes all indexed documents belonging to a named source from the vector
store, BM25 index, and SQLite metadata store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``forget_source`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def forget_source(
        source: Annotated[str, "Name of the source to remove from the index."],
        dry_run: Annotated[
            bool,
            "If true, report what would be removed without actually removing it.",
        ] = False,
    ) -> dict:
        """Remove all indexed content from a named source.

        Deletes every document belonging to *source* from the vector store,
        BM25 index, and SQLite metadata.  This is irreversible unless the
        source is re-synced.

        Args:
            source: Source name as it appears in the index (e.g. ``"jira"``).
            dry_run: Preview without deleting.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "delete")) is not None:
            return err

        records = [r for r in ctx.metadata_store.list_files() if r.source_name == source]

        if not records:
            return {
                "status": "not_found",
                "source": source,
                "message": f"No indexed documents found for source '{source}'.",
            }

        if dry_run:
            return {
                "status": "dry_run",
                "source": source,
                "would_remove": len(records),
                "paths": [r.path for r in records[:20]],
            }

        removed = 0
        errors = 0
        for rec in records:
            try:
                ctx.vector_store.delete_by_path(rec.path)
                ctx.bm25.delete_by_path(rec.path)
                ctx.metadata_store.mark_deleted(rec.path)
                removed += 1
            except Exception as exc:
                from loguru import logger

                logger.warning(f"forget_source: failed to remove {rec.path!r}: {exc}")
                errors += 1

        return {
            "status": "ok",
            "source": source,
            "removed": removed,
            "errors": errors,
        }
