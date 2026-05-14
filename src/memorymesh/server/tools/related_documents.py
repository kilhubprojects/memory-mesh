"""MCP tool: related_documents.

Finds documents semantically similar to a given document path or chunk ID
using vector similarity search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``related_documents`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def related_documents(
        path: Annotated[str, "File path or synthetic URI to find related documents for."],
        top_k: Annotated[int, "Number of related documents to return (1-20)."] = 5,
        exclude_self: Annotated[
            bool,
            "When true, exclude chunks from the same document path.",
        ] = True,
    ) -> dict:
        """Find documents semantically related to a given file or connector document.

        Retrieves the first chunk of the target document, embeds it, and
        performs a vector similarity search to find the most related content.

        Args:
            path: Absolute file path or synthetic connector URI.
            top_k: Number of similar documents to return.
            exclude_self: Whether to exclude the source document from results.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        top_k = max(1, min(20, top_k))

        # Fetch chunks for this path from the vector store.
        fetch_k = top_k * 3 if exclude_self else top_k
        results = ctx.vector_store.get_by_path(path, limit=1)

        if not results:
            return {
                "status": "not_found",
                "path": path,
                "message": "No chunks found for this path in the index.",
            }

        # Use the first chunk's text as the query.
        anchor_text = str(results[0].get("text") or results[0].get("document") or "")
        if not anchor_text:
            return {
                "status": "error",
                "message": "Could not retrieve text for the anchor chunk.",
            }

        query_embedding = ctx.provider.embed_query(anchor_text)
        hits_raw = ctx.vector_store.search(
            query_embedding,
            top_k=fetch_k,
            filter_=None,
        )

        seen_paths: set[str] = set()
        related: list[dict] = []

        for h in hits_raw:
            hit_path = h.path
            if exclude_self and hit_path == path:
                continue
            if hit_path in seen_paths:
                continue
            seen_paths.add(hit_path)
            related.append(
                {
                    "path": hit_path,
                    "score": round(h.score, 4),
                    "preview": h.preview[:200],
                }
            )
            if len(related) >= top_k:
                break

        return {
            "status": "ok",
            "anchor_path": path,
            "related": related,
        }
