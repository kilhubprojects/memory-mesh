"""MCP tool: pin_memory / unpin_memory.

Allows agents to manually pin chunks to the hot memory tier (keeping them
perpetually accessible with full relevance scores) or unpin them to return
to normal access-frequency-based tiering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register ``pin_memory`` and ``unpin_memory`` tools on *mcp*.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def pin_memory(
        chunk_id: Annotated[
            str,
            "The chunk identifier to pin, in '<absolute_path>:<chunk_index>' format. "
            "Chunk IDs are returned in search_memory results.",
        ],
    ) -> dict:
        """Pin a specific chunk to the hot memory tier.

        Pinned chunks are never demoted during maintenance runs and always
        receive full relevance scores (no decay) regardless of how long ago
        they were last accessed.

        Args:
            chunk_id: Stable chunk identifier ``<path>:<chunk_index>``.

        Returns:
            Confirmation dict with ``chunk_id``, ``tier``, and ``pinned`` fields.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "index")) is not None:
            return err

        if ctx.tiered_memory is None:
            return {"error": "Memory tiers are not enabled in the current configuration."}
        ctx.tiered_memory.pin(chunk_id)
        ctx.audit_logger.log_query(
            tool="pin_memory",
            query=chunk_id,
            n_results=0,
            latency_ms=0.0,
        )
        return {"chunk_id": chunk_id, "tier": "hot", "pinned": True}

    @mcp.tool()
    def unpin_memory(
        chunk_id: Annotated[
            str,
            "The chunk identifier to unpin, in '<absolute_path>:<chunk_index>' format.",
        ],
    ) -> dict:
        """Remove the manual pin from a chunk, allowing normal tiering.

        The chunk stays in the hot tier until the next maintenance run, after
        which it will be promoted or demoted based on its access history.

        Args:
            chunk_id: Stable chunk identifier ``<path>:<chunk_index>``.

        Returns:
            Confirmation dict with ``chunk_id`` and ``pinned`` fields.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "index")) is not None:
            return err

        if ctx.tiered_memory is None:
            return {"error": "Memory tiers are not enabled in the current configuration."}
        ctx.tiered_memory.unpin(chunk_id)
        ctx.audit_logger.log_query(
            tool="unpin_memory",
            query=chunk_id,
            n_results=0,
            latency_ms=0.0,
        )
        return {"chunk_id": chunk_id, "pinned": False}
