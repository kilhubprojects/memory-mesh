"""MCP tool: forget_memory.

Two forget semantics are available via the ``mode`` parameter:

``"suppress"`` (default)
    Adds the chunk to the ``forgotten_chunks`` suppression table so it is
    **immediately hidden** from all future search results.  The chunk stays in
    the index (no storage freed) but is never returned by the search engine.
    Use when you never want to see this chunk again.

``"cold"``
    Demotes the chunk to the cold memory tier so its effective relevance score
    **decays** over time via the forgetting policy.  The chunk remains visible
    in search but its score is multiplied by an exponential decay factor on
    each query.  Requires ``memory.tier.enabled: true`` in the config.  Use
    when you want content to gradually fade rather than vanish instantly.

The REST ``POST /api/memory/forget`` endpoint always uses the ``"suppress"``
semantic (it writes directly to the ``forgotten_chunks`` table).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``forget_memory`` tool on *mcp*.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def forget_memory(
        chunk_id: Annotated[
            str,
            "The chunk identifier to forget, in '<absolute_path>:<chunk_index>' format. "
            "Chunk IDs are returned by search_memory results.",
        ],
        mode: Annotated[
            Literal["suppress", "cold"],
            "'suppress' (default) — hide chunk from search immediately via the "
            "forgotten_chunks suppression table.  "
            "'cold' — demote to cold tier so the chunk's score decays gradually "
            "(requires memory tiers to be enabled).",
        ] = "suppress",
    ) -> dict:
        """Forget a chunk from memory using either instant suppression or gradual decay.

        Two modes are available:

        - **suppress** (default): chunk is added to the suppression list and
          immediately hidden from all search results.  Use when you never want
          to see this chunk again.
        - **cold**: chunk is demoted to the cold memory tier so its relevance
          score decays over time.  The chunk stays visible but becomes less
          prominent on each query.  Use when you prefer gradual fading.

        Use ``pin_memory`` to reverse a cold demotion and restore full relevance.
        A suppressed chunk can only be unsuppressed by direct database access
        (intentionally — suppression is permanent).

        Args:
            chunk_id: Stable chunk identifier ``<path>:<chunk_index>``.
            mode: ``"suppress"`` or ``"cold"``.  Defaults to ``"suppress"``.

        Returns:
            Confirmation dict with ``chunk_id``, ``mode``, and result fields.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "delete")) is not None:
            return err

        if mode == "suppress":
            try:
                ctx.metadata_store.forget_chunk(chunk_id)
            except AttributeError:
                return {"error": "forget_chunk not implemented by this metadata store."}
            except Exception as exc:
                return {"error": str(exc)}
            return {"chunk_id": chunk_id, "mode": "suppress", "suppressed": True}

        # mode == "cold"
        if ctx.tiered_memory is None:
            return {"error": "Memory tiers are not enabled in the current configuration."}
        ctx.tiered_memory.forget(chunk_id)
        ctx.audit_logger.log_query(
            tool="forget_memory",
            query=chunk_id,
            n_results=0,
            latency_ms=0.0,
        )
        return {"chunk_id": chunk_id, "mode": "cold", "tier": "cold", "pinned": False}
