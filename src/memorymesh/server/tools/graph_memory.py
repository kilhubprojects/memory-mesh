"""MCP tool: graph_memory.

Exposes the in-memory knowledge graph derived from entity co-occurrence.
Two entities are connected when they appear in the same indexed chunk.
Edge weight = number of shared chunks.

This tool is intentionally read-only and pure-function — the graph is computed
on the fly (with a 60 s server-side cache in the dashboard) and never mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``graph_memory`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def graph_memory(
        min_mentions: Annotated[
            int,
            "Only include entities with at least this many mentions (default 2).",
        ] = 2,
        entity_type: Annotated[
            str | None,
            "Restrict to a specific entity type (e.g. 'PERSON', 'ORG'). Omit for all.",
        ] = None,
    ) -> dict:
        """Return the entity co-occurrence knowledge graph.

        Nodes are named entities extracted from the indexed corpus.  An edge
        between two entities means they were mentioned in the same document
        chunk; the edge weight is the number of shared chunks.

        Args:
            min_mentions: Minimum mention count for a node to appear.
            entity_type: Optional entity type filter (e.g. ``"PERSON"``).

        Returns:
            Dict with ``"nodes"`` (id, label, type, mentions) and ``"edges"``
            (source, target, weight, shared_chunks) lists.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        try:
            entities = ctx.metadata_store.list_entities(
                min_mentions=min_mentions,
                entity_type=entity_type,
                limit=200,
            )
        except Exception as exc:
            return {"error": str(exc), "nodes": [], "edges": []}

        entity_chunks: dict[str, set[str]] = {}
        for ent in entities:
            try:
                chunk_ids = ctx.metadata_store.get_entity_chunks(ent.name, ent.entity_type)
            except Exception:
                chunk_ids = []
            entity_chunks[ent.name] = set(chunk_ids)

        nodes = [
            {
                "id": ent.name,
                "label": ent.name,
                "type": ent.entity_type,
                "mentions": ent.mention_count,
            }
            for ent in entities
        ]

        entity_list = list(entities)
        edges: list[dict] = []
        for i, a in enumerate(entity_list):
            for b in entity_list[i + 1 :]:
                shared = entity_chunks[a.name] & entity_chunks[b.name]
                if shared:
                    edges.append(
                        {
                            "source": a.name,
                            "target": b.name,
                            "weight": len(shared),
                            "shared_chunks": sorted(shared),
                        }
                    )

        ctx.audit_logger.log_query(
            tool="graph_memory",
            query=f"min_mentions={min_mentions} entity_type={entity_type}",
            n_results=len(nodes),
            latency_ms=0.0,
        )

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }
