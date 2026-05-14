"""MCP tool: get_entity.

Returns a named entity (person, project, concept, etc.) from the entity
store along with its list of related chunk IDs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``get_entity`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def get_entity(
        name: Annotated[str, "Entity name to look up (case-insensitive)."],
        entity_type: Annotated[
            str | None,
            "Optional entity type filter, e.g. 'person' or 'project'.",
        ] = None,
    ) -> dict:
        """Retrieve a named entity and its associated document chunks.

        Returns the entity's name, type, and the IDs of all chunks that
        mention it.  Requires entity extraction to have been enabled when
        the documents were indexed (``indexing.entity_extraction.enabled``).

        Args:
            name: Entity name (case-insensitive).
            entity_type: Optional type constraint.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        entity = ctx.metadata_store.get_entity(name.lower())
        if entity is None:
            return {
                "found": False,
                "name": name,
                "message": "Entity not found in the knowledge base.",
            }

        if entity_type and entity.entity_type != entity_type.lower():
            return {
                "found": False,
                "name": name,
                "message": (
                    f"Entity found but type mismatch: "
                    f"expected {entity_type!r}, got {entity.entity_type!r}."
                ),
            }

        mentions = ctx.metadata_store.get_entity_mentions(name.lower())

        return {
            "found": True,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "mention_count": len(mentions),
            "chunk_ids": mentions[:100],
        }
