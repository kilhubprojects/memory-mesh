"""MCP tool: query_timeline.

Exposes the episodic event log so agents can ask temporal questions like
"what was I working on last week?" or "which files were retrieved in April?".

Events are recorded automatically on each search_memory call (when
``memory.episodic.auto_record: true``) and can also be written explicitly by
agents via this tool.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

from memorymesh.core.models import EpisodicEvent

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``query_timeline`` and ``record_event`` tools on *mcp*.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def query_timeline(
        since_days: Annotated[
            float,
            "Return events from the last N days. Default 7.",
        ] = 7.0,
        event_type: Annotated[
            str | None,
            "Filter by event type: 'retrieval', 'index', 'pin', 'forget', "
            "'user_note'. Omit to return all types.",
        ] = None,
        limit: Annotated[
            int,
            "Maximum number of events to return (1-500).",
        ] = 50,
    ) -> dict:
        """Query the episodic memory timeline for recent activity.

        Returns a chronological log of retrieval and indexing events, allowing
        an agent to reconstruct what the user was working on during a given
        period.

        Args:
            since_days: How many days back to look.
            event_type: Optional category filter.
            limit: Maximum results.

        Returns:
            Dict with ``events`` list (each event has ``event_id``, ``timestamp``,
            ``event_type``, ``source``, ``chunk_ids``, ``client_id``,
            ``metadata``), ``total`` count, and ``since_ts`` / ``until_ts`` bounds.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        if not ctx.config.memory.episodic.enabled:
            return {"error": "Episodic memory is not enabled in the current configuration."}

        limit = max(1, min(500, limit))
        now = time.time()
        since_ts = now - (since_days * 86_400.0)

        events = ctx.metadata_store.list_episodic_events(
            since=since_ts,
            event_type=event_type,
            limit=limit,
        )

        return {
            "events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "source": e.source,
                    "chunk_ids": e.chunk_ids,
                    "client_id": e.client_id,
                    "metadata": e.metadata,
                }
                for e in events
            ],
            "total": len(events),
            "since_ts": since_ts,
            "until_ts": now,
        }

    @mcp.tool()
    def record_event(
        event_type: Annotated[
            str,
            "Event category: 'user_note', 'pin', 'forget', or any custom string.",
        ],
        source: Annotated[
            str,
            "File path or source name associated with this event.",
        ] = "",
        note: Annotated[
            str,
            "Optional free-text note to attach to this event.",
        ] = "",
        chunk_ids: Annotated[
            list[str],
            "Optional list of chunk IDs involved in this event.",
        ] = [],  # noqa: B006 — FastMCP reads default literally for schema generation
    ) -> dict:
        """Record a custom episodic event in the memory timeline.

        Useful for agents to annotate significant moments (e.g. "user reviewed
        this document", "agent summarised this file") without triggering a
        search.

        Args:
            event_type: Category string for the event.
            source: Associated file path or source name.
            note: Free-text annotation stored in ``metadata.note``.
            chunk_ids: Chunk identifiers involved.

        Returns:
            The persisted event dict with its generated ``event_id``.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "index")) is not None:
            return err

        if not ctx.config.memory.episodic.enabled:
            return {"error": "Episodic memory is not enabled in the current configuration."}

        event = EpisodicEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            source=source,
            chunk_ids=list(chunk_ids),
            client_id="",
            metadata={"note": note} if note else {},
        )
        ctx.metadata_store.upsert_episodic_event(event)
        ctx.audit_logger.log_query(
            tool="record_event",
            query=f"{event_type}:{source}",
            n_results=0,
            latency_ms=0.0,
        )
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "source": event.source,
        }
