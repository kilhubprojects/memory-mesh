"""Repository for episodic memory events.

:class:`EventRepository` owns the ``episodic_events`` table.  Records a
timeline of retrieval and indexing events, enabling agents to reconstruct
what was accessed during any time window.
"""

from __future__ import annotations

import json

from memorymesh.core.models import EpisodicEvent
from memorymesh.storage.db import ConnectionFactory


class EventRepository:
    """Manages the ``episodic_events`` table.

    Args:
        connection_factory: Zero-argument callable that returns the shared
            :class:`sqlite3.Connection`.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._conn = connection_factory

    def upsert_episodic_event(self, event: EpisodicEvent) -> None:
        """Insert or replace an episodic event.

        Args:
            event: The :class:`~memorymesh.core.models.EpisodicEvent` to persist.
                If ``event.event_id`` is empty a UUID must be assigned by the caller
                before this call.
        """
        self._conn().execute(
            """
            INSERT INTO episodic_events
                (event_id, timestamp, event_type, source, chunk_ids, client_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                timestamp  = excluded.timestamp,
                event_type = excluded.event_type,
                source     = excluded.source,
                chunk_ids  = excluded.chunk_ids,
                client_id  = excluded.client_id,
                metadata   = excluded.metadata
            """,
            (
                event.event_id,
                event.timestamp,
                event.event_type,
                event.source,
                json.dumps(event.chunk_ids),
                event.client_id,
                json.dumps(event.metadata),
            ),
        )
        self._conn().commit()

    def list_episodic_events(
        self,
        since: float | None = None,
        until: float | None = None,
        event_type: str | None = None,
        client_id: str | None = None,
        limit: int = 100,
    ) -> list[EpisodicEvent]:
        """Return episodic events matching the given filters.

        Args:
            since: Lower bound Unix timestamp (inclusive).  ``None`` = no lower bound.
            until: Upper bound Unix timestamp (inclusive).  ``None`` = no upper bound.
            event_type: Filter by event category.  ``None`` = all types.
            client_id: Filter by originating client.  ``None`` = all clients.
            limit: Maximum events to return (most recent first).

        Returns:
            List of :class:`~memorymesh.core.models.EpisodicEvent` in descending
            timestamp order.
        """
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if client_id is not None:
            clauses.append("client_id = ?")
            params.append(client_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = (
            self._conn()
            .execute(
                f"SELECT * FROM episodic_events {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            )
            .fetchall()
        )
        return [
            EpisodicEvent(
                event_id=r["event_id"],
                timestamp=r["timestamp"],
                event_type=r["event_type"],
                source=r["source"],
                chunk_ids=json.loads(r["chunk_ids"]),
                client_id=r["client_id"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]
