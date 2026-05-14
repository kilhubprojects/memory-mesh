"""Repository for chunk suppression (forget) records.

:class:`SuppressionRepository` owns the ``forgotten_chunks`` table.  Chunks
added here are hidden from all future search results by the search engine's
post-filter step, regardless of their tier or vector score.
"""

from __future__ import annotations

from memorymesh.storage.db import ConnectionFactory


class SuppressionRepository:
    """Manages the ``forgotten_chunks`` suppression table.

    Args:
        connection_factory: Zero-argument callable that returns the shared
            :class:`sqlite3.Connection`.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._conn = connection_factory

    def forget_chunk(self, chunk_id: str) -> None:
        """Suppress a chunk from future search results.

        Adds the *chunk_id* to the ``forgotten_chunks`` suppression table.  The
        search engine checks this table via :meth:`list_forgotten` and omits
        matching hits.

        Args:
            chunk_id: ``<path>:<chunk_index>`` identifier of the chunk to forget.
        """
        self._conn().execute(
            "INSERT OR IGNORE INTO forgotten_chunks(chunk_id) VALUES(?)",
            (chunk_id,),
        )
        self._conn().commit()

    def list_forgotten(self) -> list[str]:
        """Return all chunk IDs currently in the suppression list.

        Returns:
            List of ``<path>:<chunk_index>`` strings that are hidden from search.
            Empty list when no chunks have been suppressed.
        """
        rows = self._conn().execute("SELECT chunk_id FROM forgotten_chunks").fetchall()
        return [r["chunk_id"] for r in rows]
