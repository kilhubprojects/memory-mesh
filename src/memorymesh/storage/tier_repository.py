"""Repository for chunk memory-tier records.

:class:`TierRepository` owns the ``chunk_tiers`` table.  Used by the
tiered memory manager (Wave 3) to track hot/warm/cold tier assignments and
access frequency per chunk.
"""

from __future__ import annotations

import time

from memorymesh.core.models import ChunkTierRecord, MemoryTier
from memorymesh.storage.db import ConnectionFactory


class TierRepository:
    """Manages the ``chunk_tiers`` table.

    Args:
        connection_factory: Zero-argument callable that returns the shared
            :class:`sqlite3.Connection`.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._conn = connection_factory

    def set_chunk_tier(self, record: ChunkTierRecord) -> None:
        """Upsert the tier record for a chunk.

        Args:
            record: The :class:`~memorymesh.core.models.ChunkTierRecord` to persist.
        """
        self._conn().execute(
            """
            INSERT INTO chunk_tiers(chunk_id, tier, last_accessed, access_count, pinned)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                tier          = excluded.tier,
                last_accessed = excluded.last_accessed,
                access_count  = excluded.access_count,
                pinned        = excluded.pinned
            """,
            (
                record.chunk_id,
                record.tier.value,
                record.last_accessed,
                record.access_count,
                int(record.pinned),
            ),
        )
        self._conn().commit()

    def get_chunk_tier(self, chunk_id: str) -> ChunkTierRecord | None:
        """Return the tier record for *chunk_id*, or ``None`` if not tracked.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        row = (
            self._conn()
            .execute("SELECT * FROM chunk_tiers WHERE chunk_id = ?", (chunk_id,))
            .fetchone()
        )
        if row is None:
            return None
        return ChunkTierRecord(
            chunk_id=row["chunk_id"],
            tier=MemoryTier(row["tier"]),
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            pinned=bool(row["pinned"]),
        )

    def record_chunk_access(self, chunk_id: str) -> None:
        """Increment access_count and refresh last_accessed for *chunk_id*.

        Creates the row with ``tier='warm'`` if it does not exist yet.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        now = time.time()
        self._conn().execute(
            """
            INSERT INTO chunk_tiers(chunk_id, tier, last_accessed, access_count, pinned)
            VALUES(?, 'warm', ?, 1, 0)
            ON CONFLICT(chunk_id) DO UPDATE SET
                last_accessed = ?,
                access_count  = access_count + 1
            """,
            (chunk_id, now, now),
        )
        self._conn().commit()

    def list_chunks_by_tier(
        self,
        tier: MemoryTier,
        limit: int | None = None,
    ) -> list[ChunkTierRecord]:
        """Return all chunk tier records for a given *tier*.

        Args:
            tier: The tier to filter on.
            limit: Maximum number of rows to return (``None`` = all).
        """
        sql = "SELECT * FROM chunk_tiers WHERE tier = ? ORDER BY last_accessed DESC"
        params: list[object] = [tier.value]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [
            ChunkTierRecord(
                chunk_id=r["chunk_id"],
                tier=MemoryTier(r["tier"]),
                last_accessed=r["last_accessed"],
                access_count=r["access_count"],
                pinned=bool(r["pinned"]),
            )
            for r in rows
        ]

    def promote_chunks_to_tier(
        self,
        chunk_ids: list[str],
        tier: MemoryTier,
    ) -> int:
        """Bulk-update the tier for a list of chunk IDs.

        Only updates rows that already exist in ``chunk_tiers``.

        Args:
            chunk_ids: List of ``<path>:<chunk_index>`` identifiers.
            tier: Target tier value.

        Returns:
            Number of rows actually updated.
        """
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" * len(chunk_ids))
        cur = self._conn().execute(
            f"UPDATE chunk_tiers SET tier = ? WHERE chunk_id IN ({placeholders}) AND pinned = 0",
            [tier.value, *chunk_ids],
        )
        self._conn().commit()
        return cur.rowcount
