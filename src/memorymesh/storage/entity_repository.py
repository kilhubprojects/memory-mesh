"""Repository for named entity records and mention links.

:class:`EntityRepository` owns the ``entities`` and ``entity_mentions`` tables.
Entities are extracted by the optional NLP pipeline during indexing and
aggregated here for graph and entity-listing queries.
"""

from __future__ import annotations

from memorymesh.core.models import Entity
from memorymesh.storage.db import ConnectionFactory


class EntityRepository:
    """Manages the ``entities`` and ``entity_mentions`` tables.

    Args:
        connection_factory: Zero-argument callable that returns the shared
            :class:`sqlite3.Connection`.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._conn = connection_factory

    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity record.

        On conflict (same ``name`` + ``entity_type``), increments
        ``mention_count``, updates ``last_seen``, and preserves ``first_seen``.

        Args:
            entity: The :class:`~memorymesh.core.models.Entity` to persist.
        """
        self._conn().execute(
            """
            INSERT INTO entities(name, entity_type, mention_count, first_seen, last_seen)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(name, entity_type) DO UPDATE SET
                mention_count = mention_count + excluded.mention_count,
                last_seen     = excluded.last_seen
            """,
            (
                entity.name,
                entity.entity_type,
                entity.mention_count,
                entity.first_seen,
                entity.last_seen,
            ),
        )
        self._conn().commit()

    def list_entities(
        self,
        entity_type: str | None = None,
        min_mentions: int = 1,
        limit: int = 50,
    ) -> list[Entity]:
        """Return entities ranked by mention count.

        Args:
            entity_type: Filter by type.  ``None`` = all types.
            min_mentions: Minimum mention count to include.
            limit: Maximum entities to return.
        """
        clauses: list[str] = ["mention_count >= ?"]
        params: list[object] = [min_mentions]
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        where = "WHERE " + " AND ".join(clauses)
        params.append(limit)
        rows = (
            self._conn()
            .execute(
                f"SELECT * FROM entities {where} ORDER BY mention_count DESC LIMIT ?",
                params,
            )
            .fetchall()
        )
        return [
            Entity(
                name=r["name"],
                entity_type=r["entity_type"],
                mention_count=r["mention_count"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
            )
            for r in rows
        ]

    def get_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> Entity | None:
        """Return the entity record for *name*, or ``None`` if not found.

        Args:
            name: Canonical entity name (case-sensitive after normalisation).
            entity_type: Optional type filter.  When provided, only returns the
                entity if its type matches exactly.  When ``None``, returns the
                entity with the highest mention count across all types.
        """
        if entity_type is not None:
            row = (
                self._conn()
                .execute(
                    "SELECT * FROM entities WHERE name = ? AND entity_type = ? LIMIT 1",
                    (name, entity_type),
                )
                .fetchone()
            )
        else:
            row = (
                self._conn()
                .execute(
                    "SELECT * FROM entities WHERE name = ? ORDER BY mention_count DESC LIMIT 1",
                    (name,),
                )
                .fetchone()
            )
        if row is None:
            return None
        return Entity(
            name=row["name"],
            entity_type=row["entity_type"],
            mention_count=row["mention_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    def add_entity_mention(
        self,
        entity_name: str,
        entity_type: str,
        chunk_id: str,
    ) -> None:
        """Record that *chunk_id* mentions the given entity.

        Silently ignores duplicate (entity_name, entity_type, chunk_id) triplets.

        Args:
            entity_name: Canonical entity name.
            entity_type: Entity type string.
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        self._conn().execute(
            """
            INSERT OR IGNORE INTO entity_mentions(entity_name, entity_type, chunk_id)
            VALUES(?, ?, ?)
            """,
            (entity_name, entity_type, chunk_id),
        )
        self._conn().commit()

    def get_entity_chunks(
        self,
        entity_name: str,
        entity_type: str,
    ) -> list[str]:
        """Return all chunk IDs that mention the given entity.

        Args:
            entity_name: Canonical entity name.
            entity_type: Entity type string.

        Returns:
            List of ``<path>:<chunk_index>`` strings.
        """
        rows = (
            self._conn()
            .execute(
                "SELECT chunk_id FROM entity_mentions WHERE entity_name = ? AND entity_type = ?",
                (entity_name, entity_type),
            )
            .fetchall()
        )
        return [r["chunk_id"] for r in rows]

    def get_entity_chunk_ids(self, name: str) -> list[str]:
        """Return chunk IDs for all mentions of *name* across all entity types.

        Args:
            name: Canonical entity name.

        Returns:
            List of ``<path>:<chunk_index>`` strings.
        """
        return self.get_entity_mentions(name)

    def get_entity_mentions(self, name: str) -> list[str]:
        """Return all chunk IDs that mention the entity *name* (any type).

        Args:
            name: Canonical entity name.

        Returns:
            List of ``<path>:<chunk_index>`` strings.
        """
        rows = (
            self._conn()
            .execute(
                "SELECT chunk_id FROM entity_mentions WHERE entity_name = ?",
                (name,),
            )
            .fetchall()
        )
        return [r["chunk_id"] for r in rows]
