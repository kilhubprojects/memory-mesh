"""SQLite-backed metadata store for MemoryMesh.

SQLite is the **single source of truth** for the indexing state.  Every write
to ChromaDB or the BM25 pickle must be preceded by a SQLite transaction that
records the intended change, and the record is only committed *after* the
downstream write succeeds.  On startup, :meth:`MetadataStore.is_clean_state`
reveals whether the previous run shut down cleanly; if not, the daemon enters
reconciliation mode (handled by :class:`~memorymesh.indexer.file_indexer.FileIndexer`).

Schema
------
``files``           - one row per indexed (or attempted) file.
``sources``         - one row per configured source, updated on each full scan.
``index_state``     - key/value pairs for daemon lifecycle bookkeeping.
``chunk_tiers``     - access-frequency tiers for memory-primitive support (Wave 3).
``episodic_events`` - timeline of retrieval and indexing events (Wave 3).
``entities``        - extracted named entities, aggregated across chunks (Wave 3).
``entity_mentions`` - chunk-level entity occurrence links (Wave 3).
``forgotten_chunks`` - suppression list; chunks here are hidden from search (Wave 4).

Architecture
------------
:class:`MetadataStore` is a **facade** that delegates every table operation to a
dedicated repository class:

* :class:`~memorymesh.storage.file_repository.FileRepository`   - files, sources, index_state
* :class:`~memorymesh.storage.tier_repository.TierRepository`   - chunk_tiers
* :class:`~memorymesh.storage.event_repository.EventRepository` - episodic_events
* :class:`~memorymesh.storage.entity_repository.EntityRepository` - entities, entity_mentions
* :class:`~memorymesh.storage.suppression_repository.SuppressionRepository` - forgotten_chunks

All repositories share the single :class:`sqlite3.Connection` managed by
:meth:`MetadataStore._connection`.

Privacy invariant: this module never logs file contents or query text - only
paths, hashes, counts, and operational events.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from loguru import logger

from memorymesh.core.models import (
    ChunkTierRecord,
    Entity,
    EpisodicEvent,
    FileRecord,
    MemoryTier,
)
from memorymesh.storage.entity_repository import EntityRepository
from memorymesh.storage.event_repository import EventRepository
from memorymesh.storage.file_repository import FileRepository
from memorymesh.storage.suppression_repository import SuppressionRepository
from memorymesh.storage.tier_repository import TierRepository

#
# Version history (stored in PRAGMA user_version):
#
#   1 - initial schema: files, sources, index_state
#   2 - Wave 3 memory primitives: chunk_tiers, episodic_events, entities,
#         entity_mentions
#   3 - suppression list: forgotten_chunks table
#
# Migration policy:
#   - fresh DB (user_version == 0): run full _DDL then set to SCHEMA_VERSION.
#   - existing DB (0 < user_version < SCHEMA_VERSION): apply delta migrations
#     from _MIGRATIONS in order.
#   - up-to-date DB (user_version == SCHEMA_VERSION): no-op.
#   - future DB (user_version > SCHEMA_VERSION): log a warning and proceed.
#
SCHEMA_VERSION: int = 3

# Delta migrations - each entry is (target_version, sql_to_apply).  SQL uses
# CREATE TABLE IF NOT EXISTS / ALTER TABLE so it is idempotent.
_MIGRATIONS: list[tuple[int, str]] = [
    (
        2,
        """
CREATE TABLE IF NOT EXISTS chunk_tiers (
    chunk_id       TEXT PRIMARY KEY,
    tier           TEXT NOT NULL DEFAULT 'warm',
    last_accessed  REAL NOT NULL,
    access_count   INTEGER NOT NULL DEFAULT 0,
    pinned         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunk_tiers_tier          ON chunk_tiers(tier);
CREATE INDEX IF NOT EXISTS idx_chunk_tiers_last_accessed ON chunk_tiers(last_accessed);
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id    TEXT PRIMARY KEY,
    timestamp   REAL NOT NULL,
    event_type  TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    chunk_ids   TEXT NOT NULL DEFAULT '[]',
    client_id   TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_episodic_ts     ON episodic_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_type   ON episodic_events(event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_client ON episodic_events(client_id);
CREATE TABLE IF NOT EXISTS entities (
    name          TEXT NOT NULL,
    entity_type   TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    PRIMARY KEY (name, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_name  TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    PRIMARY KEY (entity_name, entity_type, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk ON entity_mentions(chunk_id);
""",
    ),
    (
        3,
        """
CREATE TABLE IF NOT EXISTS forgotten_chunks (
    chunk_id  TEXT PRIMARY KEY
);
""",
    ),
]

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS files (
    path                TEXT PRIMARY KEY,
    source_name         TEXT NOT NULL,
    sha256              TEXT NOT NULL,
    mtime               REAL NOT NULL,
    size_bytes          INTEGER NOT NULL,
    file_type           TEXT NOT NULL,
    n_chunks            INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL,
    error_message       TEXT,
    indexed_at          REAL NOT NULL,
    embedding_model_id  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_files_source ON files(source_name);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_mtime  ON files(mtime);

CREATE TABLE IF NOT EXISTS sources (
    name                TEXT PRIMARY KEY,
    path                TEXT NOT NULL,
    recursive           INTEGER NOT NULL DEFAULT 1,
    last_full_scan_at   REAL
);

CREATE TABLE IF NOT EXISTS index_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- Wave 3: memory primitives

CREATE TABLE IF NOT EXISTS chunk_tiers (
    chunk_id       TEXT PRIMARY KEY,
    tier           TEXT NOT NULL DEFAULT 'warm',
    last_accessed  REAL NOT NULL,
    access_count   INTEGER NOT NULL DEFAULT 0,
    pinned         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunk_tiers_tier          ON chunk_tiers(tier);
CREATE INDEX IF NOT EXISTS idx_chunk_tiers_last_accessed ON chunk_tiers(last_accessed);

CREATE TABLE IF NOT EXISTS episodic_events (
    event_id    TEXT PRIMARY KEY,
    timestamp   REAL NOT NULL,
    event_type  TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    chunk_ids   TEXT NOT NULL DEFAULT '[]',
    client_id   TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodic_ts     ON episodic_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_type   ON episodic_events(event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_client ON episodic_events(client_id);

CREATE TABLE IF NOT EXISTS entities (
    name          TEXT NOT NULL,
    entity_type   TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    PRIMARY KEY (name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_name  TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    PRIMARY KEY (entity_name, entity_type, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk ON entity_mentions(chunk_id);

CREATE TABLE IF NOT EXISTS forgotten_chunks (
    chunk_id  TEXT PRIMARY KEY
);
"""


class MetadataStore:
    """Facade over all MemoryMesh SQLite repository classes.

    :class:`MetadataStore` manages the SQLite connection and delegates every
    table operation to the appropriate repository:

    * :attr:`_files` - :class:`~memorymesh.storage.file_repository.FileRepository`
    * :attr:`_tiers` - :class:`~memorymesh.storage.tier_repository.TierRepository`
    * :attr:`_events` - :class:`~memorymesh.storage.event_repository.EventRepository`
    * :attr:`_entities` - :class:`~memorymesh.storage.entity_repository.EntityRepository`
    * :attr:`_suppression` - :class:`~memorymesh.storage.suppression_repository\
      .SuppressionRepository`

    This class keeps its public API identical to the monolithic implementation it
    replaced, so all callers continue to work without modification.

    Args:
        db_path: Absolute path to the ``.sqlite3`` file.  The parent directory
            must exist (or be created by the caller) before construction.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

        # Repositories share the same connection via self._connection
        self._files = FileRepository(self._connection)
        self._tiers = TierRepository(self._connection)
        self._events = EventRepository(self._connection)
        self._entities = EntityRepository(self._connection)
        self._suppression = SuppressionRepository(self._connection)

    def _connection(self) -> sqlite3.Connection:
        """Return (and lazily open) the SQLite connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    self._db_path.parent.chmod(0o700)
                except OSError as exc:
                    logger.warning(
                        f"MetadataStore: could not set permissions on {self._db_path.parent}: {exc}"
                    )
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self) -> None:
        """Create all tables, run pending migrations, and stamp PRAGMA user_version.

        On a **fresh** database (``user_version == 0``) the full DDL is executed
        and ``user_version`` is set to :data:`SCHEMA_VERSION`.

        On an **existing** database the function applies only the delta
        migrations needed to reach :data:`SCHEMA_VERSION`, then updates
        ``user_version``.  All migration SQL uses ``CREATE TABLE IF NOT EXISTS``
        so it is safe to re-run.

        On a **future** database (``user_version > SCHEMA_VERSION``) a warning is
        logged and the function returns without touching the schema.
        """
        conn = self._connection()
        current_version: int = conn.execute("PRAGMA user_version").fetchone()[0]

        if current_version == SCHEMA_VERSION:
            logger.debug(
                f"Metadata schema already at version {SCHEMA_VERSION} - no migration needed"
            )
            return

        if current_version > SCHEMA_VERSION:
            logger.warning(
                f"DB schema version {current_version} is newer than this build "
                f"({SCHEMA_VERSION}). Proceeding read-only migration guard."
            )
            return

        if current_version == 0:
            # Fresh database - run full DDL and stamp version
            conn.executescript(_DDL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            logger.debug(f"Metadata schema initialised at version {SCHEMA_VERSION}")
            return

        # Incremental migration: apply each step from current_version + 1 to SCHEMA_VERSION
        for target_ver, sql in _MIGRATIONS:
            if target_ver <= current_version:
                continue
            if target_ver > SCHEMA_VERSION:
                break
            logger.info(f"MetadataStore: migrating schema v{target_ver - 1} -> v{target_ver}")
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {target_ver}")
            conn.commit()

        logger.debug(f"Metadata schema migrated to version {SCHEMA_VERSION}")

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_state(self, key: str) -> str | None:
        """Return the value for *key* in ``index_state``, or ``None``."""
        return self._files.get_state(key)

    def set_state(self, key: str, value: str) -> None:
        """Upsert *key* -> *value* in ``index_state``."""
        self._files.set_state(key, value)

    def mark_startup(self) -> None:
        """Record that the daemon has started (not yet cleanly shut down).

        Increments the ``epoch`` counter and sets ``last_clean_shutdown`` to
        ``"false"``.  Called at daemon boot, *before* serving any requests.
        """
        self._files.mark_startup()

    def mark_clean_shutdown(self) -> None:
        """Record that the daemon stopped cleanly.

        Must be called during graceful shutdown *after* all in-flight indexing
        operations have completed and the BM25 pickle has been flushed.
        """
        self._files.mark_clean_shutdown()

    def is_clean_state(self) -> bool:
        """Return ``True`` if the previous run shut down cleanly.

        A missing ``last_clean_shutdown`` key means this is a fresh install -
        treated as clean because there is no prior state to reconcile.
        """
        return self._files.is_clean_state()

    def upsert_file(self, record: FileRecord) -> None:
        """Insert or update a file record.

        Args:
            record: The :class:`~memorymesh.core.models.FileRecord` to persist.
        """
        self._files.upsert_file(record)

    def get_file(self, path: str) -> FileRecord | None:
        """Return the record for *path*, or ``None`` if not found.

        Args:
            path: Absolute path string (primary key).
        """
        return self._files.get_file(path)

    def delete_file(self, path: str) -> None:
        """Hard-delete the row for *path* from the database.

        Prefer :meth:`mark_deleted` for soft-delete semantics (keeps history).

        Args:
            path: Absolute path string.
        """
        self._files.delete_file(path)

    def mark_deleted(self, path: str) -> None:
        """Set ``status = 'deleted'`` for *path* without removing the row.

        Args:
            path: Absolute path string.
        """
        self._files.mark_deleted(path)

    def mark_pending_reindex(self, path: str) -> None:
        """Set ``status = 'pending_reindex'`` - used during reconciliation.

        Args:
            path: Absolute path string.
        """
        self._files.mark_pending_reindex(path)

    def list_files(
        self,
        source_name: str | None = None,
        status: str | None = None,
    ) -> list[FileRecord]:
        """Return all file records matching the optional filters.

        Args:
            source_name: Restrict to a specific source.  ``None`` = all sources.
            status: Restrict to a specific status value.  ``None`` = all statuses.
        """
        return self._files.list_files(source_name=source_name, status=status)

    def get_stats(self, source_name: str | None = None) -> dict[str, int]:
        """Return aggregate counts by status for a source (or all sources).

        Args:
            source_name: Restrict stats to one source.  ``None`` = all.

        Returns:
            Dict mapping status string -> file count.
        """
        return self._files.get_stats(source_name=source_name)

    def upsert_source(
        self,
        name: str,
        path: str,
        recursive: bool,
        last_full_scan_at: float | None = None,
    ) -> None:
        """Insert or update a source record.

        Args:
            name: Source identifier (primary key).
            path: Absolute path string of the monitored directory.
            recursive: Whether the source is scanned recursively.
            last_full_scan_at: Unix timestamp of the most recent full scan.
        """
        self._files.upsert_source(name, path, recursive, last_full_scan_at)

    def update_source_scan_time(self, name: str) -> None:
        """Set ``last_full_scan_at`` to the current time for *name*.

        Args:
            name: Source identifier.
        """
        self._files.update_source_scan_time(name)

    def list_sources(self) -> list[dict[str, object]]:
        """Return all source rows as plain dicts."""
        return self._files.list_sources()

    def set_chunk_tier(self, record: ChunkTierRecord) -> None:
        """Upsert the tier record for a chunk.

        Args:
            record: The :class:`~memorymesh.core.models.ChunkTierRecord` to persist.
        """
        self._tiers.set_chunk_tier(record)

    def get_chunk_tier(self, chunk_id: str) -> ChunkTierRecord | None:
        """Return the tier record for *chunk_id*, or ``None`` if not tracked.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        return self._tiers.get_chunk_tier(chunk_id)

    def record_chunk_access(self, chunk_id: str) -> None:
        """Increment access_count and refresh last_accessed for *chunk_id*.

        Creates the row with ``tier='warm'`` if it does not exist yet.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        self._tiers.record_chunk_access(chunk_id)

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
        return self._tiers.list_chunks_by_tier(tier, limit=limit)

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
        return self._tiers.promote_chunks_to_tier(chunk_ids, tier)

    def upsert_episodic_event(self, event: EpisodicEvent) -> None:
        """Insert or replace an episodic event.

        Args:
            event: The :class:`~memorymesh.core.models.EpisodicEvent` to persist.
                If ``event.event_id`` is empty a UUID must be assigned by the caller
                before this call.
        """
        self._events.upsert_episodic_event(event)

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
        return self._events.list_episodic_events(
            since=since,
            until=until,
            event_type=event_type,
            client_id=client_id,
            limit=limit,
        )

    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity record.

        On conflict (same ``name`` + ``entity_type``), increments
        ``mention_count``, updates ``last_seen``, and preserves ``first_seen``.

        Args:
            entity: The :class:`~memorymesh.core.models.Entity` to persist.
        """
        self._entities.upsert_entity(entity)

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
        return self._entities.list_entities(
            entity_type=entity_type,
            min_mentions=min_mentions,
            limit=limit,
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
        self._entities.add_entity_mention(entity_name, entity_type, chunk_id)

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
        return self._entities.get_entity_chunks(entity_name, entity_type)

    def get_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> Entity | None:
        """Return the :class:`~memorymesh.core.models.Entity` for *name*.

        Args:
            name: Canonical entity name (case-sensitive after normalisation).
            entity_type: Optional type filter.  When provided, only returns the
                entity if its type matches exactly.

        Returns:
            The entity record, or ``None`` if not found.
        """
        return self._entities.get_entity(name, entity_type)

    def get_entity_mentions(self, name: str) -> list[str]:
        """Return all chunk IDs that mention the entity *name* (any type).

        Args:
            name: Canonical entity name.

        Returns:
            List of ``<path>:<chunk_index>`` strings.
        """
        return self._entities.get_entity_mentions(name)

    def forget_chunk(self, chunk_id: str) -> None:
        """Suppress a chunk from future search results.

        Adds the *chunk_id* to a ``forgotten_chunks`` suppression list.  The
        search engine checks this list and omits matching results.

        Args:
            chunk_id: ``<path>:<chunk_index>`` identifier of the chunk to forget.
        """
        self._suppression.forget_chunk(chunk_id)

    def list_forgotten(self) -> list[str]:
        """Return all chunk IDs currently in the suppression list.

        Returns:
            List of ``<path>:<chunk_index>`` strings that are hidden from search.
            Empty list when the suppression table has no entries.
        """
        return self._suppression.list_forgotten()

    def export_encrypted(
        self,
        dest: Path,
        encryption: object,
    ) -> Path:
        """Export an encrypted copy of the SQLite database.

        Uses the SQLite online backup API to create a consistent snapshot, then
        Fernet-encrypts the raw bytes and writes them to *dest*.

        Args:
            dest: Output file path for the encrypted blob.
            encryption: An :class:`~memorymesh.storage.encryption.EncryptionManager`
                instance.  Must have an ``encrypt(bytes) -> bytes`` method.

        Returns:
            The resolved *dest* path.
        """
        import io

        buf = io.BytesIO()
        # SQLite online backup into an in-memory database, then dump bytes.
        mem_conn = sqlite3.connect(":memory:")
        src_conn = self._connection()
        src_conn.backup(mem_conn)
        for line in mem_conn.iterdump():
            buf.write((line + "\n").encode())
        mem_conn.close()

        dest = dest.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(encryption.encrypt(buf.getvalue()))  # type: ignore[attr-defined,union-attr]
        logger.info(f"MetadataStore: exported encrypted backup to {dest}")
        return dest
