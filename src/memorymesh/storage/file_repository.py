"""Repository for file, source, index-state, and lifecycle records.

:class:`FileRepository` owns the ``files``, ``sources``, and ``index_state``
tables.  All methods that were previously on :class:`MetadataStore` for these
tables live here; :class:`MetadataStore` delegates to this class.
"""

from __future__ import annotations

import time

from loguru import logger

from memorymesh.core.models import FileRecord
from memorymesh.storage.db import ConnectionFactory

_INSERT_FILE = """
INSERT INTO files
    (path, source_name, sha256, mtime, size_bytes, file_type,
     n_chunks, status, error_message, indexed_at, embedding_model_id)
VALUES
    (:path, :source_name, :sha256, :mtime, :size_bytes, :file_type,
     :n_chunks, :status, :error_message, :indexed_at, :embedding_model_id)
ON CONFLICT(path) DO UPDATE SET
    source_name        = excluded.source_name,
    sha256             = excluded.sha256,
    mtime              = excluded.mtime,
    size_bytes         = excluded.size_bytes,
    file_type          = excluded.file_type,
    n_chunks           = excluded.n_chunks,
    status             = excluded.status,
    error_message      = excluded.error_message,
    indexed_at         = excluded.indexed_at,
    embedding_model_id = excluded.embedding_model_id;
"""


class FileRepository:
    """Manages the ``files``, ``sources``, and ``index_state`` tables.

    Args:
        connection_factory: Zero-argument callable that returns the shared
            :class:`sqlite3.Connection`.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        self._conn = connection_factory

    def get_state(self, key: str) -> str | None:
        """Return the value for *key* in ``index_state``, or ``None``."""
        row = self._conn().execute("SELECT value FROM index_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Upsert *key* -> *value* in ``index_state``."""
        self._conn().execute(
            "INSERT INTO index_state(key, value) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn().commit()

    def mark_startup(self) -> None:
        """Record that the daemon has started (not yet cleanly shut down).

        Increments the ``epoch`` counter and sets ``last_clean_shutdown`` to
        ``"false"``.
        """
        epoch = int(self.get_state("epoch") or "0") + 1
        conn = self._conn()
        conn.execute(
            "INSERT INTO index_state(key, value) VALUES('epoch', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(epoch),),
        )
        conn.execute(
            "INSERT INTO index_state(key, value) VALUES('last_clean_shutdown', 'false')"
            " ON CONFLICT(key) DO UPDATE SET value = 'false'"
        )
        conn.commit()
        logger.debug(f"Daemon startup recorded (epoch={epoch})")

    def mark_clean_shutdown(self) -> None:
        """Record that the daemon stopped cleanly."""
        self.set_state("last_clean_shutdown", "true")
        logger.debug("Clean shutdown recorded in index_state")

    def is_clean_state(self) -> bool:
        """Return ``True`` if the previous run shut down cleanly."""
        value = self.get_state("last_clean_shutdown")
        return value is None or value == "true"

    def epoch(self) -> int:
        """Return the current epoch counter (number of daemon startups)."""
        return int(self.get_state("epoch") or "0")

    def upsert_file(self, record: FileRecord) -> None:
        """Insert or update a file record.

        Args:
            record: The :class:`~memorymesh.core.models.FileRecord` to persist.
        """
        self._conn().execute(_INSERT_FILE, record.model_dump())
        self._conn().commit()

    def get_file(self, path: str) -> FileRecord | None:
        """Return the record for *path*, or ``None`` if not found.

        Args:
            path: Absolute path string (primary key).
        """
        row = self._conn().execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return FileRecord(**dict(row)) if row else None

    def delete_file(self, path: str) -> None:
        """Hard-delete the row for *path* from the database.

        Args:
            path: Absolute path string.
        """
        self._conn().execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn().commit()

    def mark_deleted(self, path: str) -> None:
        """Set ``status = 'deleted'`` for *path* without removing the row.

        Args:
            path: Absolute path string.
        """
        self._conn().execute("UPDATE files SET status = 'deleted' WHERE path = ?", (path,))
        self._conn().commit()

    def mark_pending_reindex(self, path: str) -> None:
        """Set ``status = 'pending_reindex'`` for *path*.

        Args:
            path: Absolute path string.
        """
        self._conn().execute("UPDATE files SET status = 'pending_reindex' WHERE path = ?", (path,))
        self._conn().commit()

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
        clauses: list[str] = []
        params: list[str] = []
        if source_name is not None:
            clauses.append("source_name = ?")
            params.append(source_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn().execute(f"SELECT * FROM files {where}", params).fetchall()
        return [FileRecord(**dict(r)) for r in rows]

    def get_stats(self, source_name: str | None = None) -> dict[str, int]:
        """Return aggregate counts by status for a source (or all sources).

        Args:
            source_name: Restrict stats to one source.  ``None`` = all.

        Returns:
            Dict mapping status string -> file count.
        """
        if source_name is not None:
            rows = (
                self._conn()
                .execute(
                    "SELECT status, COUNT(*) as cnt FROM files"
                    " WHERE source_name = ? GROUP BY status",
                    (source_name,),
                )
                .fetchall()
            )
        else:
            rows = (
                self._conn()
                .execute("SELECT status, COUNT(*) as cnt FROM files GROUP BY status")
                .fetchall()
            )
        return {r["status"]: r["cnt"] for r in rows}

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
        self._conn().execute(
            "INSERT INTO sources(name, path, recursive, last_full_scan_at)"
            " VALUES(?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET"
            "   path = excluded.path,"
            "   recursive = excluded.recursive,"
            "   last_full_scan_at = COALESCE(excluded.last_full_scan_at, last_full_scan_at)",
            (name, path, int(recursive), last_full_scan_at),
        )
        self._conn().commit()

    def update_source_scan_time(self, name: str) -> None:
        """Set ``last_full_scan_at`` to the current time for *name*.

        Args:
            name: Source identifier.
        """
        self._conn().execute(
            "UPDATE sources SET last_full_scan_at = ? WHERE name = ?",
            (time.time(), name),
        )
        self._conn().commit()

    def list_sources(self) -> list[dict[str, object]]:
        """Return all source rows as plain dicts."""
        rows = self._conn().execute("SELECT * FROM sources").fetchall()
        return [dict(r) for r in rows]
