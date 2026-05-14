"""Client revocation list for MemoryMesh (Wave 4).

Maintains a SQLite-backed deny-list of revoked client IDs.  Revoked clients
are rejected at the identity-resolution stage regardless of their ACL config.

The revocation list is intentionally simple: only client IDs can be revoked,
not individual tokens or sessions.  For the local-daemon use case this is
sufficient — if you want to block a specific agent, revoke its client_id and
update the config to remove it.

All checks are no-ops when ``auth.enabled: false``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from loguru import logger


class RevocationList:
    """SQLite-backed revocation list for MemoryMesh clients.

    Args:
        db_path: Path to the SQLite database (shared with MetadataStore).
            The revocation table is created automatically.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS revoked_clients (
        client_id   TEXT PRIMARY KEY,
        revoked_at  REAL NOT NULL,
        reason      TEXT NOT NULL DEFAULT ''
    );
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._cache: set[str] = set()
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 30.0  # Re-read from DB every 30 s.

    def _connection(self) -> sqlite3.Connection:
        """Return (and lazily open) the SQLite connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self._DDL)
            self._conn.commit()
        return self._conn

    def _refresh_cache(self) -> None:
        """Reload the revocation list from SQLite if the TTL has expired."""
        now = time.monotonic()
        if now - self._cache_ts < self._cache_ttl:
            return
        rows = self._connection().execute("SELECT client_id FROM revoked_clients").fetchall()
        self._cache = {r["client_id"] for r in rows}
        self._cache_ts = now

    def is_revoked(self, client_id: str) -> bool:
        """Return ``True`` if *client_id* has been revoked.

        Args:
            client_id: Client identifier to check.
        """
        self._refresh_cache()
        return client_id in self._cache

    def revoke(self, client_id: str, reason: str = "") -> None:
        """Add *client_id* to the revocation list.

        Args:
            client_id: Client identifier to revoke.
            reason: Human-readable reason stored in the database.
        """
        self._connection().execute(
            "INSERT INTO revoked_clients(client_id, revoked_at, reason)"
            " VALUES(?, ?, ?)"
            " ON CONFLICT(client_id) DO UPDATE SET"
            "   revoked_at = excluded.revoked_at,"
            "   reason = excluded.reason",
            (client_id, time.time(), reason),
        )
        self._connection().commit()
        self._cache.add(client_id)
        logger.warning(f"RevocationList: revoked client_id={client_id!r} reason={reason!r}")

    def unrevoke(self, client_id: str) -> None:
        """Remove *client_id* from the revocation list.

        Args:
            client_id: Client identifier to restore.
        """
        self._connection().execute("DELETE FROM revoked_clients WHERE client_id = ?", (client_id,))
        self._connection().commit()
        self._cache.discard(client_id)
        logger.info(f"RevocationList: unrevoked client_id={client_id!r}")

    def list_revoked(self) -> list[dict[str, object]]:
        """Return all revoked client records.

        Returns:
            List of dicts with ``client_id``, ``revoked_at``, and ``reason``.
        """
        rows = (
            self._connection()
            .execute("SELECT * FROM revoked_clients ORDER BY revoked_at DESC")
            .fetchall()
        )
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
