"""Append-only audit log for MCP tool calls.

Records one JSONL line per query.  The query text itself is **never** stored -
only a truncated SHA-256 hash - to preserve user privacy while still enabling
operational monitoring (latency, result counts, tool usage patterns).

File format (one JSON object per line)::

    {"ts": "2026-05-01T12:00:00+00:00", "tool": "search_memory",
     "query_hash": "a3f1b2c4d5e6f789", "n_results": 7,
     "latency_ms": 42.3, "client_id": null}

When an :class:`~memorymesh.storage.encryption.EncryptionManager` is supplied,
each line is Fernet-encrypted and written as a base64 token followed by a
newline.  Decrypt with ``EncryptionManager.decrypt(line.rstrip())``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.storage.encryption import EncryptionManager


class AuditLogger:
    """Appends one JSONL line per MCP tool invocation to the audit log.

    Args:
        audit_log_path: Absolute path to the ``.jsonl`` file.  The parent
            directory is created on first write if it does not exist.
        encryption: Optional :class:`~memorymesh.storage.encryption.EncryptionManager`.
            When provided, each record is Fernet-encrypted before being written.
    """

    def __init__(
        self,
        audit_log_path: Path,
        encryption: EncryptionManager | None = None,
    ) -> None:
        self._path = audit_log_path
        self._encryption = encryption

    def log_query(
        self,
        tool: str,
        query: str,
        n_results: int,
        latency_ms: float,
        client_id: str | None = None,
    ) -> None:
        """Append one audit record for a completed tool call.

        Args:
            tool: MCP tool name (e.g. ``"search_memory"``).
            query: The raw query string - hashed, never stored in cleartext.
            n_results: Number of results returned to the client.
            latency_ms: Total round-trip latency in milliseconds.
            client_id: Optional opaque client identifier from the MCP context.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        record: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool,
            "query_hash": query_hash,
            "n_results": n_results,
            "latency_ms": round(latency_ms, 2),
        }
        if client_id is not None:
            record["client_id"] = client_id

        self._append(record)

    def _append(self, record: dict[str, object]) -> None:
        """Serialise *record* to JSONL (optionally encrypted) and append to the log."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    self._path.parent.chmod(0o700)
                except OSError as exc:
                    logger.warning(
                        f"AuditLogger: could not set permissions on {self._path.parent}: {exc}"
                    )
            plaintext = json.dumps(record, ensure_ascii=False).encode()
            if self._encryption is not None:
                line = self._encryption.encrypt(plaintext).decode() + "\n"
            else:
                line = plaintext.decode() + "\n"
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            # Audit log failure must never crash the daemon.
            logger.warning(f"Failed to write audit log entry: {exc}")
