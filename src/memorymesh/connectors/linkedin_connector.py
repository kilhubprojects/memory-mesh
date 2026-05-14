"""LinkedIn data export connector for MemoryMesh.

Parses LinkedIn data exports (Settings -> Data Privacy -> Get a copy of
your data) and yields :class:`~memorymesh.core.models.ParsedDocument`
objects for connections, messages, and posts.

Files parsed
------------
* ``Connections.csv`` - full connections list.
* ``messages.csv`` - conversations grouped by ``ConversationID``.
* ``shares.csv`` / ``Posts.csv`` - individual posts.

All files are optional; missing files are silently skipped.

Usage
-----
::

    connector = LinkedInConnector(LinkedInConfig(
        export_path=Path("~/Downloads/LinkedIn_export"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class LinkedInConfig(BaseModel):
    """Configuration for a LinkedIn data export source.

    Args:
        export_path: Directory containing LinkedIn export CSV files.
        include_messages: Parse ``messages.csv`` when ``True``.
        include_posts: Parse ``shares.csv`` / ``Posts.csv`` when ``True``.
        include_connections: Parse ``Connections.csv`` when ``True``.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    include_messages: bool = True
    include_posts: bool = True
    include_connections: bool = True
    source_name: str = "linkedin"


class LinkedInConnector:
    """Parses LinkedIn CSV exports and yields ParsedDocuments.

    Args:
        config: Export path, feature toggles, and source settings.
    """

    def __init__(self, config: LinkedInConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse all configured LinkedIn export files and yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            connections list, one per conversation, or one per post, with
            ``file_type=".linkedin"`` and metadata indicating the record
            type.
        """
        base = self._cfg.export_path.expanduser()
        total = 0

        if self._cfg.include_connections:
            for doc in self._parse_connections(base):
                yield doc
                total += 1

        if self._cfg.include_messages:
            for doc in self._parse_messages(base):
                yield doc
                total += 1

        if self._cfg.include_posts:
            for doc in self._parse_posts(base):
                yield doc
                total += 1

        logger.info(f"LinkedInConnector: yielded {total} document(s)")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        """Read a LinkedIn CSV file, tolerating preamble comment lines.

        LinkedIn CSV files sometimes prepend ``Notes:`` comment lines
        before the header row.  This method skips those lines.

        Args:
            path: Path to the CSV file.

        Returns:
            List of row dicts, or an empty list on error or absence.
        """
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning(f"LinkedInConnector: cannot read {path}: {exc}")
            return []
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "," in stripped and not stripped.startswith("Notes:"):
                start = i
                break
        reader = csv.DictReader(lines[start:])
        return list(reader)

    def _parse_connections(self, base: Path) -> Iterator[ParsedDocument]:
        """Yield one document containing the full connections list.

        Args:
            base: Root export directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        rows = self._read_csv(base / "Connections.csv")
        if not rows:
            return

        lines: list[str] = []
        for row in rows:
            first = row.get("First Name", "").strip()
            last = row.get("Last Name", "").strip()
            name = f"{first} {last}".strip() or "Unknown"
            position = row.get("Position", "").strip()
            company = row.get("Company", "").strip()
            connected = row.get("Connected On", "").strip()
            lines.append(f"{name} - {position} at {company} (connected {connected})")

        source = self._cfg.source_name
        yield ParsedDocument(
            path=Path("linkedin://connections.linkedin"),
            text=(f"LinkedIn Connections ({len(lines)} total)\n\n" + "\n".join(sorted(lines))),
            file_type=".linkedin",
            encoding="utf-8",
            metadata={
                "type": "connections",
                "connection_count": len(lines),
                "source": source,
            },
        )

    def _parse_messages(self, base: Path) -> Iterator[ParsedDocument]:
        """Yield one document per LinkedIn conversation.

        Args:
            base: Root export directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per
            ``ConversationID``.
        """
        rows = self._read_csv(base / "messages.csv")
        if not rows:
            return

        # conv_id -> list of (sent_at, sender, body)
        by_conv: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for row in rows:
            conv_id = row.get("ConversationID", "").strip()
            sender = row.get("From", "").strip()
            sent_at = row.get("SentAt", "").strip()
            body = row.get("MessageBody", "").strip()
            if conv_id and body:
                by_conv[conv_id].append((sent_at, sender, body))

        source = self._cfg.source_name
        for conv_id, msgs in by_conv.items():
            msgs_sorted = sorted(msgs, key=lambda m: m[0])
            lines = [f"{t} {s}: {b}" for t, s, b in msgs_sorted]
            yield ParsedDocument(
                path=Path(f"linkedin://messages/{conv_id}.linkedin"),
                text="\n".join(lines),
                file_type=".linkedin",
                encoding="utf-8",
                metadata={
                    "type": "message",
                    "conversation_id": conv_id,
                    "message_count": len(lines),
                    "source": source,
                },
            )

    def _parse_posts(self, base: Path) -> Iterator[ParsedDocument]:
        """Yield one document per post (shares.csv or Posts.csv).

        Args:
            base: Root export directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per post.
        """
        posts_file = base / "shares.csv"
        if not posts_file.exists():
            posts_file = base / "Posts.csv"
        rows = self._read_csv(posts_file)

        source = self._cfg.source_name
        for i, row in enumerate(rows):
            commentary = (row.get("ShareCommentary", "") or row.get("PostCommentary", "")).strip()
            posted_at = (row.get("PostedAt", "") or row.get("Date", "")).strip()
            if not commentary:
                continue
            yield ParsedDocument(
                path=Path(f"linkedin://posts/{i}.linkedin"),
                text=f"Posted: {posted_at}\n\n{commentary}",
                file_type=".linkedin",
                encoding="utf-8",
                metadata={
                    "type": "post",
                    "posted_at": posted_at,
                    "source": source,
                },
            )
