"""Discord JSON export connector for MemoryMesh.

Parses JSON exports produced by DiscordChatExporter (open-source tool)
and yields one :class:`~memorymesh.core.models.ParsedDocument` per
channel-day pair.

Export format (DiscordChatExporter JSON)::

    {
      "guild": {"name": "Server Name"},
      "channel": {"name": "general", "type": "GuildTextChat"},
      "messages": [
        {
          "id": "...",
          "timestamp": "2024-01-01T10:00:00+00:00",
          "author": {"name": "User#1234", "isBot": false},
          "content": "Hello"
        }
      ]
    }

Features
--------
* **Bot filtering** - messages with ``author.isBot == true`` are skipped.
* **Daily grouping** - messages are grouped by (channel, UTC date).
* **Multi-file** - all ``*.json`` files in ``export_path`` are parsed.

Usage
-----
::

    connector = DiscordConnector(DiscordConfig(
        export_path=Path("~/Discord/exports"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class DiscordConfig(BaseModel):
    """Configuration for a Discord JSON export source.

    Args:
        export_path: Directory containing DiscordChatExporter ``*.json``
            files.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    source_name: str = "discord"


class DiscordConnector:
    """Parses DiscordChatExporter exports and yields per-channel-day docs.

    Args:
        config: Export path and source settings.
    """

    def __init__(self, config: DiscordConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Glob all JSON exports and yield one document per channel-day.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            (channel, date) pair, with ``file_type=".discord"`` and
            metadata containing ``guild``, ``channel``, ``date``, and
            ``message_count``.
        """
        export_path = self._cfg.export_path.expanduser()
        json_files = sorted(export_path.glob("*.json"))

        if not json_files:
            logger.warning(f"DiscordConnector: no JSON files in {export_path}")
            return

        total_docs = 0
        for json_file in json_files:
            for doc in self._process_file(json_file):
                yield doc
                total_docs += 1

        logger.info(f"DiscordConnector: yielded {total_docs} document(s)")

    def _process_file(self, json_file: Path) -> Iterator[ParsedDocument]:
        """Parse one DiscordChatExporter JSON file.

        Args:
            json_file: Path to the export JSON.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per
            channel-day.
        """
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"DiscordConnector: cannot read {json_file}: {exc}")
            return

        guild_name = data.get("guild", {}).get("name", "unknown_guild")
        channel_name = data.get("channel", {}).get("name", "unknown_channel")
        messages = data.get("messages", [])

        # date -> list of (time_str, author, content)
        by_date: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for msg in messages:
            author_obj = msg.get("author", {})
            if author_obj.get("isBot"):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            ts = msg.get("timestamp", "")
            # "2024-01-01T10:00:00+00:00" -> date "2024-01-01"
            date_str = ts[:10] if len(ts) >= 10 else "unknown"
            time_str = ts[11:16] if len(ts) >= 16 else "00:00"
            author = author_obj.get("name", "unknown")
            by_date[date_str].append((time_str, author, content))

        safe_guild = guild_name.replace("/", "_").replace(" ", "_")
        safe_channel = channel_name.replace("/", "_").replace(" ", "_")
        source = self._cfg.source_name

        for date_str in sorted(by_date):
            msgs = by_date[date_str]
            lines = [f"{t} {a}: {c}" for t, a, c in msgs]
            yield ParsedDocument(
                path=Path(f"discord://{safe_guild}/{safe_channel}/{date_str}.discord"),
                text="\n".join(lines),
                file_type=".discord",
                encoding="utf-8",
                metadata={
                    "guild": guild_name,
                    "channel": channel_name,
                    "date": date_str,
                    "message_count": len(lines),
                    "source": source,
                },
            )
