"""Telegram Desktop export connector for MemoryMesh.

Parses the JSON export produced by Telegram Desktop (Settings -> Export
Telegram Data -> Machine-readable JSON format) and yields conversation
chunks as :class:`~memorymesh.core.models.ParsedDocument` objects.

Features
--------
* **Auto-locate** - if ``export_path`` is a directory, ``result.json`` is
  found automatically inside it.
* **Chat-type filter** - only process the chat types listed in
  ``chat_types`` (default: personal, group, supergroup chats).
* **Polymorphic text** - Telegram's ``text`` field may be a plain string or
  a list of formatting entities; both are flattened to plain text.
* **Service skip** - system events (pinned messages, joins, etc.) with
  ``"type": "service"`` are silently dropped.
* **Chunking** - messages are grouped into ~``chunk_size``-message documents.
* **Privacy** - message content is never logged at INFO level.
* **Stdlib only** - no third-party dependencies.

Usage
-----
::

    connector = TelegramConnector(TelegramConfig(
        export_path=Path("~/Downloads/Telegram Desktop/result.json"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class TelegramConfig(BaseModel):
    """Configuration for a Telegram export source.

    Args:
        export_path: Path to ``result.json`` *or* the directory containing it.
        chat_types: Only process chats whose ``type`` field is in this list.
            An empty list means all chat types are accepted.
        chunk_size: Number of messages grouped into one ParsedDocument.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    chat_types: list[str] = ["personal_chat", "private_group", "private_supergroup"]
    chunk_size: int = 50
    source_name: str = "telegram"


def _flatten_text(text_field: Any) -> str:
    """Convert Telegram's polymorphic ``text`` field to a plain string.

    Telegram's ``text`` is either a plain string *or* a list whose elements
    are either plain strings or formatting-entity dicts (each with a ``text``
    key).

    Args:
        text_field: The value of the ``text`` key in a message object.

    Returns:
        Plain concatenated string.
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: list[str] = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _safe_name(text: str) -> str:
    """Sanitise *text* for use in a synthetic URL path segment.

    Args:
        text: Raw string (chat name, etc.).

    Returns:
        Alphanumeric + safe punctuation only, max 60 characters.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)[:60]


class TelegramConnector:
    """Parses a Telegram Desktop export JSON and yields conversation chunks.

    Args:
        config: Export path and filtering settings.
    """

    def __init__(self, config: TelegramConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse ``result.json`` and yield message chunks as ParsedDocuments.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            ~``chunk_size`` messages, with ``file_type=".telegram"`` and
            metadata containing ``chat_name``, ``chat_type``,
            ``message_count``, and ``participants``.
        """
        p = self._cfg.export_path
        if p.is_dir():
            json_file = p / "result.json"
        elif p.is_file():
            json_file = p
        else:
            logger.warning(f"TelegramConnector: path not found: {p}")
            return

        if not json_file.exists():
            logger.warning(f"TelegramConnector: result.json not found at {json_file}")
            return

        try:
            data = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"TelegramConnector: cannot parse {json_file}: {exc}")
            return

        chats = data.get("chats", {}).get("list", [])
        total = 0

        for chat in chats:
            for doc in self._process_chat(chat):
                yield doc
                total += 1

        logger.info(f"TelegramConnector: yielded {total} chunk(s)")

    def _process_chat(self, chat: dict[str, Any]) -> Iterator[ParsedDocument]:
        """Process one chat entry from the export JSON.

        Args:
            chat: A chat dict from ``chats.list``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per message chunk.
        """
        chat_type = chat.get("type", "")
        if self._cfg.chat_types and chat_type not in self._cfg.chat_types:
            return

        chat_name = chat.get("name", "") or "unknown"
        raw_msgs = chat.get("messages", [])

        # Keep only real messages with non-empty text
        messages = [
            m
            for m in raw_msgs
            if m.get("type") == "message" and _flatten_text(m.get("text", "")).strip()
        ]

        if not messages:
            return

        logger.info(f"TelegramConnector: {len(messages)} message(s) in {chat_name!r}")

        safe = _safe_name(chat_name)
        source = self._cfg.source_name
        chunk_size = self._cfg.chunk_size

        for chunk_idx, offset in enumerate(range(0, len(messages), chunk_size)):
            chunk = messages[offset : offset + chunk_size]
            participants = sorted({m.get("from", "") or "" for m in chunk if m.get("from")})
            lines = [
                f"{m.get('date', '')} {m.get('from', '')}: {_flatten_text(m.get('text', ''))}"
                for m in chunk
            ]
            block = "\n".join(lines)

            yield ParsedDocument(
                path=Path(f"telegram://{safe}/{chunk_idx}.telegram"),
                text=block,
                file_type=".telegram",
                encoding="utf-8",
                metadata={
                    "chat_name": chat_name,
                    "chat_type": chat_type,
                    "message_count": len(chunk),
                    "participants": participants,
                    "source": source,
                },
            )
