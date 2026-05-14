"""Parser for AI conversation JSON exports.

Supports two formats and auto-detects the source application:

**Claude.ai export:**
An array of conversation objects with::

    [{"uuid": "…", "name": "…", "chat_messages": [{"role": "…", "content": "…", …}]}]

**ChatGPT export:**
A single conversation object with a ``mapping`` dict::

    {"title": "…", "mapping": {"<id>": {"message": {"role": "…", "content": {"parts": ["…"]}}}}}

Each conversation turn becomes its own :class:`~memorymesh.core.models.ParsedDocument`.
Turns longer than ``_LONG_TURN_THRESHOLD`` characters are split by the
:class:`~memorymesh.chunking.recursive_chunker.RecursiveChunker` at index time
because the parser itself returns the full turn text — chunking is a downstream
concern.

The parser is registered for ``.json`` files when a source is configured with
``source.type: conversations``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

# If a single turn is longer than this many characters we warn the user so
# they can tune chunk_size if needed.  We don't split here — that is done
# downstream by the chunker.
_LONG_TURN_THRESHOLD = 4_096


def _detect_source(data: Any) -> str | None:
    """Return ``"claude"`` or ``"chatgpt"`` or ``None`` if format unrecognised.

    Args:
        data: Parsed JSON value (list or dict).

    Returns:
        Format identifier string or ``None``.
    """
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "chat_messages" in first:
            return "claude"
    if isinstance(data, dict) and "mapping" in data:
        return "chatgpt"
    return None


def _parse_claude(data: list[dict[str, Any]], path: Path) -> list[ParsedDocument]:
    """Parse a Claude.ai JSON export.

    Args:
        data: Parsed JSON — an array of conversation objects.
        path: Source file path (used for metadata).

    Returns:
        One :class:`~memorymesh.core.models.ParsedDocument` per conversation turn.
    """
    docs: list[ParsedDocument] = []

    for conv in data:
        if not isinstance(conv, dict):
            continue

        session_id: str = str(conv.get("uuid", ""))
        title: str = str(conv.get("name", ""))
        messages: list[Any] = conv.get("chat_messages", [])

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            role: str = str(msg.get("role", "unknown"))
            content: Any = msg.get("content", "")

            # Content can be a plain string or a list of content blocks.
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                text = str(content).strip()

            if not text:
                continue

            if len(text) > _LONG_TURN_THRESHOLD:
                logger.debug(
                    f"ConversationParser: long turn ({len(text)} chars) in "
                    f"{path.name!r} session={session_id!r} index={i}"
                )

            timestamp: str = str(msg.get("created_at", msg.get("updated_at", "")))

            meta: dict[str, object] = {
                "role": role,
                "timestamp": timestamp,
                "session_id": session_id,
                "conversation_title": title,
                "source_app": "claude",
                "turn_index": i,
            }

            docs.append(
                ParsedDocument(
                    path=path,
                    text=text,
                    file_type=".json",
                    encoding="utf-8",
                    metadata=meta,
                )
            )

    return docs


def _parse_chatgpt(data: dict[str, Any], path: Path) -> list[ParsedDocument]:
    """Parse a ChatGPT JSON export.

    Args:
        data: Parsed JSON — a single conversation object.
        path: Source file path (used for metadata).

    Returns:
        One :class:`~memorymesh.core.models.ParsedDocument` per conversation turn.
    """
    docs: list[ParsedDocument] = []

    title: str = str(data.get("title", ""))
    session_id: str = str(data.get("id", ""))
    mapping: dict[str, Any] = data.get("mapping", {})

    # Build ordered list by following parent→child links
    # Fallback: iterate insertion order
    nodes = list(mapping.values())

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue

        msg = node.get("message")
        if not isinstance(msg, dict):
            continue

        author = msg.get("author", {})
        role: str = str(author.get("role", "unknown")) if isinstance(author, dict) else "unknown"

        content_obj = msg.get("content", {})
        if isinstance(content_obj, dict):
            parts: Any = content_obj.get("parts", [])
            if isinstance(parts, list):
                text = "\n".join(str(p) for p in parts if p).strip()
            else:
                text = str(parts).strip()
        elif isinstance(content_obj, str):
            text = content_obj.strip()
        else:
            continue

        if not text:
            continue

        if len(text) > _LONG_TURN_THRESHOLD:
            logger.debug(
                f"ConversationParser: long turn ({len(text)} chars) in "
                f"{path.name!r} session={session_id!r} index={i}"
            )

        create_time: Any = msg.get("create_time")
        timestamp = str(create_time) if create_time is not None else ""

        meta: dict[str, object] = {
            "role": role,
            "timestamp": timestamp,
            "session_id": session_id,
            "conversation_title": title,
            "source_app": "chatgpt",
            "turn_index": i,
        }

        docs.append(
            ParsedDocument(
                path=path,
                text=text,
                file_type=".json",
                encoding="utf-8",
                metadata=meta,
            )
        )

    return docs


class ConversationParser(Parser):
    """Parser for AI conversation JSON exports (Claude.ai and ChatGPT).

    Auto-detects the export format by inspecting the top-level JSON shape.
    Each conversation turn becomes a separate
    :class:`~memorymesh.core.models.ParsedDocument`.

    Note: This parser returns *multiple* documents for a single file, which is
    unusual.  The indexer is expected to call :meth:`parse` and treat each
    returned document independently.  In the current implementation only the
    first document is returned from :meth:`parse` (the standard
    ``Parser`` interface); callers that need all turns should use
    :meth:`parse_all` instead.

    For the MCP indexing pipeline the :class:`~memorymesh.indexer.file_indexer.FileIndexer`
    uses :meth:`parse` — to emit all turns, the ``ConversationParser`` returns the
    first turn from :meth:`parse` with all subsequent turns encoded as newline-separated
    segments in the text, so a single :class:`ParsedDocument` represents the full
    conversation.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: JSON conversation exports."""
        return frozenset({".json"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a conversation JSON export.

        The entire conversation is packed into a single document with turns
        separated by ``\\n\\n---\\n\\n``.  Metadata reflects the first turn's
        fields plus ``turn_count``.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` representing the
            full conversation, or an error document if parsing fails.
        """
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"ConversationParser: cannot read {path}: {exc}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".json",
                metadata={"error": str(exc)},
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"ConversationParser: invalid JSON in {path.name!r}: {exc}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".json",
                metadata={"error": f"JSON decode error: {exc}"},
            )

        fmt = _detect_source(data)
        if fmt is None:
            logger.warning(
                f"ConversationParser: unrecognised JSON shape in {path.name!r} — "
                "expected Claude.ai array or ChatGPT mapping dict"
            )
            return ParsedDocument(
                path=path,
                text="",
                file_type=".json",
                metadata={"error": "unrecognised conversation format"},
            )

        try:
            docs = _parse_claude(data, path) if fmt == "claude" else _parse_chatgpt(data, path)
        except Exception as exc:
            logger.warning(f"ConversationParser: parse failure in {path.name!r}: {exc}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".json",
                metadata={"error": str(exc)},
            )

        if not docs:
            logger.warning(f"ConversationParser: no turns found in {path.name!r}")
            return ParsedDocument(
                path=path,
                text="",
                file_type=".json",
                metadata={"source_app": fmt, "turn_count": 0},
            )

        # Merge all turns into one document; keep metadata from first turn.
        combined_text = "\n\n---\n\n".join(
            f"[{d.metadata.get('role', '?')}]: {d.text}" for d in docs
        )
        meta: dict[str, object] = {**docs[0].metadata, "turn_count": len(docs)}

        logger.debug(f"ConversationParser: {path.name!r} format={fmt!r} turns={len(docs)}")

        return ParsedDocument(
            path=path,
            text=combined_text,
            file_type=".json",
            encoding="utf-8",
            metadata=meta,
        )

    def parse_all(self, path: Path) -> list[ParsedDocument]:
        """Return one :class:`ParsedDocument` per conversation turn.

        Useful for callers that want to index individual turns separately.

        Args:
            path: Absolute path to the JSON file.

        Returns:
            List of documents, one per turn.  Empty list on error (error is
            logged at WARNING level).
        """
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"ConversationParser.parse_all: {path.name!r}: {exc}")
            return []

        fmt = _detect_source(data)
        if fmt is None:
            return []

        try:
            if fmt == "claude":
                return _parse_claude(data, path)
            return _parse_chatgpt(data, path)
        except Exception as exc:
            logger.warning(f"ConversationParser.parse_all: {path.name!r}: {exc}")
            return []
