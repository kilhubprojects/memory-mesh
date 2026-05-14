"""WhatsApp chat export connector for MemoryMesh.

Parses WhatsApp chat export ``.txt`` files (exported via "Export Chat" in
the mobile app) and yields conversation chunks as
:class:`~memorymesh.core.models.ParsedDocument` objects.

Features
--------
* **Dual-format** - handles both Android (``MM/DD/YYYY, HH:MM - Sender: text``)
  and iOS (``[DD/MM/YYYY, HH:MM:SS] Sender: text``) export variants.
* **Multi-file** - if ``export_path`` is a directory, all ``.txt`` files are
  processed in sorted order.
* **Chunking** - messages are grouped into ~``chunk_size``-message documents.
* **Media skip** - ``<Media omitted>`` and similar lines are silently dropped.
* **Privacy** - message content is never logged at INFO level.
* **Stdlib only** - no third-party dependencies.

Usage
-----
::

    connector = WhatsAppConnector(WhatsAppConfig(
        export_path=Path("~/Downloads/WhatsApp Chat with Alice.txt"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class WhatsAppConfig(BaseModel):
    """Configuration for a WhatsApp chat export source.

    Args:
        export_path: Path to a ``.txt`` export file *or* a directory
            containing multiple export ``.txt`` files.
        chunk_size: Number of messages grouped into one ParsedDocument.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    chunk_size: int = 50
    source_name: str = "whatsapp"


# Android: 12/31/2024, 14:30 - Alice: Hello
_RE_ANDROID = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s[AP]M)?)\s-\s"
    r"([^:]+):\s(.*)$"
)
# iOS: [31/12/2024, 14:30:00] Alice: Hello
_RE_IOS = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s"
    r"(\d{1,2}:\d{2}:\d{2}(?:\s[AP]M)?)\]\s"
    r"([^:]+):\s(.*)$"
)
# Lines to drop: <Media omitted>, image omitted, video omitted, etc.
_RE_MEDIA = re.compile(
    r"^<[^>]+>$|^(?:image|video|audio|sticker|GIF|document|contact)\s+omitted$",
    re.IGNORECASE,
)


@dataclass
class _Msg:
    """A single parsed WhatsApp message."""

    timestamp: str
    sender: str
    text: str


def _safe_name(text: str) -> str:
    """Sanitise *text* for use in a synthetic URL path segment.

    Args:
        text: Raw string (chat name, filename stem, etc.).

    Returns:
        Alphanumeric + safe punctuation only, max 60 characters.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)[:60]


def _detect_format(lines: list[str]) -> str:
    """Detect the WhatsApp export format from the first meaningful line.

    Args:
        lines: All text lines from the export file.

    Returns:
        ``"ios"`` if the iOS bracket format is detected; ``"android"``
        otherwise (including when the format cannot be determined).
    """
    for line in lines[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        if _RE_IOS.match(stripped):
            return "ios"
        if _RE_ANDROID.match(stripped):
            return "android"
    return "android"


def _parse_file(text: str) -> list[_Msg]:
    """Parse the full text of a WhatsApp export into a list of messages.

    Handles multi-line messages (continuation lines that have no timestamp).
    Silently drops media omission lines.  Strips the UTF-8 BOM if present.

    Args:
        text: Raw text content of the export file.

    Returns:
        List of :class:`_Msg` in chronological order.
    """
    text = text.lstrip("ï»¿")
    lines = text.splitlines()
    pattern = _RE_IOS if _detect_format(lines) == "ios" else _RE_ANDROID
    messages: list[_Msg] = []
    current: _Msg | None = None

    for line in lines:
        m = pattern.match(line)
        if m:
            date, time_s = m.group(1), m.group(2)
            sender = m.group(3).strip()
            body = m.group(4).strip()
            if _RE_MEDIA.match(body):
                current = None
                continue
            current = _Msg(timestamp=f"{date}, {time_s}", sender=sender, text=body)
            messages.append(current)
        else:
            stripped = line.strip()
            if current is not None and stripped and not _RE_MEDIA.match(stripped):
                current.text = f"{current.text}\n{stripped}" if current.text else stripped

    return messages


class WhatsAppConnector:
    """Parses WhatsApp chat export files and yields conversation chunks.

    Args:
        config: Export path and chunking settings.
    """

    def __init__(self, config: WhatsAppConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse WhatsApp export(s) and yield message chunks as ParsedDocuments.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            ~``chunk_size`` messages, with ``file_type=".whatsapp"`` and
            metadata containing ``chat_name``, ``message_count``,
            ``start_date``, ``end_date``, and ``participants``.
        """
        p = self._cfg.export_path
        if p.is_dir():
            files = sorted(p.glob("*.txt"))
            if not files:
                logger.warning(f"WhatsAppConnector: no .txt files found in {p}")
                return
        elif p.is_file():
            files = [p]
        else:
            logger.warning(f"WhatsAppConnector: path not found: {p}")
            return

        total = 0
        for f in files:
            for doc in self._process_file(f):
                yield doc
                total += 1

        logger.info(f"WhatsAppConnector: yielded {total} chunk(s)")

    def _process_file(self, path: Path) -> Iterator[ParsedDocument]:
        """Parse one export file and yield message-chunk ParsedDocuments.

        Args:
            path: Path to a WhatsApp ``.txt`` export file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per message chunk.
        """
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning(f"WhatsAppConnector: cannot read {path}: {exc}")
            return

        messages = _parse_file(raw)
        if not messages:
            logger.info(f"WhatsAppConnector: no messages parsed in {path.name!r}")
            return

        logger.info(f"WhatsAppConnector: {len(messages)} message(s) in {path.name!r}")

        chat_name = path.stem
        safe = _safe_name(chat_name)
        source = self._cfg.source_name
        chunk_size = self._cfg.chunk_size

        for chunk_idx, offset in enumerate(range(0, len(messages), chunk_size)):
            chunk = messages[offset : offset + chunk_size]
            participants = sorted({m.sender for m in chunk})
            block = "\n".join(f"{m.timestamp} {m.sender}: {m.text}" for m in chunk)

            yield ParsedDocument(
                path=Path(f"whatsapp://{safe}/{chunk_idx}.whatsapp"),
                text=block,
                file_type=".whatsapp",
                encoding="utf-8",
                metadata={
                    "chat_name": chat_name,
                    "message_count": len(chunk),
                    "start_date": chunk[0].timestamp,
                    "end_date": chunk[-1].timestamp,
                    "participants": participants,
                    "source": source,
                },
            )
