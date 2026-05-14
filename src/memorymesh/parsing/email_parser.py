"""Email parser for Unix mbox files.

Uses only stdlib: ``mailbox.mbox`` for reading and ``html.parser`` for stripping
HTML from ``text/html`` parts.

Each email in the mbox becomes one :class:`~memorymesh.core.models.ParsedDocument`.
Binary attachments are skipped with a WARNING log entry.  The ``max_messages``
limit (from :class:`~memorymesh.core.models.EmailSourceConfig`) prevents
accidentally indexing a huge mbox on first run.

Activated for ``.mbox`` files when a source is configured with
``source.type: email`` (or simply by having ``.mbox`` in the extension list).
"""

from __future__ import annotations

import contextlib
import mailbox
from email.header import decode_header as _decode_header
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

_DEFAULT_MAX_MESSAGES = 10_000


class _HTMLStripper(HTMLParser):
    """Minimal HTML→plaintext converter using only stdlib."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track entry into script/style blocks to suppress their content."""
        if tag.lower() in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        """Track exit from script/style blocks."""
        if tag.lower() in ("script", "style"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        """Collect visible text, ignoring script/style content."""
        if self._skip == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        """The accumulated plain-text content extracted from the HTML."""
        return "\n".join(self._parts)


def _strip_html(html: str) -> str:
    """Convert HTML to plain text, discarding markup.

    Args:
        html: Raw HTML string.

    Returns:
        Extracted plain text (markup removed).
    """
    stripper = _HTMLStripper()
    with contextlib.suppress(Exception):
        stripper.feed(html)
    return stripper.text


def _decode_header_value(raw: Any) -> str:
    """Decode an RFC 2047-encoded email header to a plain string.

    Args:
        raw: The header value as returned by ``email.message.Message.__getitem__``.

    Returns:
        Decoded header string, or empty string if decoding fails.
    """
    if raw is None:
        return ""
    try:
        parts: list[str] = []
        for chunk, charset in _decode_header(str(raw)):
            if isinstance(chunk, bytes):
                enc = charset or "utf-8"
                try:
                    parts.append(chunk.decode(enc, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    parts.append(chunk.decode("latin-1", errors="replace"))
            else:
                parts.append(chunk)
        return " ".join(parts)
    except Exception:
        return str(raw)


def _extract_text(msg: mailbox.mboxMessage) -> tuple[str, bool]:
    """Extract plain text from an email message.

    Strategy:
    1. Walk all parts, collect ``text/plain`` payloads.
    2. If no plain-text part, fall back to ``text/html`` → strip tags.
    3. Binary attachments: skip and log warning.

    Args:
        msg: An mbox message object.

    Returns:
        ``(text, had_attachment)`` where ``had_attachment`` is True when at
        least one binary part was skipped.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    had_attachment = False

    for part in msg.walk():
        ctype = part.get_content_type().lower()
        disposition = str(part.get("Content-Disposition", "")).lower()

        if "attachment" in disposition:
            had_attachment = True
            logger.debug(
                f"EmailParser: skipping attachment name={part.get_filename()!r} type={ctype!r}"
            )
            continue

        if ctype == "text/plain":
            try:
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        plain_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        plain_parts.append(payload.decode("latin-1", errors="replace"))
            except Exception as exc:
                logger.warning(f"EmailParser: could not decode text/plain part: {exc}")

        elif ctype == "text/html":
            try:
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        html = payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        html = payload.decode("latin-1", errors="replace")
                    html_parts.append(_strip_html(html))
            except Exception as exc:
                logger.warning(f"EmailParser: could not decode text/html part: {exc}")

        # All other content types (image/*, application/*) → skip silently
        # unless they carry an attachment disposition (handled above).

    if plain_parts:
        return "\n\n".join(plain_parts).strip(), had_attachment
    if html_parts:
        return "\n\n".join(html_parts).strip(), had_attachment
    return "", had_attachment


class EmailParser(Parser):
    """Parser for Unix mbox files.

    Each email in the mbox is converted to a separate
    :class:`~memorymesh.core.models.ParsedDocument`.  Because the standard
    ``Parser.parse`` interface returns one document per call (one *file*),
    this parser returns the text of all messages concatenated with separators,
    and surfaces per-message metadata for only the *first* message.

    For fine-grained per-message indexing, callers can use :meth:`parse_all`.

    Args:
        max_messages: Maximum number of messages to process per mbox file.
            Messages beyond this limit are silently ignored.
    """

    def __init__(self, max_messages: int = _DEFAULT_MAX_MESSAGES) -> None:
        self._max_messages = max_messages

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: mbox and eml email files."""
        return frozenset({".mbox"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse an mbox file, returning all messages as one document.

        Args:
            path: Absolute path to the ``.mbox`` file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with the combined
            text of all messages and metadata from the first message.
        """
        docs = self.parse_all(path)
        if not docs:
            return ParsedDocument(
                path=path,
                text="",
                file_type=".mbox",
                metadata={"message_count": 0},
            )

        combined = "\n\n---\n\n".join(
            f"From: {d.metadata.get('from_addr', '')}\n"
            f"Subject: {d.metadata.get('subject', '')}\n\n"
            f"{d.text}"
            for d in docs
        )
        meta: dict[str, object] = {**docs[0].metadata, "message_count": len(docs)}
        return ParsedDocument(
            path=path,
            text=combined,
            file_type=".mbox",
            encoding="utf-8",
            metadata=meta,
        )

    def parse_all(self, path: Path) -> list[ParsedDocument]:
        """Return one :class:`ParsedDocument` per email message.

        Args:
            path: Absolute path to the ``.mbox`` file.

        Returns:
            List of documents, one per message.  Stops at ``max_messages``.
            Returns empty list on error (error is logged at WARNING level).
        """
        try:
            mbox = mailbox.mbox(str(path), create=False)
        except Exception as exc:
            logger.warning(f"EmailParser: cannot open mbox {path}: {exc}")
            return []

        docs: list[ParsedDocument] = []

        try:
            for i, msg in enumerate(mbox):
                if i >= self._max_messages:
                    logger.warning(
                        f"EmailParser: reached max_messages={self._max_messages} "
                        f"for {path.name!r} — remaining messages skipped"
                    )
                    break

                text, had_attachment = _extract_text(msg)
                if not text:
                    logger.debug(f"EmailParser: empty message at index {i} in {path.name!r}")

                from_addr = _decode_header_value(msg.get("From", ""))
                to_addr = _decode_header_value(msg.get("To", ""))
                subject = _decode_header_value(msg.get("Subject", ""))
                date = _decode_header_value(msg.get("Date", ""))
                message_id = _decode_header_value(msg.get("Message-ID", ""))

                meta: dict[str, object] = {
                    "from_addr": from_addr,
                    "to_addr": to_addr,
                    "subject": subject,
                    "date": date,
                    "message_id": message_id,
                    "had_attachment": had_attachment,
                    "message_index": i,
                }

                docs.append(
                    ParsedDocument(
                        path=path,
                        text=text,
                        file_type=".mbox",
                        encoding="utf-8",
                        metadata=meta,
                    )
                )
        except Exception as exc:
            logger.warning(f"EmailParser: error iterating {path.name!r}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                mbox.close()

        logger.debug(f"EmailParser: {path.name!r} messages_parsed={len(docs)}")
        return docs
