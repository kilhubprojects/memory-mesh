"""IMAP email source connector for MemoryMesh.

Connects to an IMAP server, fetches messages in a configurable mailbox, and
yields :class:`~memorymesh.core.models.ParsedDocument` objects that the normal
indexing pipeline can chunk and embed.

Features
--------
* **Incremental sync** - tracks the highest ``UID`` seen and only fetches new
  messages on subsequent runs.  UID checkpoints are stored in the
  :class:`~memorymesh.storage.metadata_store.MetadataStore` under a synthetic
  path key ``imap://<host>/<mailbox>``.
* **HTML stripping** - HTML-only messages are stripped to plain text via a
  minimal ``html.parser`` pass.
* **Privacy** - message bodies are never logged; only UID ranges, message
  counts, and subject line lengths are written to the log.
* **Stdlib only** - uses ``imaplib`` and ``email`` from the standard library;
  no third-party dependency required.

Usage
-----
Instantiate :class:`IMAPConnector` and call :meth:`fetch_documents`::

    connector = IMAPConnector(IMAPConfig(
        host="imap.gmail.com",
        username="you@gmail.com",
        password="app-password",
        mailbox="INBOX",
        max_messages=500,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import contextlib
import email
import email.header
import email.message
import email.policy
import email.utils
import imaplib
import re
import time
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._html import html_to_text
from memorymesh.core.models import ParsedDocument


class IMAPConfig(BaseModel):
    """Configuration for a single IMAP email source.

    Args:
        host: IMAP server hostname.
        port: IMAP server port.  Default 993 (IMAPS).
        use_ssl: Whether to use SSL (recommended).
        username: IMAP login name.
        password: IMAP password or app password.
        mailbox: Mailbox / folder to sync (default ``INBOX``).
        max_messages: Cap on total messages fetched per run (0 = no limit).
        batch_size: Messages fetched per IMAP FETCH command.
        fetch_body: Whether to download full bodies (vs. headers only).
        source_name: Name used in the MemoryMesh source registry.
    """

    host: str
    port: int = 993
    use_ssl: bool = True
    username: str
    password: SecretStr
    mailbox: str = "INBOX"
    max_messages: int = 1_000
    batch_size: int = 50
    fetch_body: bool = True
    source_name: str = ""


def _decode_header_value(raw: str | None) -> str:
    """Decode an RFC 2047-encoded header value to a plain string.

    Args:
        raw: Raw header value string (may contain ``=?charset?encoding?...?=``).

    Returns:
        Decoded string, or ``""`` if *raw* is falsy.
    """
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            decoded.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(chunk)
    return "".join(decoded)


def _extract_body(msg: email.message.Message) -> str:
    """Extract the best available body text from an email message.

    Preference order: ``text/plain`` > ``text/html`` (stripped).

    Args:
        msg: A parsed :class:`email.message.Message`.

    Returns:
        Plain-text body string (may be empty for non-text attachments).
    """
    plain: str | None = None
    html_body: str | None = None

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if ct == "text/plain" and plain is None:
                plain = text
            elif ct == "text/html" and html_body is None:
                html_body = text
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_body = text
            else:
                plain = text

    if plain is not None:
        return plain.strip()
    if html_body is not None:
        return html_to_text(html_body)
    return ""


class IMAPConnector:
    """Fetches email messages from an IMAP server as ParsedDocuments.

    The connector maintains state via a UID checkpoint so that subsequent
    calls to :meth:`fetch_documents` only download new messages.

    Args:
        config: IMAP connection and fetch settings.
        checkpoint_path: Optional file path for persisting the last-seen UID
            across runs.  When ``None``, each run starts from scratch.
    """

    def __init__(
        self,
        config: IMAPConfig,
        checkpoint_path: Path | None = None,
    ) -> None:
        self._cfg = config
        self._checkpoint_path = checkpoint_path
        self._last_uid: int = self._load_checkpoint()

    def _load_checkpoint(self) -> int:
        """Return the last-seen UID from disk, or 0 if not set."""
        if self._checkpoint_path and self._checkpoint_path.exists():
            try:
                return int(self._checkpoint_path.read_text().strip())
            except (ValueError, OSError) as exc:
                logger.debug(f"IMAPConnector: checkpoint unreadable, starting at zero: {exc}")
        return 0

    def _save_checkpoint(self, uid: int) -> None:
        """Write the last-seen UID to disk.

        Args:
            uid: The highest UID successfully fetched.
        """
        if self._checkpoint_path:
            try:
                self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self._checkpoint_path.write_text(str(uid))
            except OSError as exc:
                logger.warning(f"IMAPConnector: cannot save checkpoint: {exc}")

    def _connect(self) -> imaplib.IMAP4 | imaplib.IMAP4_SSL:
        """Open and authenticate the IMAP connection.

        Returns:
            Authenticated IMAP connection with the configured mailbox selected.

        Raises:
            OSError: If connection or login fails.
        """
        conn: imaplib.IMAP4 = (
            imaplib.IMAP4_SSL(self._cfg.host, self._cfg.port)
            if self._cfg.use_ssl
            else imaplib.IMAP4(self._cfg.host, self._cfg.port)
        )

        password = self._cfg.password.get_secret_value()
        try:
            conn.login(self._cfg.username, password)
        except imaplib.IMAP4.error as exc:
            raise OSError(f"IMAP login failed for {self._cfg.username}: {exc}") from exc

        status, _ = conn.select(self._cfg.mailbox, readonly=True)
        if status != "OK":
            raise OSError(f"Cannot select mailbox {self._cfg.mailbox!r} on {self._cfg.host}")
        return conn

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Connect to IMAP and yield new email messages as ParsedDocuments.

        Only messages with UID > ``last_uid`` are downloaded.  The checkpoint
        is updated after each successful batch.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per message,
            with ``file_type=".eml"`` and metadata containing ``subject``,
            ``from_addr``, ``to_addrs``, ``date_ts``, and ``uid``.

        Raises:
            OSError: On connection / auth failures (caller should catch and log).
        """
        logger.info(
            f"IMAPConnector: connecting to {self._cfg.host} "
            f"mailbox={self._cfg.mailbox!r} last_uid={self._last_uid}"
        )

        try:
            conn = self._connect()
        except OSError as exc:
            logger.error(f"IMAPConnector: connection failed: {exc}")
            return

        try:
            yield from self._fetch_loop(conn)
        finally:
            with contextlib.suppress(Exception):
                conn.logout()

    def _fetch_loop(
        self,
        conn: imaplib.IMAP4 | imaplib.IMAP4_SSL,
    ) -> Iterator[ParsedDocument]:
        """Core fetch logic - searches UIDs and fetches in batches.

        Args:
            conn: Authenticated and selected IMAP connection.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per message.
        """
        search_criteria = f"UID {self._last_uid + 1}:*"
        status, data = conn.uid("search", None, search_criteria)  # type: ignore[arg-type]
        if status != "OK" or not data or not data[0]:
            logger.info("IMAPConnector: no new messages found")
            return

        uid_list_raw = data[0]
        uid_str = uid_list_raw.decode() if isinstance(uid_list_raw, bytes) else str(uid_list_raw)

        uids = [int(u) for u in uid_str.split() if u.strip().isdigit()]
        if not uids:
            logger.info("IMAPConnector: search returned no UIDs")
            return

        # Filter out UIDs we've already seen (handles IMAP UID * edge case).
        uids = [u for u in uids if u > self._last_uid]
        if not uids:
            return

        if self._cfg.max_messages > 0:
            uids = uids[: self._cfg.max_messages]

        logger.info(f"IMAPConnector: {len(uids)} new message(s) to fetch")

        fetched = 0
        highest_uid = self._last_uid

        for batch_start in range(0, len(uids), self._cfg.batch_size):
            batch = uids[batch_start : batch_start + self._cfg.batch_size]
            uid_range = ",".join(str(u) for u in batch)

            fetch_spec = "(RFC822)" if self._cfg.fetch_body else "(RFC822.HEADER)"
            status, items = conn.uid("fetch", uid_range, fetch_spec)  # type: ignore[arg-type]

            if status != "OK" or not items:
                logger.warning(f"IMAPConnector: FETCH failed for UIDs {uid_range[:40]!r}")
                continue

            for item in items:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                raw_bytes = item[1]
                if not isinstance(raw_bytes, bytes):
                    continue

                try:
                    doc = self._parse_message(raw_bytes, uid=batch[fetched % len(batch)])
                except Exception as exc:
                    logger.warning(f"IMAPConnector: parse error: {exc}")
                    continue

                if doc is not None:
                    uid_val = int(doc.metadata.get("uid", 0))
                    highest_uid = max(highest_uid, uid_val)
                    yield doc

                fetched += 1

            # Save checkpoint after each successful batch
            if highest_uid > self._last_uid:
                self._last_uid = highest_uid
                self._save_checkpoint(highest_uid)
                logger.debug(f"IMAPConnector: checkpoint updated uid={highest_uid}")

    def _parse_message(self, raw: bytes, uid: int) -> ParsedDocument | None:
        """Parse a raw RFC 822 message into a ParsedDocument.

        Args:
            raw: Raw message bytes.
            uid: IMAP UID for this message.

        Returns:
            Parsed document or ``None`` if the message is empty.
        """
        msg = email.message_from_bytes(raw, policy=email.policy.compat32)

        subject = _decode_header_value(msg.get("Subject"))
        from_addr = _decode_header_value(msg.get("From"))
        to_addrs = _decode_header_value(msg.get("To"))
        date_raw = msg.get("Date", "")
        try:
            date_ts = email.utils.parsedate_to_datetime(date_raw).timestamp()
        except Exception:
            date_ts = time.time()

        body = _extract_body(msg) if self._cfg.fetch_body else ""

        if not subject.strip() and not body.strip():
            return None

        # Build the document text as a structured block
        text_parts: list[str] = []
        if subject:
            text_parts.append(f"Subject: {subject}")
        if from_addr:
            text_parts.append(f"From: {from_addr}")
        if to_addrs:
            text_parts.append(f"To: {to_addrs}")
        if date_raw:
            text_parts.append(f"Date: {date_raw}")
        text_parts.append("")
        if body:
            text_parts.append(body)

        text = "\n".join(text_parts)

        # Synthetic path so the metadata store can key on it
        source = self._cfg.source_name or self._cfg.mailbox
        synthetic_path = Path(f"imap://{self._cfg.host}/{self._cfg.mailbox}/{uid}.eml")

        return ParsedDocument(
            path=synthetic_path,
            text=text,
            file_type=".eml",
            encoding="utf-8",
            metadata={
                "uid": uid,
                "subject": subject,
                "from_addr": from_addr,
                "to_addrs": to_addrs,
                "date_ts": date_ts,
                "source": source,
                "mailbox": self._cfg.mailbox,
                "host": self._cfg.host,
            },
        )

    _RE_WHITESPACE: re.Pattern[str] = re.compile(r"\s{2,}")
