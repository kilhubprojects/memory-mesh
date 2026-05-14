"""Facebook/Meta data archive connector for MemoryMesh.

Parses Facebook data exports (Settings -> Your Facebook Information ->
Download Your Information -> JSON format) and yields
:class:`~memorymesh.core.models.ParsedDocument` objects for posts and
message threads.

Encoding note
-------------
Facebook mis-encodes UTF-8 strings in its exports: the raw bytes of a
UTF-8 character are each stored as the corresponding Latin-1 code point.
For example, ``?`` (UTF-8 bytes ``0xC3 0xA9``) appears as ``ÃƒÂ©`` in the
exported JSON.  The :func:`_fix_encoding` helper corrects this.

Files parsed
------------
* ``your_posts/your_posts_1.json``
* ``messages/inbox/{thread}/message_1.json``

Usage
-----
::

    connector = FacebookConnector(FacebookConfig(
        export_path=Path("~/Downloads/facebook-export"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class FacebookConfig(BaseModel):
    """Configuration for a Facebook data archive source.

    Args:
        export_path: Directory containing the extracted Facebook archive,
            or a ``.zip`` archive.
        include_posts: Parse ``your_posts_1.json`` when ``True``.
        include_messages: Parse inbox message threads when ``True``.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    include_posts: bool = True
    include_messages: bool = True
    source_name: str = "facebook"


def _fix_encoding(text: str) -> str:
    """Fix Facebook's broken UTF-8 encoding in JSON exports.

    Facebook encodes UTF-8 strings as Latin-1 code points.  For example,
    ``?`` (UTF-8: 0xC3 0xA9) appears as ``\\u00c3\\u00a9`` in the JSON.
    Re-encoding as Latin-1 then decoding as UTF-8 restores the original.

    Args:
        text: Potentially mis-encoded string from the Facebook archive.

    Returns:
        Correctly decoded string, or the original on failure.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class FacebookConnector:
    """Parses Facebook archive JSON and yields ParsedDocuments.

    Args:
        config: Export path, feature toggles, and source settings.
    """

    def __init__(self, config: FacebookConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Extract archive if needed and yield posts and message docs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            post (``file_type=".facebook"``) or one per message thread.
        """
        export_path = self._cfg.export_path.expanduser()
        if export_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(export_path) as zf:
                        zf.extractall(tmp)
                except Exception as exc:
                    logger.warning(f"FacebookConnector: cannot open zip {export_path}: {exc}")
                    return
                yield from self._process_dir(Path(tmp))
        else:
            yield from self._process_dir(export_path)

    def _process_dir(self, base: Path) -> Iterator[ParsedDocument]:
        """Dispatch to per-content parsers.

        Args:
            base: Root of the extracted archive.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        total = 0
        if self._cfg.include_posts:
            for doc in self._parse_posts(base):
                yield doc
                total += 1
        if self._cfg.include_messages:
            for doc in self._parse_messages(base):
                yield doc
                total += 1
        logger.info(f"FacebookConnector: yielded {total} document(s)")

    def _parse_posts(self, base: Path) -> Iterator[ParsedDocument]:
        """Parse ``your_posts/your_posts_1.json``.

        Args:
            base: Archive root directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per post.
        """
        posts_file = base / "your_posts" / "your_posts_1.json"
        if not posts_file.exists():
            return
        try:
            data = json.loads(posts_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"FacebookConnector: cannot read {posts_file}: {exc}")
            return
        if not isinstance(data, list):
            return

        source = self._cfg.source_name
        for entry in data:
            ts = entry.get("timestamp", 0)
            title = _fix_encoding(entry.get("title", "") or "")
            post_text = ""
            for d in entry.get("data", []):
                if "post" in d:
                    post_text = _fix_encoding(d["post"])
                    break
            content = post_text or title
            if not content:
                continue
            dt_str = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
            yield ParsedDocument(
                path=Path(f"facebook://posts/{ts}.facebook"),
                text=content,
                file_type=".facebook",
                encoding="utf-8",
                metadata={
                    "type": "post",
                    "timestamp": ts,
                    "date": dt_str,
                    "title": title,
                    "source": source,
                },
            )

    def _parse_messages(self, base: Path) -> Iterator[ParsedDocument]:
        """Parse all message threads under ``messages/inbox/``.

        Args:
            base: Archive root directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per thread.
        """
        inbox_dir = base / "messages" / "inbox"
        if not inbox_dir.exists():
            return
        source = self._cfg.source_name

        for thread_dir in sorted(inbox_dir.iterdir()):
            if not thread_dir.is_dir():
                continue
            msg_file = thread_dir / "message_1.json"
            if not msg_file.exists():
                continue
            try:
                data = json.loads(msg_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"FacebookConnector: cannot read {msg_file}: {exc}")
                continue

            thread_title = _fix_encoding(data.get("title", thread_dir.name))
            participants = data.get("participants", [])
            raw_messages = data.get("messages", [])

            lines: list[str] = []
            for msg in sorted(
                raw_messages,
                key=lambda m: m.get("timestamp_ms", 0),
            ):
                sender = _fix_encoding(msg.get("sender_name", ""))
                content = _fix_encoding((msg.get("content") or "").strip())
                if not content:
                    continue
                ts_ms = msg.get("timestamp_ms", 0)
                dt_str = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")
                lines.append(f"{dt_str} {sender}: {content}")

            if not lines:
                continue

            safe_name = thread_dir.name.replace(" ", "_")
            yield ParsedDocument(
                path=Path(f"facebook://messages/{safe_name}.facebook"),
                text="\n".join(lines),
                file_type=".facebook",
                encoding="utf-8",
                metadata={
                    "type": "message",
                    "thread": thread_title,
                    "participant_count": len(participants),
                    "message_count": len(lines),
                    "source": source,
                },
            )


# Backward-compatible alias
FacebookArchiveConnector = FacebookConnector
