"""Instagram data export connector for MemoryMesh.

Parses Meta/Instagram data exports (Settings -> Your Activity ->
Download Your Information -> JSON format) and yields
:class:`~memorymesh.core.models.ParsedDocument` objects for posts and
message threads.

Encoding note
-------------
Instagram (like Facebook) mis-encodes UTF-8 strings as Latin-1 code
points in its JSON exports.  The :func:`_fix_encoding` helper corrects
this by re-encoding the string as Latin-1 bytes and decoding as UTF-8.

Files parsed
------------
* ``your_instagram_activity/content/posts_1.json``
* ``your_instagram_activity/messages/inbox/{thread}/message_1.json``

Both paths have fallbacks for older export layouts.

Usage
-----
::

    connector = InstagramConnector(InstagramConfig(
        export_path=Path("~/Downloads/instagram-export"),
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


class InstagramConfig(BaseModel):
    """Configuration for an Instagram data export source.

    Args:
        export_path: Directory containing the extracted Instagram archive,
            or a ``.zip`` archive.
        include_posts: Parse ``posts_1.json`` when ``True``.
        include_messages: Parse inbox message threads when ``True``.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    include_posts: bool = True
    include_messages: bool = True
    source_name: str = "instagram"


def _fix_encoding(text: str) -> str:
    """Fix Meta's broken UTF-8 encoding in Instagram JSON exports.

    Instagram encodes UTF-8 strings as Latin-1 code points in the export
    JSON, identical to the Facebook encoding bug.  Re-encoding as Latin-1
    then decoding as UTF-8 restores the original characters.

    Args:
        text: Potentially mis-encoded string from the Instagram archive.

    Returns:
        Correctly decoded string, or the original string on failure.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class InstagramConnector:
    """Parses Instagram archive JSON and yields ParsedDocuments.

    Args:
        config: Export path, feature toggles, and source settings.
    """

    def __init__(self, config: InstagramConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Extract archive if needed and yield post and message docs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            post caption (``file_type=".instagram"``) or one per message
            thread.
        """
        export_path = self._cfg.export_path.expanduser()
        if export_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(export_path) as zf:
                        zf.extractall(tmp)
                except Exception as exc:
                    logger.warning(f"InstagramConnector: cannot open zip {export_path}: {exc}")
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
        logger.info(f"InstagramConnector: yielded {total} document(s)")

    def _find_posts_file(self, base: Path) -> Path | None:
        """Locate ``posts_1.json`` using several candidate paths.

        Args:
            base: Archive root directory.

        Returns:
            :class:`~pathlib.Path` to the file, or ``None`` if absent.
        """
        candidates = [
            base / "your_instagram_activity" / "content" / "posts_1.json",
            base / "content" / "posts_1.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        found = list(base.rglob("posts_1.json"))
        return found[0] if found else None

    def _parse_posts(self, base: Path) -> Iterator[ParsedDocument]:
        """Parse ``posts_1.json`` and yield one document per caption.

        Args:
            base: Archive root directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per post.
        """
        posts_file = self._find_posts_file(base)
        if posts_file is None:
            return
        try:
            data = json.loads(posts_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"InstagramConnector: cannot read {posts_file}: {exc}")
            return
        if not isinstance(data, list):
            return

        source = self._cfg.source_name
        for entry in data:
            for media in entry.get("media", []):
                ts = media.get("creation_timestamp", 0)
                caption = _fix_encoding((media.get("title") or "").strip())
                if not caption:
                    continue
                yield ParsedDocument(
                    path=Path(f"instagram://posts/{ts}.instagram"),
                    text=caption,
                    file_type=".instagram",
                    encoding="utf-8",
                    metadata={
                        "type": "post",
                        "timestamp": ts,
                        "caption_length": len(caption),
                        "source": source,
                    },
                )

    def _find_inbox_dir(self, base: Path) -> Path | None:
        """Locate the messages inbox directory.

        Args:
            base: Archive root directory.

        Returns:
            :class:`~pathlib.Path` to the inbox, or ``None`` if absent.
        """
        candidates = [
            base / "your_instagram_activity" / "messages" / "inbox",
            base / "messages" / "inbox",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _parse_messages(self, base: Path) -> Iterator[ParsedDocument]:
        """Parse all message threads under the inbox directory.

        Args:
            base: Archive root directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per thread.
        """
        inbox_dir = self._find_inbox_dir(base)
        if inbox_dir is None:
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
                logger.warning(f"InstagramConnector: cannot read {msg_file}: {exc}")
                continue

            thread_title = _fix_encoding(data.get("title", thread_dir.name))
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
                path=Path(f"instagram://messages/{safe_name}.instagram"),
                text="\n".join(lines),
                file_type=".instagram",
                encoding="utf-8",
                metadata={
                    "type": "message",
                    "thread": thread_title,
                    "message_count": len(lines),
                    "source": source,
                },
            )
