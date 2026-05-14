"""Twitter/X data archive connector for MemoryMesh.

Parses the Twitter/X account data archive (downloaded from Settings ->
Your Account -> Download an archive) and yields tweets as
:class:`~memorymesh.core.models.ParsedDocument` objects.

Archive structure
-----------------
The archive may be an unzipped directory *or* a ``.zip`` file.  Inside it,
``data/tweets.js`` holds original tweets and ``data/likes.js`` holds liked
tweets (when ``include_likes`` is ``True``).

Both files are JavaScript files with a variable assignment prefix
(e.g. ``window.YTD.tweets.part0 = [...]``) that is stripped before JSON
parsing.

Features
--------
* **Zip support** - extracts ``.zip`` archives to a temp directory automatically.
* **Retweet skip** - tweets whose text starts with ``RT @`` are dropped.
* **Likes** - optionally indexes liked tweets from ``data/likes.js``.
* **Privacy** - tweet text is never logged at INFO level.
* **Stdlib only** - no third-party dependencies.

Usage
-----
::

    connector = TwitterConnector(TwitterConfig(
        archive_path=Path("~/Downloads/twitter-2024-01-01"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class TwitterConfig(BaseModel):
    """Configuration for a Twitter/X archive source.

    Args:
        archive_path: Path to the unzipped archive directory *or* a ``.zip``
            file.
        include_likes: When ``True``, also index tweets from
            ``data/likes.js``.
        source_name: Name used in the MemoryMesh source registry.
    """

    archive_path: Path
    include_likes: bool = False
    source_name: str = "twitter"


_RE_JS_PREFIX = re.compile(r"^\s*window\.[^=]+=\s*")


def _load_js(path: Path) -> list[Any]:
    """Strip the JS variable prefix and parse a Twitter archive ``.js`` file.

    Args:
        path: Path to a ``.js`` file from the archive (e.g. ``tweets.js``).

    Returns:
        Parsed list of objects, or an empty list on any error.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning(f"TwitterConnector: cannot read {path}: {exc}")
        return []

    raw = _RE_JS_PREFIX.sub("", raw, count=1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"TwitterConnector: JSON parse error in {path.name}: {exc}")
        return []

    return data if isinstance(data, list) else []


class TwitterConnector:
    """Parses a Twitter/X data archive and yields tweets as ParsedDocuments.

    Args:
        config: Archive path and filter settings.
    """

    def __init__(self, config: TwitterConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the Twitter archive and yield tweets as ParsedDocuments.

        Handles both directory and ``.zip`` archive inputs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per tweet,
            with ``file_type=".tweet"`` and metadata containing ``tweet_id``,
            ``created_at``, ``retweets``, and ``likes``.
        """
        p = self._cfg.archive_path

        if p.suffix.lower() == ".zip":
            if not p.exists():
                logger.warning(f"TwitterConnector: zip not found: {p}")
                return
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(tmp)
                except (zipfile.BadZipFile, OSError) as exc:
                    logger.warning(f"TwitterConnector: cannot extract {p}: {exc}")
                    return
                yield from self._process_dir(Path(tmp))
        elif p.is_dir():
            yield from self._process_dir(p)
        else:
            logger.warning(f"TwitterConnector: path not found: {p}")

    def _process_dir(self, archive_dir: Path) -> Iterator[ParsedDocument]:
        """Process an extracted (or pre-existing) archive directory.

        Args:
            archive_dir: Root of the Twitter archive directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per tweet/like.
        """
        # Support both data/tweets.js (standard) and top-level tweets.js
        tweets_js = archive_dir / "data" / "tweets.js"
        if not tweets_js.exists():
            tweets_js = archive_dir / "tweets.js"

        total = 0

        if tweets_js.exists():
            for doc in self._parse_js_file(tweets_js, is_likes=False):
                yield doc
                total += 1
        else:
            logger.warning(f"TwitterConnector: tweets.js not found in {archive_dir}")

        if self._cfg.include_likes:
            likes_js = archive_dir / "data" / "likes.js"
            if not likes_js.exists():
                likes_js = archive_dir / "likes.js"
            if likes_js.exists():
                for doc in self._parse_js_file(likes_js, is_likes=True):
                    yield doc
                    total += 1

        logger.info(f"TwitterConnector: yielded {total} document(s)")

    def _parse_js_file(
        self,
        path: Path,
        *,
        is_likes: bool,
    ) -> Iterator[ParsedDocument]:
        """Parse one ``tweets.js`` or ``likes.js`` and yield ParsedDocuments.

        Args:
            path: Path to the ``.js`` file.
            is_likes: ``True`` when parsing a ``likes.js`` file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per tweet/like.
        """
        items = _load_js(path)
        source = self._cfg.source_name
        yielded = 0
        skipped = 0

        for item in items:
            if is_likes:
                obj = item.get("like", item)
                tweet_id = str(obj.get("tweetId", obj.get("id_str", "")))
                full_text = str(obj.get("fullText", obj.get("full_text", "")))
                created_at = ""
                retweets = 0
                likes = 0
            else:
                obj = item.get("tweet", item)
                tweet_id = str(obj.get("id_str", ""))
                full_text = str(obj.get("full_text", ""))
                created_at = str(obj.get("created_at", ""))
                retweets = int(obj.get("retweet_count", 0) or 0)
                likes = int(obj.get("favorite_count", 0) or 0)

            if not full_text.strip():
                skipped += 1
                continue
            if not is_likes and full_text.startswith("RT @"):
                skipped += 1
                continue

            if not tweet_id:
                tweet_id = f"unknown_{yielded}"

            parts: list[str] = []
            if created_at:
                parts.append(created_at)
                parts.append("")
            parts.append(full_text)
            text = "\n".join(parts)

            yield ParsedDocument(
                path=Path(f"twitter://{tweet_id}.tweet"),
                text=text,
                file_type=".tweet",
                encoding="utf-8",
                metadata={
                    "tweet_id": tweet_id,
                    "created_at": created_at,
                    "retweets": retweets,
                    "likes": likes,
                    "source": source,
                },
            )
            yielded += 1

        logger.info(f"TwitterConnector: {path.name} -> yielded={yielded} skipped={skipped}")


# Backward-compatible aliases
TwitterArchiveConfig = TwitterConfig
TwitterArchiveConnector = TwitterConnector
