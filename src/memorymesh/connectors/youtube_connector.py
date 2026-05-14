"""YouTube watch history connector for MemoryMesh.

Parses ``watch-history.json`` from Google Takeout and yields one
:class:`~memorymesh.core.models.ParsedDocument` per calendar month.

Export format (watch-history.json)::

    [
      {
        "header": "YouTube",
        "title": "Watched Video Title",
        "titleUrl": "https://www.youtube.com/watch?v=...",
        "subtitles": [{"name": "Channel Name", "url": "..."}],
        "time": "2024-01-01T10:00:00.000Z"
      }
    ]

Features
--------
* **Auto-detection** - accepts ``watch-history.json`` directly, a
  directory, or a ``.zip`` archive.
* **Date filtering** - only entries within ``days_past`` are included.
* **Skip removed videos** - entries whose title starts with
  ``"Watched a video that has been removed"`` are skipped.
* **Top channels** - up to 10 most-watched channels per month are
  stored in metadata.

Usage
-----
::

    connector = YouTubeConnector(YouTubeConfig(
        export_path=Path("~/Takeout/YouTube"),
        days_past=365,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class YouTubeConfig(BaseModel):
    """Configuration for a YouTube watch history source.

    Args:
        export_path: Path to ``watch-history.json``, a Takeout directory,
            or a ``.zip`` archive containing the file.
        days_past: Only include entries within this many days.  0 = no
            cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    days_past: int = 365
    source_name: str = "youtube"


class YouTubeConnector:
    """Parses YouTube watch history and yields per-month ParsedDocuments.

    Args:
        config: Export path, date range, and source settings.
    """

    def __init__(self, config: YouTubeConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Locate ``watch-history.json`` and yield per-month documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            calendar month, with ``file_type=".youtube"`` and metadata
            containing ``year_month``, ``video_count``, and ``channels``
            (up to 10 most frequent channel names).
        """
        export_path = self._cfg.export_path.expanduser()

        if export_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(export_path) as zf:
                        zf.extractall(tmp)
                except Exception as exc:
                    logger.warning(f"YouTubeConnector: cannot open zip {export_path}: {exc}")
                    return
                yield from self._process_dir(Path(tmp))
        elif export_path.is_dir():
            yield from self._process_dir(export_path)
        elif export_path.is_file():
            yield from self._parse_history(export_path)
        else:
            logger.warning(f"YouTubeConnector: path not found: {export_path}")

    def _process_dir(self, base: Path) -> Iterator[ParsedDocument]:
        """Search *base* recursively for ``watch-history.json``.

        Args:
            base: Root directory to search.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        candidates = list(base.rglob("watch-history.json"))
        if not candidates:
            logger.warning(f"YouTubeConnector: no watch-history.json found in {base}")
            return
        yield from self._parse_history(candidates[0])

    def _cutoff_dt(self) -> datetime | None:
        """Return the UTC cutoff datetime, or ``None`` if no cutoff.

        Returns:
            Aware :class:`datetime` or ``None``.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _parse_history(self, path: Path) -> Iterator[ParsedDocument]:
        """Parse ``watch-history.json`` and yield monthly documents.

        Args:
            path: Path to the JSON file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per month.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"YouTubeConnector: cannot read {path}: {exc}")
            return
        if not isinstance(data, list):
            logger.warning(f"YouTubeConnector: unexpected format in {path}")
            return

        cutoff = self._cutoff_dt()
        # year_month -> list of (date_str, title, channel)
        by_month: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)

        for entry in data:
            title = entry.get("title", "")
            if title.startswith("Watched a video that has been removed"):
                continue
            subtitles = entry.get("subtitles") or []
            if not subtitles:
                continue
            channel = subtitles[0].get("name", "Unknown")
            time_str = entry.get("time", "")
            if not time_str:
                continue
            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if cutoff is not None and dt < cutoff:
                continue
            year_month = dt.strftime("%Y-%m")
            date_str = dt.strftime("%Y-%m-%d")
            by_month[year_month].append((date_str, title, channel))

        source = self._cfg.source_name
        total_docs = 0
        for year_month, entries in sorted(by_month.items()):
            entries_sorted = sorted(entries, key=lambda e: e[0])
            lines = [f"{d} - {t} by {ch}" for d, t, ch in entries_sorted]
            channel_counts: Counter[str] = Counter(ch for _, _, ch in entries_sorted)
            top_channels = [ch for ch, _ in channel_counts.most_common(10)]
            yield ParsedDocument(
                path=Path(f"youtube://{year_month}.youtube"),
                text="\n".join(lines),
                file_type=".youtube",
                encoding="utf-8",
                metadata={
                    "year_month": year_month,
                    "video_count": len(entries_sorted),
                    "channels": top_channels,
                    "source": source,
                },
            )
            total_docs += 1

        logger.info(f"YouTubeConnector: yielded {total_docs} document(s)")


# Backward-compatible alias
YouTubeTakeoutConnector = YouTubeConnector
