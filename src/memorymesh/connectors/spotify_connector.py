"""Spotify extended streaming history connector for MemoryMesh.

Reads the extended streaming history JSON files downloaded from Spotify
(Account -> Privacy -> Download your data -> Extended streaming history) and
yields one :class:`~memorymesh.core.models.ParsedDocument` per calendar
month.

Features
--------
* **Multi-file** - globs all ``Streaming_History_Audio_*.json`` files in
  the configured directory.
* **Duration filter** - tracks played for less than ``min_ms_played``
  milliseconds are skipped (default: 30 000 ms = 30 s).
* **Monthly grouping** - plays are aggregated into one document per month,
  giving a natural temporal granularity without creating thousands of tiny
  docs.
* **Privacy** - track names and artists are never logged at INFO level;
  only counts and dates are written.
* **Stdlib only** - no third-party dependencies.

Usage
-----
::

    connector = SpotifyConnector(SpotifyConfig(
        history_path=Path("~/Downloads/spotify/"),
        min_ms_played=30_000,
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


class SpotifyConfig(BaseModel):
    """Configuration for a Spotify streaming history source.

    Args:
        history_path: Directory containing ``Streaming_History_Audio_*.json``
            files downloaded from Spotify's privacy settings.
        min_ms_played: Minimum playback duration in milliseconds.  Tracks
            played for less than this threshold are skipped.  Defaults to
            30 000 ms (30 s) to exclude accidental plays.
        source_name: Name used in the MemoryMesh source registry.
    """

    history_path: Path
    min_ms_played: int = 30_000
    source_name: str = "spotify"


class SpotifyConnector:
    """Reads Spotify extended streaming history and yields monthly summaries.

    Args:
        config: History directory path and filter settings.
    """

    def __init__(self, config: SpotifyConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse streaming history files and yield one document per month.

        All files matching ``Streaming_History_Audio_*.json`` are read,
        filtered by ``min_ms_played``, and grouped by ISO year-month
        (``YYYY-MM``).  One :class:`~memorymesh.core.models.ParsedDocument`
        is emitted for each month that contains at least one qualifying play.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            calendar month, with ``file_type=".spotify"`` and metadata
            containing ``year_month``, ``track_count``, and ``total_hours``.
        """
        p = self._cfg.history_path
        if not p.is_dir():
            logger.warning(f"SpotifyConnector: directory not found: {p}")
            return

        files = sorted(p.glob("Streaming_History_Audio_*.json"))
        if not files:
            logger.warning(f"SpotifyConnector: no Streaming_History_Audio_*.json in {p}")
            return

        logger.info(f"SpotifyConnector: found {len(files)} history file(s)")

        # month (YYYY-MM) -> list of formatted play lines
        months: defaultdict[str, list[str]] = defaultdict(list)
        months_ms: defaultdict[str, int] = defaultdict(int)
        total_raw = 0
        total_kept = 0

        for f in files:
            try:
                records = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"SpotifyConnector: cannot parse {f.name}: {exc}")
                continue

            if not isinstance(records, list):
                continue

            for rec in records:
                total_raw += 1
                ms_played = int(rec.get("ms_played", 0) or 0)
                if ms_played < self._cfg.min_ms_played:
                    continue

                ts = str(rec.get("ts", ""))
                year_month = ts[:7] if len(ts) >= 7 else "unknown"
                track = str(rec.get("master_metadata_track_name", "") or "")
                artist = str(rec.get("master_metadata_album_artist_name", "") or "")
                album = str(rec.get("master_metadata_album_album_name", "") or "")
                secs = ms_played // 1_000

                line = f"{ts} - {artist} - {track} ({album}) [{secs}s]"
                months[year_month].append(line)
                months_ms[year_month] += ms_played
                total_kept += 1

        logger.info(
            f"SpotifyConnector: kept {total_kept}/{total_raw} plays across {len(months)} month(s)"
        )

        source = self._cfg.source_name
        for year_month in sorted(months):
            lines = months[year_month]
            total_hours = round(months_ms[year_month] / 3_600_000, 2)

            yield ParsedDocument(
                path=Path(f"spotify://{year_month}.spotify"),
                text="\n".join(lines),
                file_type=".spotify",
                encoding="utf-8",
                metadata={
                    "year_month": year_month,
                    "track_count": len(lines),
                    "total_hours": total_hours,
                    "source": source,
                },
            )


# Backward-compatible alias
SpotifyHistoryConnector = SpotifyConnector
