"""Chess.com connector for MemoryMesh.

Fetches game archives from Chess.com via the public API (no auth required)
and yields one :class:`~memorymesh.core.models.ParsedDocument` per
calendar month of games.

API reference
-------------
``https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}``

Authentication
--------------
None required - Chess.com's public API is open.  A ``User-Agent`` header
is recommended for politeness.

Features
--------
* **Monthly iteration** - fetches all available months from the archive
  list and processes each in reverse-chronological order.
* **PGN header parsing** - extracts game metadata (Event, Site, Date,
  White, Black, Result, TimeControl, ECO, Opening) from PGN headers using
  a regex match.
* **Date filtering** - months fully before the ``days_past`` cutoff are
  skipped.

Usage
-----
::

    connector = ChessComConnector(ChessComConfig(
        username="myhandle",
        days_past=365,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://api.chess.com/pub"
_USER_AGENT = "MemoryMesh/0.7 (+https://github.com/memorymesh)"
_HEADERS = {"User-Agent": _USER_AGENT}
_PGN_HEADER_RE = re.compile(r'\[(\w+)\s+"([^"]+)"\]')


class ChessComConfig(BaseModel):
    """Configuration for a Chess.com source.

    Args:
        username: Chess.com username (case-insensitive).
        days_past: Only include months with games within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    username: str
    days_past: int = 365
    source_name: str = "chesscom"


class ChessComConnector:
    """Fetches Chess.com game archives grouped by month.

    Each document represents one calendar month of games for the user.

    Args:
        config: Username, date filter, and source settings.
    """

    def __init__(self, config: ChessComConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Iterate monthly archives and yield one document per month.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            month, with ``file_type=".chess"`` and metadata containing
            ``username``, ``year``, ``month``, and ``game_count``.
        """
        username = self._cfg.username.lower()
        archives_url = f"{_BASE}/player/{username}/games/archives"
        data = api_get(archives_url, _HEADERS)
        if not isinstance(data, dict):
            logger.warning("ChessComConnector: failed to fetch archives list")
            return

        archive_urls: list[str] = data.get("archives", [])
        if not archive_urls:
            return

        cutoff = self._cutoff()
        total = 0

        for url in reversed(archive_urls):
            parts = url.rstrip("/").rsplit("/", 2)
            if len(parts) < 3:
                continue
            try:
                year = int(parts[-2])
                month = int(parts[-1])
            except ValueError:
                continue

            if cutoff:
                import calendar

                last_day = calendar.monthrange(year, month)[1]
                month_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=UTC)
                if month_end < cutoff:
                    break

            doc = self._fetch_month(username, url, year, month)
            if doc is not None:
                yield doc
                total += 1

        logger.info(f"ChessComConnector: yielded {total} monthly archive(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _parse_pgn_headers(self, pgn: str) -> dict[str, str]:
        """Extract PGN tag headers from a PGN string.

        Args:
            pgn: Raw PGN text for a single game.

        Returns:
            Dict mapping tag names to their values.
        """
        return dict(_PGN_HEADER_RE.findall(pgn))

    def _fetch_month(
        self,
        username: str,
        url: str,
        year: int,
        month: int,
    ) -> ParsedDocument | None:
        """Fetch and parse games for one calendar month.

        Args:
            username: Chess.com username.
            url: Archive URL for this month.
            year: Calendar year.
            month: Calendar month (1-12).

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` or ``None``
            if no games are found.
        """
        data = api_get(url, _HEADERS)
        if not isinstance(data, dict):
            return None

        games: list[dict[str, Any]] = data.get("games", [])
        if not games:
            return None

        lines: list[str] = [
            f"Chess.com games for {username}: {year}-{month:02d}",
            f"Total games: {len(games)}",
        ]
        for game in games:
            pgn = game.get("pgn", "")
            if pgn:
                headers = self._parse_pgn_headers(pgn)
                white = headers.get("White", "?")
                black = headers.get("Black", "?")
                result = headers.get("Result", "?")
                opening = headers.get("Opening", "")
                time_ctrl = headers.get("TimeControl", "")
                line = f"{white} vs {black} - {result}"
                if opening:
                    line += f" ({opening})"
                if time_ctrl:
                    line += f" [{time_ctrl}]"
                lines.append(line)

        return ParsedDocument(
            path=Path(f"chess://{username}/{year}-{month:02d}.chess"),
            text="\n".join(lines),
            file_type=".chess",
            encoding="utf-8",
            metadata={
                "username": username,
                "year": year,
                "month": month,
                "game_count": len(games),
                "source": self._cfg.source_name,
            },
        )
