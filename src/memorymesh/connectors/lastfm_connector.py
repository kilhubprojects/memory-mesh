"""Last.fm scrobble history connector for MemoryMesh.

Fetches a user's listening history from the Last.fm API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per calendar month.

API reference
-------------
``https://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks``

Features
--------
* **Monthly grouping** - tracks are grouped by calendar month, giving
  natural temporal granularity for search.
* **Pagination** - iterates via ``totalPages`` until all pages are
  fetched.
* **Currently-playing skip** - tracks with ``@attr.nowplaying = "true"``
  have no timestamp and are skipped automatically.
* **Top artists** - up to 5 most-scrobbled artists per month are stored
  in document metadata.
* **Free API key** - obtain one at https://www.last.fm/api/account/create

Usage
-----
::

    connector = LastFmConnector(LastFmConfig(
        api_key=SecretStr("your_api_key"),
        username="my_lastfm_user",
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_API_BASE = "https://ws.audioscrobbler.com/2.0/"


class LastFmConfig(BaseModel):
    """Configuration for a Last.fm scrobble history source.

    Args:
        api_key: Last.fm API key.  Obtain a free key at
            https://www.last.fm/api/account/create.
        username: Last.fm username to fetch scrobbles for.
        days_past: How many days of history to fetch.  0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    username: str
    days_past: int = 90
    source_name: str = "lastfm"


class LastFmConnector:
    """Fetches Last.fm scrobbles and yields per-month ParsedDocuments.

    Args:
        config: API credentials, username, and date range.
    """

    def __init__(self, config: LastFmConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch all scrobbles within ``days_past`` and yield monthly docs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            calendar month, with ``file_type=".lastfm"`` and metadata
            containing ``username``, ``year_month``, ``track_count``,
            and ``top_artists`` (up to 5).
        """
        now = datetime.now(tz=UTC)
        from_ts = (
            int((now - timedelta(days=self._cfg.days_past)).timestamp())
            if self._cfg.days_past > 0
            else 0
        )
        to_ts = int(now.timestamp())

        # year_month -> list of (unix_ts, datetime_str, artist, track, album)
        by_month: defaultdict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)

        page = 1
        while True:
            params: dict[str, Any] = {
                "method": "user.getrecenttracks",
                "user": self._cfg.username,
                "api_key": self._cfg.api_key.get_secret_value(),
                "format": "json",
                "limit": 200,
                "page": page,
                "to": to_ts,
            }
            if from_ts:
                params["from"] = from_ts

            url = _API_BASE + "?" + urllib.parse.urlencode(params)
            data = api_get(url, {})
            if not isinstance(data, dict):
                break

            rt = data.get("recenttracks", {})
            tracks = rt.get("track", [])
            if not isinstance(tracks, list):
                tracks = [tracks]

            attr = rt.get("@attr", {})
            total_pages = int(attr.get("totalPages", 1))

            for track in tracks:
                track_attr = track.get("@attr")
                if isinstance(track_attr, dict) and track_attr.get("nowplaying") == "true":
                    continue
                date_obj = track.get("date")
                if not date_obj:
                    continue
                uts = int(date_obj.get("uts", 0))
                dt = datetime.fromtimestamp(uts, tz=UTC)
                year_month = dt.strftime("%Y-%m")
                dt_str = dt.strftime("%Y-%m-%d %H:%M")
                name = track.get("name", "")
                artist_obj = track.get("artist", {})
                artist = artist_obj.get("#text", "") if isinstance(artist_obj, dict) else ""
                album_obj = track.get("album", {})
                album = album_obj.get("#text", "") if isinstance(album_obj, dict) else ""
                by_month[year_month].append((uts, dt_str, artist, name, album))

            if page >= total_pages:
                break
            page += 1

        source = self._cfg.source_name
        username = self._cfg.username
        total_docs = 0

        for year_month, entries in sorted(by_month.items()):
            entries_sorted = sorted(entries, key=lambda e: e[0])
            lines = [
                f"{dt_str} - {artist} - {track} [{album}]"
                for _, dt_str, artist, track, album in entries_sorted
            ]
            artist_counts: Counter[str] = Counter(artist for _, _, artist, _, _ in entries_sorted)
            top_artists = [a for a, _ in artist_counts.most_common(5)]

            yield ParsedDocument(
                path=Path(f"lastfm://{username}/{year_month}.lastfm"),
                text="\n".join(lines),
                file_type=".lastfm",
                encoding="utf-8",
                metadata={
                    "username": username,
                    "year_month": year_month,
                    "track_count": len(entries_sorted),
                    "top_artists": top_artists,
                    "source": source,
                },
            )
            total_docs += 1

        logger.info(f"LastFmConnector: yielded {total_docs} month document(s)")
