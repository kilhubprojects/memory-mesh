"""Steam Web API connector for MemoryMesh.

Fetches a user's game library and playtime from the Steam Web API and
yields one :class:`~memorymesh.core.models.ParsedDocument` per owned
game plus a summary document for recently played games.

API reference
-------------
``https://api.steampowered.com/``

Endpoints used
--------------
* ``IPlayerService/GetOwnedGames/v1/`` - full library with app info.
* ``ISteamUserStats/GetPlayerAchievements/v1/`` - optional, per-game
  achievement stats (only fetched for games with >= 1 h played).
* ``IPlayerService/GetRecentlyPlayedGames/v1/`` - up to 50 games played
  in the last two weeks (summary document).

Features
--------
* **Playtime filter** - games below ``min_playtime_hours`` are skipped.
* **Achievements** - optional; 1 s sleep between calls to respect Steam
  rate limits.  Only fetched for games with >= 1 h of playtime.
* **Recent-games summary** - always fetched; yields one extra document
  at ``steam://recent.steam`` with last-2-weeks hours per game.
* **No extra packages** - stdlib ``urllib`` via the shared helper.

Usage
-----
::

    connector = SteamConnector(SteamConfig(
        api_key=SecretStr("your_steam_api_key"),
        steam_id="76561198012345678",
        min_playtime_hours=0.5,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_API_BASE = "https://api.steampowered.com"
_ACH_SLEEP = 1.0  # seconds between achievement API calls


class SteamConfig(BaseModel):
    """Configuration for a Steam game library source.

    Args:
        api_key: Steam Web API key.  Obtain at
            https://steamcommunity.com/dev/apikey.
        steam_id: 64-bit SteamID of the profile to fetch.
        include_achievements: When ``True``, fetch achievement counts for
            each game with >= 1 h of playtime and append to the document
            text.  Adds one API call per qualifying game.
        min_playtime_hours: Skip games with less than this many hours of
            total playtime.  Default 0.1 h filters out brief trials.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    steam_id: str
    include_achievements: bool = False
    min_playtime_hours: float = 0.1
    source_name: str = "steam"


class SteamConnector:
    """Fetches Steam library and yields one ParsedDocument per game.

    Args:
        config: Steam API key, Steam ID, and filter settings.
    """

    def __init__(self, config: SteamConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch library, optional achievements, recent games, yield docs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            owned game above the playtime threshold
            (``file_type=".steam"``) plus one summary document at
            ``steam://recent.steam``.
        """
        key = self._cfg.api_key.get_secret_value()
        steam_id = self._cfg.steam_id
        min_minutes = int(self._cfg.min_playtime_hours * 60)
        source = self._cfg.source_name
        total = 0

        lib_url = (
            f"{_API_BASE}/IPlayerService/GetOwnedGames/v1/"
            f"?key={key}&steamid={steam_id}"
            f"&include_appinfo=true&include_played_free_games=true"
        )
        lib_data = api_get(lib_url, {})
        games: list[dict[str, Any]] = []
        if isinstance(lib_data, dict):
            games = lib_data.get("response", {}).get("games", []) or []

        for game in games:
            playtime = int(game.get("playtime_forever", 0) or 0)
            if playtime < min_minutes:
                continue
            doc = self._build_game_doc(game, key, steam_id)
            if doc is not None:
                yield doc
                total += 1

        recent_url = (
            f"{_API_BASE}/IPlayerService/GetRecentlyPlayedGames/v1/"
            f"?key={key}&steamid={steam_id}&count=50"
        )
        recent_data = api_get(recent_url, {})
        if isinstance(recent_data, dict):
            recent_games = recent_data.get("response", {}).get("games", []) or []
            if recent_games:
                yield self._build_recent_doc(recent_games, source)
                total += 1

        logger.info(f"SteamConnector: yielded {total} document(s)")

    def _build_game_doc(
        self,
        game: dict[str, Any],
        key: str,
        steam_id: str,
    ) -> ParsedDocument | None:
        """Build a ParsedDocument for one owned game.

        Args:
            game: Raw Steam game object from ``GetOwnedGames``.
            key: Steam API key (used for optional achievement lookup).
            steam_id: Steam ID (used for optional achievement lookup).

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the ``appid`` field is missing.
        """
        app_id = game.get("appid")
        if not app_id:
            return None

        name = game.get("name", "Unknown Game")
        playtime_min = int(game.get("playtime_forever", 0) or 0)
        hours = playtime_min / 60
        last_played_ts = int(game.get("rtime_last_played", 0) or 0)
        last_played = (
            datetime.fromtimestamp(last_played_ts, tz=UTC).strftime("%Y-%m-%d")
            if last_played_ts
            else "Never"
        )

        text_parts = [
            f"Game: {name}",
            f"Total playtime: {hours:.1f}h",
            f"Last played: {last_played}",
        ]

        if self._cfg.include_achievements and hours >= 1.0:
            ach_text = self._fetch_achievements(key, steam_id, app_id)
            if ach_text:
                text_parts.append(ach_text)
            time.sleep(_ACH_SLEEP)

        return ParsedDocument(
            path=Path(f"steam://{app_id}.steam"),
            text="\n".join(text_parts),
            file_type=".steam",
            encoding="utf-8",
            metadata={
                "app_id": app_id,
                "name": name,
                "playtime_hours": round(hours, 1),
                "last_played": last_played,
                "source": self._cfg.source_name,
            },
        )

    def _fetch_achievements(
        self,
        key: str,
        steam_id: str,
        app_id: int,
    ) -> str | None:
        """Fetch achievement counts for one game.

        Args:
            key: Steam API key.
            steam_id: Steam ID.
            app_id: Steam application ID.

        Returns:
            ``"Achievements: {unlocked}/{total}"`` string, or ``None``
            if the API returns no achievement data.
        """
        url = (
            f"{_API_BASE}/ISteamUserStats/GetPlayerAchievements/v1/"
            f"?key={key}&steamid={steam_id}&appid={app_id}"
        )
        data = api_get(url, {})
        if not isinstance(data, dict):
            return None
        achievements = data.get("playerstats", {}).get("achievements", []) or []
        if not achievements:
            return None
        unlocked = sum(1 for a in achievements if a.get("achieved"))
        return f"Achievements: {unlocked}/{len(achievements)}"

    def _build_recent_doc(
        self,
        recent_games: list[dict[str, Any]],
        source: str,
    ) -> ParsedDocument:
        """Build a summary ParsedDocument for recently played games.

        Args:
            recent_games: List of game objects from
                ``GetRecentlyPlayedGames``.
            source: Source name for metadata.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        lines: list[str] = []
        for game in recent_games:
            name = game.get("name", "Unknown")
            hours_2w = int(game.get("playtime_2weeks", 0) or 0) / 60
            lines.append(f"{name}: {hours_2w:.1f}h (last 2 weeks)")

        return ParsedDocument(
            path=Path("steam://recent.steam"),
            text=(f"Recently Played Games ({len(recent_games)})\n\n" + "\n".join(lines)),
            file_type=".steam",
            encoding="utf-8",
            metadata={
                "type": "recent_games",
                "game_count": len(recent_games),
                "source": source,
            },
        )
