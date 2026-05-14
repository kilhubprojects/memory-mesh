"""Slack Web API connector for MemoryMesh.

Fetches messages from Slack channels and yields one
:class:`~memorymesh.core.models.ParsedDocument` per channel-day pair.

API reference: https://api.slack.com/methods

Features
--------
* **Channel auto-discovery** - if no ``channel_ids`` are configured, the
  connector lists all public channels the bot can see via
  ``conversations.list``.
* **Daily batching** - messages are grouped into one document per
  (channel, date) pair, giving natural temporal granularity.
* **User caching** - display names are resolved once per user ID and cached
  for the session.
* **Rate-limit aware** - 1.2 s sleep between channel history calls (Slack
  Tier 3 ~= 50 req/min).
* **Pagination** - all list/history calls paginate via ``next_cursor``.
* **Privacy** - message text is never logged at INFO level.

Bot token scopes required
-------------------------
``channels:history``, ``channels:read``, ``users:read``

Usage
-----
::

    connector = SlackConnector(SlackConfig(
        token=SecretStr("xoxb-..."),
        days_past=30,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://slack.com/api"
_HISTORY_SLEEP = 1.2  # seconds between conversations.history calls


class SlackConfig(BaseModel):
    """Configuration for a Slack workspace source.

    Args:
        token: Slack Bot OAuth token (``xoxb-...``).  The bot must have the
            ``channels:history``, ``channels:read``, and ``users:read``
            scopes.
        channel_ids: Explicit list of channel IDs to index.  An empty list
            causes the connector to discover all accessible public channels
            automatically via ``conversations.list``.
        days_past: How many days of history to fetch (0 = no cutoff).
        max_messages_per_channel: Maximum messages fetched per channel per
            run.  0 means no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    token: SecretStr
    channel_ids: list[str] = []
    days_past: int = 30
    max_messages_per_channel: int = 1_000
    source_name: str = "slack"


def _ts_to_date(ts: str) -> str:
    """Convert a Slack message timestamp to a ``YYYY-MM-DD`` string (UTC).

    Args:
        ts: Slack timestamp string (e.g. ``"1609459200.000001"``).

    Returns:
        ISO date string in UTC, or ``"unknown"`` on parse error.
    """
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return "unknown"


def _ts_to_time(ts: str) -> str:
    """Convert a Slack message timestamp to a ``HH:MM`` string (UTC).

    Args:
        ts: Slack timestamp string.

    Returns:
        Time string (UTC), or ``"00:00"`` on parse error.
    """
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).strftime("%H:%M")
    except (ValueError, OSError):
        return "00:00"


class SlackConnector:
    """Fetches Slack messages and yields per-channel-day ParsedDocuments.

    Args:
        config: Slack API token and source settings.
    """

    def __init__(self, config: SlackConfig) -> None:
        self._cfg = config
        self._headers = bearer_header(config.token.get_secret_value())
        self._user_cache: dict[str, str] = {}

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Discover channels, fetch history, and yield per-day documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            (channel, date), with ``file_type=".slack"`` and metadata
            containing ``channel_id``, ``channel_name``, ``date``, and
            ``message_count``.
        """
        channel_ids = self._cfg.channel_ids

        if channel_ids:
            # Use provided IDs with placeholder names; resolved later
            channels = [{"id": cid, "name": cid} for cid in channel_ids]
        else:
            channels = list(self._list_channels())

        if not channels:
            logger.warning("SlackConnector: no channels found")
            return

        cutoff_ts = ""
        if self._cfg.days_past > 0:
            cutoff_ts = str(time.time() - self._cfg.days_past * 86_400)

        total_docs = 0
        for channel in channels:
            channel_id = channel.get("id", "")
            channel_name = channel.get("name", channel_id)
            if not channel_id:
                continue

            time.sleep(_HISTORY_SLEEP)
            messages = list(self._get_history(channel_id, cutoff_ts))

            if not messages:
                continue

            for doc in self._group_by_day(messages, channel_id, channel_name):
                yield doc
                total_docs += 1

        logger.info(f"SlackConnector: yielded {total_docs} document(s)")

    def _list_channels(self) -> Iterator[dict[str, Any]]:
        """Paginate through all public channels the bot can access.

        Yields:
            Channel dict (``id``, ``name``, etc.).
        """
        cursor: str = ""

        while True:
            url = f"{_BASE}/conversations.list?types=public_channel&limit=200"
            if cursor:
                url += f"&cursor={cursor}"
            data = api_get(url, self._headers)
            if not data or not data.get("ok"):
                logger.warning(
                    f"SlackConnector: conversations.list failed:"
                    f" {data.get('error') if data else 'no response'}"
                )
                break
            yield from data.get("channels", [])
            cursor = data.get("response_metadata", {}).get("next_cursor", "") or ""
            if not cursor:
                break

    def _get_history(
        self,
        channel_id: str,
        oldest: str,
    ) -> Iterator[dict[str, Any]]:
        """Paginate through message history for one channel.

        Skips subtypes (channel_join, bot_message, etc.) and empty messages.

        Args:
            channel_id: Slack channel ID.
            oldest: Earliest message Unix timestamp as string.  Empty string
                means no cutoff.

        Yields:
            Message dict (``text``, ``ts``, ``user``, etc.).
        """
        cursor: str = ""
        yielded = 0
        limit = self._cfg.max_messages_per_channel

        while True:
            url = f"{_BASE}/conversations.history?channel={channel_id}&limit=200"
            if oldest:
                url += f"&oldest={oldest}"
            if cursor:
                url += f"&cursor={cursor}"

            data = api_get(url, self._headers)
            if not data or not data.get("ok"):
                logger.warning(
                    f"SlackConnector: conversations.history failed for"
                    f" {channel_id}:"
                    f" {data.get('error') if data else 'no response'}"
                )
                break

            for msg in data.get("messages", []):
                if limit > 0 and yielded >= limit:
                    return
                # Skip subtypes (joins, leaves, bot messages without text, etc.)
                if msg.get("subtype"):
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                yield msg
                yielded += 1

            if not data.get("has_more"):
                break
            cursor = data.get("response_metadata", {}).get("next_cursor", "") or ""
            if not cursor:
                break

    def _get_username(self, user_id: str) -> str:
        """Resolve a Slack user ID to a display name, caching the result.

        Args:
            user_id: Slack user ID (e.g. ``"U012AB3CD"``).

        Returns:
            Display name string, or the raw user ID as fallback.
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        data = api_get(f"{_BASE}/users.info?user={user_id}", self._headers)
        name = user_id
        if data and data.get("ok"):
            user = data.get("user", {})
            profile = user.get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("real_name")
                or user_id
            )
        self._user_cache[user_id] = name
        return name

    def _group_by_day(
        self,
        messages: list[dict[str, Any]],
        channel_id: str,
        channel_name: str,
    ) -> Iterator[ParsedDocument]:
        """Group messages by date and yield one ParsedDocument per day.

        Args:
            messages: List of Slack message dicts (in API order, newest first).
            channel_id: Slack channel ID.
            channel_name: Human-readable channel name.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per channel-day.
        """
        # Group messages by date (UTC)
        by_date: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for msg in messages:
            date = _ts_to_date(msg.get("ts", ""))
            by_date[date].append(msg)

        source = self._cfg.source_name
        safe_name = channel_name.replace("/", "_").replace(" ", "_")

        for date in sorted(by_date):
            day_messages = sorted(by_date[date], key=lambda m: m.get("ts", ""))
            lines: list[str] = []
            for msg in day_messages:
                user_id = msg.get("user", "")
                username = self._get_username(user_id) if user_id else "unknown"
                time_str = _ts_to_time(msg.get("ts", ""))
                text = (msg.get("text") or "").strip()
                lines.append(f"{time_str} {username}: {text}")

            yield ParsedDocument(
                path=Path(f"slack://{safe_name}/{date}.slack"),
                text="\n".join(lines),
                file_type=".slack",
                encoding="utf-8",
                metadata={
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "date": date,
                    "message_count": len(lines),
                    "source": source,
                },
            )
