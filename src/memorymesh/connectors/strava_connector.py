"""Strava API connector for MemoryMesh.

Fetches cycling, running, and other activities from the Strava API and
yields one :class:`~memorymesh.core.models.ParsedDocument` per activity.

API reference: https://developers.strava.com/docs/reference/

Authentication
--------------
Strava uses OAuth 2.0 with short-lived access tokens (6-hour TTL).  On
each run the connector automatically exchanges the stored ``refresh_token``
for a fresh ``access_token`` via POST ``/oauth/token``.

Features
--------
* **Token auto-refresh** - no manual token rotation needed.
* **Pagination** - iterates ``?page={n}&per_page=100`` until the API
  returns an empty list.
* **Date filter** - only activities after the ``cutoff_epoch`` (derived
  from ``days_past``) are fetched.
* **No extra packages** - uses only stdlib ``urllib``.

Usage
-----
::

    connector = StravaConnector(StravaConfig(
        client_id="12345",
        client_secret=SecretStr("abc..."),
        refresh_token=SecretStr("xyz..."),
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.core.models import ParsedDocument

_API_BASE = "https://www.strava.com/api/v3"
_TOKEN_URL = "https://www.strava.com/oauth/token"


class StravaConfig(BaseModel):
    """Configuration for a Strava activity source.

    Args:
        client_id: Strava OAuth application client ID.
        client_secret: Strava OAuth application client secret.
        refresh_token: Long-lived refresh token obtained during the initial
            OAuth authorization flow.
        days_past: Only fetch activities within this many days.  0 means
            no cutoff.
        max_activities: Maximum number of activities to fetch per run.
            0 means no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    client_id: str
    client_secret: SecretStr
    refresh_token: SecretStr
    days_past: int = 365
    max_activities: int = 500
    source_name: str = "strava"


class StravaConnector:
    """Fetches Strava activities and yields one ParsedDocument per activity.

    Args:
        config: Strava credentials and filter settings.
    """

    def __init__(self, config: StravaConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Refresh the access token, fetch all activities, and yield docs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            activity, with ``file_type=".strava"`` and metadata containing
            ``activity_id``, ``type``, ``date``, ``distance_km``,
            ``duration_min``, ``elevation_m``, and ``avg_hr``.
        """
        access_token = self._refresh_token()
        if not access_token:
            logger.warning("StravaConnector: token refresh failed - aborting")
            return

        headers = bearer_header(access_token)
        cutoff_epoch = self._cutoff_epoch()
        yielded = 0
        page = 1
        limit = self._cfg.max_activities

        while True:
            if limit > 0 and yielded >= limit:
                break

            url = f"{_API_BASE}/athlete/activities?per_page=100&page={page}"
            if cutoff_epoch:
                url += f"&after={cutoff_epoch}"

            data = self._get(url, headers)
            if not isinstance(data, list) or not data:
                break

            for activity in data:
                if limit > 0 and yielded >= limit:
                    break
                doc = self._build_document(activity)
                if doc is not None:
                    yield doc
                    yielded += 1

            page += 1

        logger.info(f"StravaConnector: yielded {yielded} activity/activities")

    def _refresh_token(self) -> str | None:
        """Exchange the refresh token for a fresh access token.

        Returns:
            New access token string, or ``None`` on failure.
        """
        payload = urllib.parse.urlencode(
            {
                "client_id": self._cfg.client_id,
                "client_secret": self._cfg.client_secret.get_secret_value(),
                "refresh_token": self._cfg.refresh_token.get_secret_value(),
                "grant_type": "refresh_token",
            }
        ).encode()

        req = urllib.request.Request(
            _TOKEN_URL,
            data=payload,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("access_token")
        except urllib.error.HTTPError as exc:
            logger.warning(f"StravaConnector: token refresh HTTP {exc.code}: {exc.reason}")
            return None
        except Exception as exc:
            logger.warning(f"StravaConnector: token refresh failed: {exc}")
            return None

    def _cutoff_epoch(self) -> int | None:
        """Return the Unix epoch cutoff for activity date filtering.

        Returns:
            Integer epoch seconds, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        cutoff = datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)
        return int(cutoff.timestamp())

    def _get(self, url: str, headers: dict[str, str]) -> Any:
        """HTTP GET with JSON parsing.

        Args:
            url: Full request URL.
            headers: HTTP request headers.

        Returns:
            Parsed JSON value or ``None`` on error.
        """
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            logger.warning(f"StravaConnector: HTTP {exc.code} GET {url}: {exc.reason}")
            return None
        except Exception as exc:
            logger.warning(f"StravaConnector: GET failed {url}: {exc}")
            return None

    def _build_document(self, activity: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Strava activity dict to a ParsedDocument.

        Args:
            activity: Raw Strava activity JSON object.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if essential fields are missing.
        """
        activity_id = activity.get("id")
        if not activity_id:
            return None

        name = activity.get("name", "Untitled")
        sport_type = activity.get("sport_type") or activity.get("type", "")
        start_date = activity.get("start_date_local", "")
        distance_km = float(activity.get("distance", 0) or 0) / 1000
        moving_time_s = int(activity.get("moving_time", 0) or 0)
        moving_time_min = moving_time_s / 60
        elevation_gain = float(activity.get("total_elevation_gain", 0) or 0)
        avg_hr_raw = activity.get("average_heartrate")
        avg_hr = f"{avg_hr_raw:.0f}" if avg_hr_raw is not None else "N/A"
        max_hr_raw = activity.get("max_heartrate")
        max_hr = f"{max_hr_raw:.0f}" if max_hr_raw is not None else "N/A"

        text = (
            f"Activity: {name}\n"
            f"Type: {sport_type}\n"
            f"Date: {start_date}\n"
            f"Distance: {distance_km:.1f} km\n"
            f"Time: {moving_time_min:.0f} min\n"
            f"Elevation: {elevation_gain:.0f} m\n"
            f"Avg HR: {avg_hr} bpm\n"
            f"Max HR: {max_hr} bpm"
        )

        return ParsedDocument(
            path=Path(f"strava://{activity_id}.strava"),
            text=text,
            file_type=".strava",
            encoding="utf-8",
            metadata={
                "activity_id": activity_id,
                "type": sport_type,
                "date": start_date,
                "distance_km": round(distance_km, 2),
                "duration_min": round(moving_time_min, 1),
                "elevation_m": round(elevation_gain, 1),
                "avg_hr": avg_hr,
                "source": self._cfg.source_name,
            },
        )
