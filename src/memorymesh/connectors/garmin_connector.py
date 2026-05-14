"""Garmin Connect activity connector for MemoryMesh.

Reads activity data from a Garmin Connect data export (ZIP or directory)
and yields one :class:`~memorymesh.core.models.ParsedDocument` per
activity record.

Export format
-------------
Garmin Connect exports can be requested at
https://www.garmin.com/en-US/account/datamanagement/.
The export contains a ``DI_CONNECT/DI-Connect-Fitness/`` folder with
``summarizedActivities.json`` (bulk activity summaries) among other files.

Features
--------
* **ZIP or directory** - accepts either the raw export ZIP or an
  extracted directory.
* **Single activity documents** - one document per activity entry in
  ``summarizedActivities.json``.
* **Date filtering** - activities outside the ``days_past`` window are
  skipped.

Usage
-----
::

    connector = GarminConnector(GarminConfig(
        export_path=Path("GarminExport.zip"),
        days_past=365,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_ACTIVITIES_FILENAME = "summarizedActivities.json"
_SEARCH_GLOBS = [
    "**/summarizedActivities.json",
    "**/summarized_activities.json",
]


class GarminConfig(BaseModel):
    """Configuration for a Garmin Connect export source.

    Args:
        export_path: Path to the Garmin export ZIP or extracted directory.
        days_past: Only include activities within this many days.  0 = no
            cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    days_past: int = 365
    source_name: str = "garmin"


class GarminConnector:
    """Reads Garmin activity data and yields one ParsedDocument per activity.

    Args:
        config: Export path, date filter, and source settings.
    """

    def __init__(self, config: GarminConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the Garmin export and yield activity documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            activity, with ``file_type=".garmin"`` and metadata
            containing ``activity_id``, ``activity_type``, ``start_time``,
            ``duration_s``, ``distance_m``, and ``avg_hr``.
        """
        activities = self._load_activities()
        if not activities:
            return

        cutoff = self._cutoff()
        total = 0

        for activity in activities:
            if not isinstance(activity, dict):
                continue
            doc = self._build_doc(activity, cutoff)
            if doc is not None:
                yield doc
                total += 1

        logger.info(f"GarminConnector: yielded {total} activity records")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _load_activities(self) -> list[Any]:
        """Load the activities JSON from ZIP or directory.

        Returns:
            List of activity dicts, or empty list on failure.
        """
        path = self._cfg.export_path
        try:
            if path.is_file() and path.suffix.lower() == ".zip":
                return self._from_zip(path)
            if path.is_dir():
                return self._from_dir(path)
        except Exception as exc:
            logger.warning(f"GarminConnector: failed to load export: {exc}")
        return []

    def _from_zip(self, zip_path: Path) -> list[Any]:
        """Extract and parse the activities JSON from a ZIP file.

        Args:
            zip_path: Path to the Garmin export ZIP.

        Returns:
            List of activity dicts.
        """
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            target = next(
                (n for n in names if n.endswith(_ACTIVITIES_FILENAME)),
                None,
            )
            if not target:
                logger.warning("GarminConnector: no summarizedActivities.json in ZIP")
                return []
            with zf.open(target) as f:
                data = json.loads(f.read())
                return data if isinstance(data, list) else []

    def _from_dir(self, dir_path: Path) -> list[Any]:
        """Search a directory tree for the activities JSON file.

        Args:
            dir_path: Path to the extracted Garmin export directory.

        Returns:
            List of activity dicts.
        """
        for pattern in _SEARCH_GLOBS:
            matches = list(dir_path.glob(pattern))
            if matches:
                data = json.loads(matches[0].read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
        logger.warning("GarminConnector: summarizedActivities.json not found")
        return []

    def _build_doc(
        self,
        activity: dict[str, Any],
        cutoff: datetime | None,
    ) -> ParsedDocument | None:
        """Convert a Garmin activity record to a ParsedDocument.

        Args:
            activity: Raw Garmin activity summary dict.
            cutoff: UTC datetime cutoff; skip activities before this.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the activity should be skipped.
        """
        activity_id = str(activity.get("activityId") or activity.get("summaryId") or "")
        if not activity_id:
            return None

        start_time = activity.get("startTimeLocal") or activity.get("startTimeGMT") or ""
        if cutoff and start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                if dt < cutoff:
                    return None
            except ValueError as exc:
                logger.debug(f"GarminConnector: ignoring unparsable timestamp: {exc}")

        activity_type = activity.get("activityType") or activity.get("activityTypeKey") or "Unknown"
        if isinstance(activity_type, dict):
            activity_type = activity_type.get("typeKey", "Unknown")

        duration_s = float(activity.get("duration", 0) or 0)
        distance_m = float(activity.get("distance", 0) or 0)
        avg_hr = activity.get("averageHR") or activity.get("averageHeartRate")

        text_parts = [
            f"Activity: {activity_type}",
            f"Date: {start_time}",
            f"Duration: {duration_s:.0f}s",
            f"Distance: {distance_m:.0f}m",
        ]
        if avg_hr is not None:
            text_parts.append(f"Avg HR: {avg_hr} bpm")

        return ParsedDocument(
            path=Path(f"garmin://{activity_id}.garmin"),
            text="\n".join(text_parts),
            file_type=".garmin",
            encoding="utf-8",
            metadata={
                "activity_id": activity_id,
                "activity_type": activity_type,
                "start_time": start_time,
                "duration_s": duration_s,
                "distance_m": distance_m,
                "avg_hr": avg_hr,
                "source": self._cfg.source_name,
            },
        )
