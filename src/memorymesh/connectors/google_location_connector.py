"""Google Maps location history connector for MemoryMesh.

Parses location history exported from Google Takeout and yields
:class:`~memorymesh.core.models.ParsedDocument` objects from either raw
GPS point data (``Records.json``) or inferred place visits (Semantic
Location History).

Two export formats
------------------
* **Records.json** - raw GPS coordinates.  Points are grouped by UTC day;
  only aggregated summaries are emitted (point count, average accuracy) -
  no raw coordinates are stored.
* **Semantic Location History/** - monthly JSON files with inferred
  place visits.  One document per visit.

Features
--------
* **Auto-detection** - inspects the export path to choose the correct
  parser automatically.
* **Zip support** - if ``export_path`` is a ``.zip``, it is extracted to
  a temporary directory before parsing.
* **Privacy** - raw GPS coordinates are never included in document text.

Usage
-----
::

    connector = GoogleLocationConnector(GoogleLocationConfig(
        export_path=Path("~/Takeout/Location History"),
        days_past=180,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class GoogleLocationConfig(BaseModel):
    """Configuration for a Google Maps location history source.

    Args:
        export_path: Path to ``Records.json``, the ``Semantic Location
            History`` directory, or the Takeout ``.zip`` archive.
        days_past: Only include records within this many days.  0 means
            no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    days_past: int = 365
    source_name: str = "google_location"


class GoogleLocationConnector:
    """Parses Google Takeout location history exports.

    Args:
        config: Export path and date-range settings.
    """

    def __init__(self, config: GoogleLocationConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Auto-detect the export format and yield location ParsedDocuments.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per day
            (Records.json format) or one per place visit (Semantic format).
        """
        export_path = self._cfg.export_path.expanduser()

        if export_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(export_path) as zf:
                        zf.extractall(tmp)
                except Exception as exc:
                    logger.warning(f"GoogleLocationConnector: cannot open zip {export_path}: {exc}")
                    return
                yield from self._process_dir(Path(tmp))
        else:
            yield from self._process_dir(export_path)

    def _cutoff_ms(self) -> int | None:
        """Return the cutoff epoch timestamp in milliseconds.

        Returns:
            Millisecond timestamp or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        cutoff = datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)
        return int(cutoff.timestamp() * 1000)

    def _process_dir(self, base: Path) -> Iterator[ParsedDocument]:
        """Detect the export format under *base* and dispatch to the parser.

        Args:
            base: Root directory or path to ``Records.json``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        if base.is_file() and base.name == "Records.json":
            yield from self._parse_records(base)
            return

        records_file = base / "Records.json"
        if not records_file.exists():
            candidates = list(base.rglob("Records.json"))
            if candidates:
                records_file = candidates[0]

        if records_file.exists():
            yield from self._parse_records(records_file)
            return

        semantic_dirs = list(base.rglob("Semantic Location History"))
        semantic_dir: Path | None = semantic_dirs[0] if semantic_dirs else None
        if semantic_dir is None and base.is_dir() and any(base.glob("*.json")):
            semantic_dir = base

        if semantic_dir is not None:
            yield from self._parse_semantic(semantic_dir)
        else:
            logger.warning(f"GoogleLocationConnector: no supported location data found in {base}")

    def _parse_records(self, path: Path) -> Iterator[ParsedDocument]:
        """Parse a ``Records.json`` file (raw GPS format).

        Groups points by UTC day.  Only aggregated summaries are emitted;
        raw coordinates are not stored.

        Args:
            path: Path to ``Records.json``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        cutoff_ms = self._cutoff_ms()
        source = self._cfg.source_name

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"GoogleLocationConnector: cannot parse {path}: {exc}")
            return

        locations = data.get("locations", [])
        if not isinstance(locations, list):
            logger.warning(f"GoogleLocationConnector: unexpected Records.json structure in {path}")
            return

        # day -> list of accuracy values
        by_day: defaultdict[str, list[float]] = defaultdict(list)
        for loc in locations:
            ts_ms_raw = loc.get("timestampMs") or loc.get("timestamp", "")
            try:
                ts_ms = int(ts_ms_raw)
            except (ValueError, TypeError):
                continue
            if cutoff_ms is not None and ts_ms < cutoff_ms:
                continue
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            day = dt.strftime("%Y-%m-%d")
            accuracy = float(loc.get("accuracy", 0))
            by_day[day].append(accuracy)

        total_docs = 0
        for idx, (day, accuracies) in enumerate(sorted(by_day.items())):
            count = len(accuracies)
            valid_acc = [a for a in accuracies if a > 0]
            avg_acc = f"{sum(valid_acc) / len(valid_acc):.1f} m" if valid_acc else "N/A"
            text = (
                f"Location history - {day}\n"
                f"Points recorded: {count}\n"
                f"Average GPS accuracy: {avg_acc}"
            )
            yield ParsedDocument(
                path=Path(f"google_location://{day}/{idx}.location"),
                text=text,
                file_type=".location",
                encoding="utf-8",
                metadata={
                    "date": day,
                    "point_count": count,
                    "accuracy_avg": avg_acc,
                    "source": source,
                },
            )
            total_docs += 1

        logger.info(f"GoogleLocationConnector: {total_docs} day-document(s) from Records.json")

    def _parse_semantic(self, semantic_dir: Path) -> Iterator[ParsedDocument]:
        """Parse all monthly JSON files from a Semantic Location History dir.

        Args:
            semantic_dir: Path to the ``Semantic Location History``
                directory (or any directory containing ``*.json`` files in
                the semantic format).

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        cutoff_ms = self._cutoff_ms()
        source = self._cfg.source_name
        total_docs = 0

        for json_file in sorted(semantic_dir.rglob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"GoogleLocationConnector: cannot read {json_file}: {exc}")
                continue

            timeline = data.get("timelineObjects", [])
            if not isinstance(timeline, list):
                continue

            for obj in timeline:
                place_visit = obj.get("placeVisit")
                if not place_visit:
                    continue

                duration = place_visit.get("duration", {})
                start_ms_raw = duration.get("startTimestampMs", "0")
                end_ms_raw = duration.get("endTimestampMs", "0")
                try:
                    start_ms = int(start_ms_raw)
                    end_ms = int(end_ms_raw)
                except (ValueError, TypeError):
                    continue

                if cutoff_ms is not None and start_ms < cutoff_ms:
                    continue

                location = place_visit.get("location", {})
                name = location.get("name", "Unknown place")
                address = location.get("address", "")
                start_dt = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
                date_str = start_dt.strftime("%Y-%m-%d")
                duration_min = max(0, (end_ms - start_ms) // 60_000)

                text = (
                    f"Visited: {name}\n"
                    f"Address: {address}\n"
                    f"Date: {date_str}\n"
                    f"Duration: {duration_min} min"
                )
                yield ParsedDocument(
                    path=Path(f"google_location://{date_str}/{total_docs}.location"),
                    text=text,
                    file_type=".location",
                    encoding="utf-8",
                    metadata={
                        "place_name": name,
                        "address": address,
                        "date": date_str,
                        "duration_min": duration_min,
                        "source": source,
                    },
                )
                total_docs += 1

        logger.info(f"GoogleLocationConnector: {total_docs} place-visit document(s)")
