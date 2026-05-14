"""Apple Health XML export connector for MemoryMesh.

Parses the Apple Health export (Health app -> Profile -> Export All Health
Data -> export.zip) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per record-type/month
pair plus one per workout activity.

Features
--------
* **Streaming XML parse** - uses ``xml.etree.ElementTree.iterparse`` so
  multi-GB export files never need to be fully loaded into memory.
* **Zip support** - if ``export_path`` points to a ``.zip``, the
  connector extracts ``apple_health_export/export.xml`` to a temporary
  directory before parsing.
* **Record-type filtering** - optionally restrict to specific
  ``HKQuantityTypeIdentifier*`` type strings.
* **Date filtering** - only records whose ``startDate`` falls within
  ``days_past`` days are included.
* **Workout support** - ``<Workout>`` elements are yielded as separate
  documents with ``file_type=".workout"``.

Usage
-----
::

    connector = AppleHealthConnector(AppleHealthConfig(
        export_path=Path("~/Health/export.zip"),
        record_types=["HKQuantityTypeIdentifierStepCount"],
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import contextlib
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_DATE_FMT = "%Y-%m-%d %H:%M:%S %z"


class AppleHealthConfig(BaseModel):
    """Configuration for an Apple Health export source.

    Args:
        export_path: Path to ``export.xml`` or the ``export.zip`` archive.
        record_types: Restrict to these HK record type identifiers.  An
            empty list means all record types are indexed.
        days_past: Only include records whose ``startDate`` is within this
            many days.  0 means no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    record_types: list[str] = []
    days_past: int = 365
    source_name: str = "apple_health"


def _parse_hk_date(s: str) -> datetime | None:
    """Parse an Apple Health date string to a timezone-aware datetime.

    Args:
        s: Date string in Apple Health format
            (``"YYYY-MM-DD HH:MM:SS +/-HHMM"``).

    Returns:
        Timezone-aware :class:`datetime`, or ``None`` on parse error.
    """
    try:
        return datetime.strptime(s, _DATE_FMT)
    except (ValueError, TypeError):
        return None


def _year_month(dt: datetime) -> str:
    """Return a ``YYYY-MM`` string from a datetime.

    Args:
        dt: Any datetime.

    Returns:
        Year-month string.
    """
    return dt.strftime("%Y-%m")


class AppleHealthConnector:
    """Parses Apple Health XML export and yields per-type-month documents.

    Args:
        config: Export path, optional record-type filter, and date range.
    """

    def __init__(self, config: AppleHealthConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Stream the XML export and yield ParsedDocuments.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            (record type, calendar month) for health records, and one per
            workout activity.
        """
        export_path = self._cfg.export_path.expanduser()

        if export_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(export_path) as zf:
                        zf.extractall(tmp)
                except Exception as exc:
                    logger.warning(f"AppleHealthConnector: cannot open zip {export_path}: {exc}")
                    return
                xml_path = Path(tmp) / "apple_health_export" / "export.xml"
                if not xml_path.exists():
                    xml_path = Path(tmp) / "export.xml"
                if not xml_path.exists():
                    logger.warning("AppleHealthConnector: export.xml not found in zip")
                    return
                yield from self._parse_xml(xml_path)
        else:
            yield from self._parse_xml(export_path)

    def _cutoff_dt(self) -> datetime | None:
        """Return the UTC cutoff datetime, or ``None`` if ``days_past == 0``.

        Returns:
            Aware :class:`datetime` or ``None``.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _parse_xml(self, xml_path: Path) -> Iterator[ParsedDocument]:
        """Iterparse the XML file and yield grouped documents.

        Args:
            xml_path: Absolute path to ``export.xml``.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per group.
        """
        cutoff = self._cutoff_dt()
        filter_types: set[str] = set(self._cfg.record_types)

        # (type, year_month) -> list of (value, unit, start_str)
        records: defaultdict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        workouts: list[dict[str, Any]] = []

        try:
            for _event, elem in ET.iterparse(xml_path, events=("end",)):
                if elem.tag == "Record":
                    self._handle_record(elem, filter_types, cutoff, records)
                elif elem.tag == "Workout":
                    self._handle_workout(elem, cutoff, workouts)
                elem.clear()
        except ET.ParseError as exc:
            logger.warning(f"AppleHealthConnector: XML parse error in {xml_path}: {exc}")
            return
        except OSError as exc:
            logger.warning(f"AppleHealthConnector: cannot read {xml_path}: {exc}")
            return

        yield from self._yield_record_docs(records)
        yield from self._yield_workout_docs(workouts)

    def _handle_record(
        self,
        elem: ET.Element,
        filter_types: set[str],
        cutoff: datetime | None,
        records: defaultdict[tuple[str, str], list[tuple[str, str, str]]],
    ) -> None:
        """Accumulate one ``<Record>`` element into the grouping dict.

        Args:
            elem: The ``<Record>`` XML element.
            filter_types: Set of allowed type strings (empty = all).
            cutoff: Oldest allowed startDate (``None`` = no limit).
            records: Accumulator mapping (type, year_month) -> sample list.
        """
        rec_type = elem.get("type", "")
        if filter_types and rec_type not in filter_types:
            return

        start_str = elem.get("startDate", "")
        dt = _parse_hk_date(start_str)
        if dt is None:
            return
        if cutoff is not None and dt < cutoff:
            return

        value = elem.get("value", "")
        unit = elem.get("unit", "")
        records[(rec_type, _year_month(dt))].append((value, unit, start_str))

    def _handle_workout(
        self,
        elem: ET.Element,
        cutoff: datetime | None,
        workouts: list[dict[str, Any]],
    ) -> None:
        """Accumulate one ``<Workout>`` element.

        Args:
            elem: The ``<Workout>`` XML element.
            cutoff: Oldest allowed startDate (``None`` = no limit).
            workouts: Accumulator list.
        """
        start_str = elem.get("startDate", "")
        dt = _parse_hk_date(start_str)
        if dt is None:
            return
        if cutoff is not None and dt < cutoff:
            return
        workouts.append(
            {
                "type": elem.get("workoutActivityType", ""),
                "duration": elem.get("duration", ""),
                "duration_unit": elem.get("durationUnit", "min"),
                "energy": elem.get("totalEnergyBurned", ""),
                "energy_unit": elem.get("totalEnergyBurnedUnit", "kcal"),
                "start": start_str,
            }
        )

    def _yield_record_docs(
        self,
        records: defaultdict[tuple[str, str], list[tuple[str, str, str]]],
    ) -> Iterator[ParsedDocument]:
        """Emit one ParsedDocument per (record_type, year_month) group.

        Args:
            records: Accumulated record groups.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        source = self._cfg.source_name
        for (rec_type, year_month), samples in sorted(records.items()):
            values: list[float] = []
            for val, _unit, _start in samples:
                with contextlib.suppress(ValueError):
                    values.append(float(val))

            unit = samples[0][1] if samples else ""
            count = len(samples)
            min_val = f"{min(values):.2f}" if values else "N/A"
            max_val = f"{max(values):.2f}" if values else "N/A"
            avg_val = f"{sum(values) / len(values):.2f}" if values else "N/A"

            first_20 = samples[:20]
            sample_lines = "\n".join(f"  {start}: {val} {unit}" for val, unit, start in first_20)

            text = (
                f"Type: {rec_type}\n"
                f"Month: {year_month}\n"
                f"Count: {count}\n"
                f"Min: {min_val} {unit}\n"
                f"Max: {max_val} {unit}\n"
                f"Avg: {avg_val} {unit}\n"
                f"\nSamples:\n{sample_lines}"
            )

            safe_type = rec_type.replace("/", "_")
            yield ParsedDocument(
                path=Path(f"apple_health://{safe_type}/{year_month}.health"),
                text=text,
                file_type=".health",
                encoding="utf-8",
                metadata={
                    "record_type": rec_type,
                    "year_month": year_month,
                    "count": count,
                    "unit": unit,
                    "min": min_val,
                    "max": max_val,
                    "avg": avg_val,
                    "source": source,
                },
            )

        logger.info(f"AppleHealthConnector: {len(records)} record group(s)")

    def _yield_workout_docs(self, workouts: list[dict[str, Any]]) -> Iterator[ParsedDocument]:
        """Emit one ParsedDocument per workout.

        Args:
            workouts: Accumulated workout dicts.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        source = self._cfg.source_name
        for i, w in enumerate(workouts):
            activity = w["type"].replace("HKWorkoutActivityType", "")
            energy_str = f"{w['energy']} {w['energy_unit']}" if w["energy"] else "N/A"
            text = (
                f"Workout: {activity}\n"
                f"Date: {w['start']}\n"
                f"Duration: {w['duration']} {w['duration_unit']}\n"
                f"Energy: {energy_str}"
            )
            safe_act = activity.replace("/", "_")
            start_date = w["start"][:10].replace(" ", "_")
            yield ParsedDocument(
                path=Path(f"apple_health://workout/{safe_act}_{start_date}_{i}.workout"),
                text=text,
                file_type=".workout",
                encoding="utf-8",
                metadata={
                    "activity_type": activity,
                    "date": w["start"],
                    "duration": w["duration"],
                    "duration_unit": w["duration_unit"],
                    "energy_burned": w["energy"],
                    "energy_unit": w["energy_unit"],
                    "source": source,
                },
            )

        logger.info(f"AppleHealthConnector: {len(workouts)} workout(s)")
