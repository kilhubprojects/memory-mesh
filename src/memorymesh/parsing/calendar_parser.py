"""iCalendar (.ics) parser.

Each ``VEVENT`` component in the file becomes one
:class:`~memorymesh.core.models.ParsedDocument`.  ``VTODO`` and ``VJOURNAL``
components are silently ignored.

Requires the ``icalendar`` package (added to ``pyproject.toml`` as an optional
dep under the ``[calendar]`` extra, or as a required dep when
``source.type: calendar`` is configured).

Activated for ``.ics`` files.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser


def _dt_to_str(dt_value: object) -> str:
    """Convert an icalendar date/datetime to an ISO 8601 string.

    Args:
        dt_value: Value returned by icalendar for a DTSTART/DTEND property.
            Can be a :class:`~datetime.datetime`, :class:`~datetime.date`,
            a ``vDatetime``, ``vDate``, or a ``vText`` wrapper.

    Returns:
        ISO 8601 string, or empty string if conversion fails.
    """
    if dt_value is None:
        return ""
    try:
        # icalendar DT types have a .dt attribute
        dt = getattr(dt_value, "dt", dt_value)
        return dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
    except Exception:
        return str(dt_value)


def _to_str(value: object) -> str:
    """Safely convert an icalendar property value to a plain string.

    Args:
        value: Any icalendar property object or ``None``.

    Returns:
        String representation or empty string.
    """
    if value is None:
        return ""
    # icalendar vText / vCalAddress etc. expose their string value via str().
    return str(value).strip()


class CalendarParser(Parser):
    """Parser for iCalendar files.

    Processes ``VEVENT`` components only; ``VTODO`` and ``VJOURNAL`` are
    ignored.  The calendar name is read from the ``X-WR-CALNAME`` property
    when present.

    Each event is returned as a separate document.  Because the standard
    ``Parser.parse`` interface returns one document per call, this parser
    concatenates all events into a single document.  Callers that need
    per-event documents should use :meth:`parse_all`.

    Metadata populated per event:

    * ``dtstart`` - ISO 8601 start datetime.
    * ``dtend`` - ISO 8601 end datetime.
    * ``uid`` - ``UID`` property value.
    * ``organizer`` - ``ORGANIZER`` property value (usually a ``mailto:`` URI).
    * ``calendar_name`` - ``X-WR-CALNAME`` from the top-level calendar, if present.
    * ``event_index`` - zero-based position in the file.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: iCalendar formats."""
        return frozenset({".ics", ".ical", ".ifb"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse an iCalendar file, returning all events as one document.

        Args:
            path: Absolute path to the ``.ics`` file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with all event
            texts concatenated and metadata from the first event.
        """
        docs = self.parse_all(path)
        if not docs:
            return ParsedDocument(
                path=path,
                text="",
                file_type=".ics",
                metadata={"event_count": 0},
            )

        combined = "\n\n---\n\n".join(
            f"Event: {d.metadata.get('summary', '(no title)')}\n"
            f"Start: {d.metadata.get('dtstart', '')}\n\n"
            f"{d.text}"
            for d in docs
        )
        meta: dict[str, object] = {**docs[0].metadata, "event_count": len(docs)}
        return ParsedDocument(
            path=path,
            text=combined,
            file_type=".ics",
            encoding="utf-8",
            metadata=meta,
        )

    def parse_all(self, path: Path) -> list[ParsedDocument]:
        """Return one :class:`ParsedDocument` per ``VEVENT``.

        Args:
            path: Absolute path to the ``.ics`` file.

        Returns:
            List of documents, one per event.  Returns empty list on error
            (error is logged at WARNING level).
        """
        try:
            import icalendar  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "CalendarParser: 'icalendar' package not installed. "
                "Install it with: pip install icalendar"
            )
            return []

        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning(f"CalendarParser: cannot read {path}: {exc}")
            return []

        try:
            cal = icalendar.Calendar.from_ical(raw)
        except Exception as exc:
            logger.warning(f"CalendarParser: invalid iCal in {path.name!r}: {exc}")
            return []

        # Extract calendar-level metadata
        calendar_name = _to_str(cal.get("X-WR-CALNAME"))

        docs: list[ParsedDocument] = []
        event_index = 0

        for component in cal.walk():
            comp_name = getattr(component, "name", None)

            if comp_name == "VTODO" or comp_name == "VJOURNAL":
                logger.debug(f"CalendarParser: ignoring {comp_name} component in {path.name!r}")
                continue

            if comp_name != "VEVENT":
                continue

            summary = _to_str(component.get("SUMMARY"))
            description = _to_str(component.get("DESCRIPTION"))
            location = _to_str(component.get("LOCATION"))

            text_parts = [p for p in (summary, description, location) if p]
            text = "\n".join(text_parts).strip()

            dtstart = _dt_to_str(component.get("DTSTART"))
            dtend = _dt_to_str(component.get("DTEND"))
            uid = _to_str(component.get("UID"))
            organizer = _to_str(component.get("ORGANIZER"))

            meta: dict[str, object] = {
                "dtstart": dtstart,
                "dtend": dtend,
                "uid": uid,
                "organizer": organizer,
                "calendar_name": calendar_name,
                "summary": summary,
                "event_index": event_index,
            }

            docs.append(
                ParsedDocument(
                    path=path,
                    text=text,
                    file_type=".ics",
                    encoding="utf-8",
                    metadata=meta,
                )
            )
            event_index += 1

        logger.debug(f"CalendarParser: {path.name!r} events={len(docs)}")
        return docs
