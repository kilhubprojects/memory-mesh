"""CalDAV calendar source connector for MemoryMesh.

Fetches calendar events from any CalDAV server (Google Calendar, iCloud,
Nextcloud, FastMail, etc.) and converts them into
:class:`~memorymesh.core.models.ParsedDocument` objects so they can be
searched like any other personal knowledge.

Requires the ``caldav`` package::

    uv add caldav

Usage
-----
::

    connector = CalDAVConnector(CalDAVConfig(
        url="https://caldav.fastmail.com/",
        username="you@fastmail.com",
        password="app-password",
        calendar_name="Personal",   # omit to sync all calendars
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)

Privacy note
------------
No event text is logged at INFO level.  Only counts, calendar names, and
date ranges are written to the log.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.core.models import ParsedDocument


class CalDAVConfig(BaseModel):
    """Configuration for a CalDAV source.

    Args:
        url: CalDAV server URL (e.g. ``https://caldav.fastmail.com/``).
        username: Account username / email.
        password: Account password or app-specific password.
        calendar_name: Name of the calendar to sync.  ``None`` = all calendars.
        days_past: How many days in the past to fetch (0 = from beginning).
        days_future: How many days into the future to fetch.
        max_events: Cap on total events per sync run (0 = no limit).
        source_name: Name used in the MemoryMesh source registry.
    """

    url: str
    username: str
    password: SecretStr
    calendar_name: str | None = None
    days_past: int = 30
    days_future: int = 7
    max_events: int = 500
    source_name: str = "calendar"


class CalDAVConnector:
    """Fetches calendar events from a CalDAV server as ParsedDocuments.

    Args:
        config: CalDAV connection and fetch settings.
    """

    def __init__(self, config: CalDAVConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Connect to CalDAV and yield events as ParsedDocuments.

        Requires the ``caldav`` package.  If not installed, logs a warning and
        yields nothing rather than raising an ImportError.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per event,
            with ``file_type=".ics"`` and metadata containing ``summary``,
            ``dtstart``, ``dtend``, ``location``, and ``uid``.
        """
        try:
            import caldav  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "CalDAVConnector: 'caldav' package not installed. "
                "Run `uv add caldav` to enable calendar sync."
            )
            return

        password = self._cfg.password.get_secret_value()
        try:
            client = caldav.DAVClient(
                url=self._cfg.url,
                username=self._cfg.username,
                password=password,
            )
            principal = client.principal()
        except Exception as exc:
            logger.error(f"CalDAVConnector: connection failed: {exc}")
            return

        try:
            calendars = principal.calendars()
        except Exception as exc:
            logger.error(f"CalDAVConnector: cannot list calendars: {exc}")
            return

        if self._cfg.calendar_name:
            calendars = [c for c in calendars if getattr(c, "name", "") == self._cfg.calendar_name]
            if not calendars:
                logger.warning(f"CalDAVConnector: calendar {self._cfg.calendar_name!r} not found")
                return

        logger.info(f"CalDAVConnector: syncing {len(calendars)} calendar(s) from {self._cfg.url}")

        total = 0
        for calendar in calendars:
            for doc in self._fetch_calendar(calendar):
                yield doc
                total += 1
                if self._cfg.max_events > 0 and total >= self._cfg.max_events:
                    return

    def _fetch_calendar(self, calendar: object) -> Iterator[ParsedDocument]:
        """Fetch events from a single calendar object.

        Args:
            calendar: A ``caldav.Calendar`` instance.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per event.
        """
        import datetime

        now = datetime.datetime.now(tz=datetime.UTC)
        start = now - datetime.timedelta(days=self._cfg.days_past) if self._cfg.days_past else None
        end = now + datetime.timedelta(days=self._cfg.days_future)

        cal_name = getattr(calendar, "name", "unknown")

        try:
            events = calendar.date_search(start=start, end=end) if start else calendar.events()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(f"CalDAVConnector: search failed for {cal_name!r}: {exc}")
            return

        logger.info(f"CalDAVConnector: {len(events)} event(s) in {cal_name!r}")

        for event in events:
            try:
                doc = self._event_to_document(event, cal_name)
                if doc is not None:
                    yield doc
            except Exception as exc:
                logger.warning(f"CalDAVConnector: parse error in {cal_name!r}: {exc}")

    def _event_to_document(self, event: object, cal_name: str) -> ParsedDocument | None:
        """Convert a caldav Event object to a ParsedDocument.

        Args:
            event: A ``caldav.Event`` object.
            cal_name: Human-readable calendar name.

        Returns:
            Parsed document or ``None`` if the event cannot be parsed.
        """
        try:
            from icalendar import Calendar as iCal  # type: ignore[import-untyped]

            raw_ics: bytes = getattr(event, "data", b"")
            if isinstance(raw_ics, str):
                raw_ics = raw_ics.encode()
            cal = iCal.from_ical(raw_ics)
        except Exception:
            # Fallback: use whatever string repr is available
            raw_str = str(getattr(event, "data", ""))
            uid = str(time.time())
            return self._build_document(
                summary="Calendar event",
                description=raw_str[:500],
                dtstart="",
                dtend="",
                location="",
                uid=uid,
                cal_name=cal_name,
            )

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            def _str(key: str, _comp: object = component) -> str:
                val = _comp.get(key)  # type: ignore[attr-defined,union-attr]
                if val is None:
                    return ""
                if hasattr(val, "dt"):
                    return str(val.dt)
                return str(val)

            summary = _str("SUMMARY")
            description = _str("DESCRIPTION")
            dtstart = _str("DTSTART")
            dtend = _str("DTEND")
            location = _str("LOCATION")
            uid = _str("UID") or str(time.time())

            return self._build_document(
                summary=summary,
                description=description,
                dtstart=dtstart,
                dtend=dtend,
                location=location,
                uid=uid,
                cal_name=cal_name,
            )

        return None

    def _build_document(
        self,
        *,
        summary: str,
        description: str,
        dtstart: str,
        dtend: str,
        location: str,
        uid: str,
        cal_name: str,
    ) -> ParsedDocument:
        """Assemble a ParsedDocument from event fields.

        Args:
            summary: Event title / summary.
            description: Event description body.
            dtstart: Start datetime string.
            dtend: End datetime string.
            location: Event location.
            uid: Unique event identifier.
            cal_name: Human-readable calendar name.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        parts: list[str] = []
        if summary:
            parts.append(f"Title: {summary}")
        if dtstart:
            parts.append(f"Start: {dtstart}")
        if dtend:
            parts.append(f"End: {dtend}")
        if location:
            parts.append(f"Location: {location}")
        if cal_name:
            parts.append(f"Calendar: {cal_name}")
        if description:
            parts.append("")
            parts.append(description)

        text = "\n".join(parts)
        source = self._cfg.source_name or "calendar"
        safe_uid = re.sub(r"[^a-zA-Z0-9_.-]", "_", uid)[:64]
        synthetic_path = Path(f"caldav://{source}/{cal_name}/{safe_uid}.ics")

        return ParsedDocument(
            path=synthetic_path,
            text=text,
            file_type=".ics",
            encoding="utf-8",
            metadata={
                "uid": uid,
                "summary": summary,
                "dtstart": dtstart,
                "dtend": dtend,
                "location": location,
                "calendar": cal_name,
                "source": source,
            },
        )
