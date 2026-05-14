"""Browser history connector for MemoryMesh.

Reads Chrome, Firefox, or Edge local SQLite history databases and yields
visited pages as :class:`~memorymesh.core.models.ParsedDocument` objects.

Features
--------
* **Auto-detection** - finds the default profile path on Windows, Linux, and
  macOS for Chrome, Firefox, and Edge.
* **Safe read** - copies the SQLite file to a temp path before opening it,
  so the live browser database is never locked or corrupted.
* **Privacy** - URL content is never logged at INFO level; only counts and
  timestamps are written.
* **Stdlib only** - uses only ``sqlite3``, ``shutil``, ``tempfile``,
  ``hashlib``, and ``pathlib``.

Usage
-----
::

    connector = BrowserHistoryConnector(BrowserHistoryConfig(
        browser="chrome",
        days_past=90,
        max_urls=5000,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

# Seconds between 1601-01-01 (Windows FILETIME epoch) and 1970-01-01 (Unix)
_WIN_EPOCH_DELTA_S: float = 11_644_473_600.0


class BrowserHistoryConfig(BaseModel):
    """Configuration for a browser history source.

    Args:
        browser: Which browser to read - ``"chrome"``, ``"firefox"``, or
            ``"edge"``.
        profile_path: Explicit path to the SQLite history file.  ``None`` = use
            the OS default for the chosen browser.
        days_past: How many days back to include (0 = no cutoff).
        max_urls: Maximum number of URLs to yield per run (0 = no limit).
        source_name: Name used in the MemoryMesh source registry.
    """

    browser: Literal["chrome", "firefox", "edge"] = "chrome"
    profile_path: Path | None = None
    days_past: int = 90
    max_urls: int = 5_000
    source_name: str = "browser_history"


def _default_history_path(browser: str) -> Path | None:
    """Return the default SQLite history file for *browser* on the current OS.

    Args:
        browser: ``"chrome"``, ``"firefox"``, or ``"edge"``.

    Returns:
        Path to the SQLite file, or ``None`` if the path cannot be determined.
    """
    platform = sys.platform

    if browser in ("chrome", "edge"):
        vendor = "Google/Chrome" if browser == "chrome" else "Microsoft/Edge"
        if platform == "win32":
            base = Path.home() / "AppData" / "Local" / vendor / "User Data" / "Default"
        elif platform == "darwin":
            base = Path.home() / "Library" / "Application Support" / vendor / "Default"
        else:
            # Linux
            slug = "google-chrome" if browser == "chrome" else "microsoft-edge"
            base = Path.home() / ".config" / slug / "Default"
        candidate = base / "History"
        return candidate if candidate.exists() else None

    if browser == "firefox":
        if platform == "win32":
            profiles_dir = Path.home() / "AppData" / "Roaming" / "Mozilla" / "Firefox" / "Profiles"
        elif platform == "darwin":
            profiles_dir = Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
        else:
            profiles_dir = Path.home() / ".mozilla" / "firefox"

        if not profiles_dir.exists():
            return None
        # Pick the first *.default* profile we find
        for candidate in sorted(profiles_dir.iterdir()):
            if "default" in candidate.name.lower() and candidate.is_dir():
                db = candidate / "places.sqlite"
                if db.exists():
                    return db
        return None

    return None


def _chrome_ts_to_unix(chrome_micros: int) -> float:
    """Convert Chrome/Edge ``last_visit_time`` to a Unix timestamp.

    Chrome stores time as microseconds since 1601-01-01.

    Args:
        chrome_micros: Microseconds since 1601-01-01.

    Returns:
        Unix timestamp (float seconds since 1970-01-01).
    """
    return chrome_micros / 1_000_000 - _WIN_EPOCH_DELTA_S


def _unix_to_chrome_micros(unix_ts: float) -> int:
    """Convert a Unix timestamp to Chrome microseconds since 1601-01-01.

    Args:
        unix_ts: Unix timestamp (seconds since 1970-01-01).

    Returns:
        Microseconds since 1601-01-01.
    """
    return int((unix_ts + _WIN_EPOCH_DELTA_S) * 1_000_000)


def _url_hash(url: str) -> str:
    """Return a short, stable hex hash of *url* for use in synthetic paths.

    Args:
        url: The page URL string.

    Returns:
        16-character hexadecimal string.
    """
    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:16]


class BrowserHistoryConnector:
    """Reads browser history from a local SQLite database.

    Opens a *copy* of the database file so it never interferes with a running
    browser session.

    Args:
        config: Browser and fetch settings.
    """

    def __init__(self, config: BrowserHistoryConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Locate the history database and yield visited pages as ParsedDocuments.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per URL,
            with ``file_type=".url"`` and metadata containing ``url``,
            ``title``, ``visit_count``, and ``last_visit_ts``.
        """
        db_path = self._cfg.profile_path or _default_history_path(self._cfg.browser)
        if db_path is None or not db_path.exists():
            logger.warning(
                f"BrowserHistoryConnector: cannot find {self._cfg.browser} history "
                "database. Set profile_path explicitly in config."
            )
            return

        logger.info(
            f"BrowserHistoryConnector: reading {self._cfg.browser} history "
            f"days_past={self._cfg.days_past} max_urls={self._cfg.max_urls}"
        )

        # Copy to temp so we never lock the live browser file
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            shutil.copy2(db_path, tmp_path)
        except OSError as exc:
            logger.warning(f"BrowserHistoryConnector: cannot copy database: {exc}")
            tmp_path.unlink(missing_ok=True)
            return

        try:
            if self._cfg.browser == "firefox":
                yield from self._fetch_firefox(tmp_path)
            else:
                yield from self._fetch_chrome(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _fetch_chrome(self, db_path: Path) -> Iterator[ParsedDocument]:
        """Query Chrome/Edge history SQLite and yield ParsedDocuments.

        Args:
            db_path: Path to a (copied) Chrome/Edge ``History`` SQLite file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per visited URL.
        """
        cutoff_micros = 0
        if self._cfg.days_past > 0:
            cutoff_ts = time.time() - self._cfg.days_past * 86_400
            cutoff_micros = _unix_to_chrome_micros(cutoff_ts)

        limit = self._cfg.max_urls if self._cfg.max_urls > 0 else -1

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError as exc:
            logger.warning(f"BrowserHistoryConnector: cannot open Chrome DB: {exc}")
            return

        try:
            cursor = conn.execute(
                """
                SELECT url, title, visit_count, last_visit_time
                FROM urls
                WHERE last_visit_time > ?
                ORDER BY last_visit_time DESC
                LIMIT ?
                """,
                (cutoff_micros, limit),
            )
            yielded = 0
            for row in cursor:
                url, title, visit_count, last_visit_time = row
                title = title or ""
                url = url or ""
                if not url:
                    continue
                doc = self._build_document(
                    url=url,
                    title=title,
                    visit_count=int(visit_count or 0),
                    last_visit_ts=_chrome_ts_to_unix(int(last_visit_time or 0)),
                )
                yield doc
                yielded += 1
            logger.info(f"BrowserHistoryConnector: yielded {yielded} Chrome/Edge URL(s)")
        except sqlite3.Error as exc:
            logger.warning(f"BrowserHistoryConnector: Chrome query error: {exc}")
        finally:
            conn.close()

    def _fetch_firefox(self, db_path: Path) -> Iterator[ParsedDocument]:
        """Query Firefox ``places.sqlite`` history and yield ParsedDocuments.

        Args:
            db_path: Path to a (copied) Firefox ``places.sqlite`` file.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per visited URL.
        """
        cutoff_micros = 0
        if self._cfg.days_past > 0:
            cutoff_ts = time.time() - self._cfg.days_past * 86_400
            cutoff_micros = int(cutoff_ts * 1_000_000)

        limit = self._cfg.max_urls if self._cfg.max_urls > 0 else -1

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError as exc:
            logger.warning(f"BrowserHistoryConnector: cannot open Firefox DB: {exc}")
            return

        try:
            cursor = conn.execute(
                """
                SELECT url, title, visit_count, last_visit_date
                FROM moz_places
                WHERE last_visit_date > ?
                ORDER BY last_visit_date DESC
                LIMIT ?
                """,
                (cutoff_micros, limit),
            )
            yielded = 0
            for row in cursor:
                url, title, visit_count, last_visit_date = row
                title = title or ""
                url = url or ""
                if not url:
                    continue
                last_visit_ts = (int(last_visit_date or 0)) / 1_000_000
                doc = self._build_document(
                    url=url,
                    title=title,
                    visit_count=int(visit_count or 0),
                    last_visit_ts=last_visit_ts,
                )
                yield doc
                yielded += 1
            logger.info(f"BrowserHistoryConnector: yielded {yielded} Firefox URL(s)")
        except sqlite3.Error as exc:
            logger.warning(f"BrowserHistoryConnector: Firefox query error: {exc}")
        finally:
            conn.close()

    def _build_document(
        self,
        *,
        url: str,
        title: str,
        visit_count: int,
        last_visit_ts: float,
    ) -> ParsedDocument:
        """Assemble a ParsedDocument from a visited URL record.

        Args:
            url: The page URL.
            title: Page title (may be empty).
            visit_count: Number of times the URL was visited.
            last_visit_ts: Unix timestamp of the most recent visit.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        display_title = title or url
        text = f"Title: {display_title}\nURL: {url}\nVisits: {visit_count}"

        url_hash = _url_hash(url)
        source = self._cfg.source_name
        synthetic_path = Path(f"browser://{self._cfg.browser}/{url_hash}.url")

        return ParsedDocument(
            path=synthetic_path,
            text=text,
            file_type=".url",
            encoding="utf-8",
            metadata={
                "url": url,
                "title": title,
                "visit_count": visit_count,
                "last_visit_ts": last_visit_ts,
                "browser": self._cfg.browser,
                "source": source,
            },
        )
