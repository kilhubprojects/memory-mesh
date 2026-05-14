"""Browser history parser for Chrome, Firefox, and Brave.

Safety rules:
* **Never opens the browser's live SQLite database.**  The file is first copied
  to ``~/.memorymesh/tmp/history_<md5>.db`` in a ``try`` block, then deleted
  in the corresponding ``finally``.
* Only the stdlib is used: ``sqlite3``, ``shutil``, ``hashlib``, ``datetime``,
  ``pathlib``, ``platform``, ``os``.

The parser auto-detects the browser by checking known default paths for each
OS.  It activates for ``.db`` files when a source is configured with
``source.type: browser_history``.  Alternatively, the config can point directly
at the history DB file via ``source.path``.

Each visited URL (with ``visit_count >= 2``) becomes one
:class:`~memorymesh.core.models.ParsedDocument`.

Metadata:
* ``url`` - the full URL.
* ``visit_count`` - number of recorded visits.
* ``last_visit`` - ISO 8601 datetime string.
* ``browser`` - ``"chrome"``, ``"firefox"``, or ``"brave"``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

_TMP_DIR = Path("~/.memorymesh/tmp").expanduser()
_MIN_VISIT_COUNT = 2


_CHROME_PATHS: dict[str, list[Path]] = {
    "Windows": [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data/Default/History",
    ],
    "Darwin": [Path("~/Library/Application Support/Google/Chrome/Default/History").expanduser()],
    "Linux": [Path("~/.config/google-chrome/Default/History").expanduser()],
}

_BRAVE_PATHS: dict[str, list[Path]] = {
    "Windows": [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "BraveSoftware/Brave-Browser/User Data/Default/History",
    ],
    "Darwin": [
        Path(
            "~/Library/Application Support/BraveSoftware/Brave-Browser/Default/History"
        ).expanduser()
    ],
    "Linux": [Path("~/.config/BraveSoftware/Brave-Browser/Default/History").expanduser()],
}

_FIREFOX_PATHS: dict[str, list[Path]] = {
    "Windows": [Path(os.environ.get("APPDATA", "")) / "Mozilla/Firefox/Profiles"],
    "Darwin": [Path("~/Library/Application Support/Firefox/Profiles").expanduser()],
    "Linux": [Path("~/.mozilla/firefox").expanduser()],
}


def _find_firefox_db() -> Path | None:
    """Find the default Firefox ``places.sqlite`` by scanning profile dirs.

    Returns:
        Absolute :class:`~pathlib.Path` to ``places.sqlite``, or ``None``.
    """
    system = platform.system()
    candidates = _FIREFOX_PATHS.get(system, [])
    for base in candidates:
        if not base.exists():
            continue
        # Firefox profiles are named ``<random>.default`` or ``<random>.default-release``
        for profile in base.iterdir():
            db = profile / "places.sqlite"
            if db.exists():
                return db
    return None


def _safe_copy(src: Path) -> Path:
    """Copy *src* to a temp path to avoid opening a live SQLite WAL.

    Args:
        src: Source database path (live browser database).

    Returns:
        :class:`~pathlib.Path` to the temporary copy.
    """
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(src).encode()).hexdigest()
    dst = _TMP_DIR / f"history_{digest}.db"
    shutil.copy2(str(src), str(dst))
    return dst


def _chrome_epoch_to_dt(microseconds: int) -> datetime:
    """Convert Chrome's epoch (microseconds since 1601-01-01) to a UTC datetime.

    Args:
        microseconds: Chrome timestamp value.

    Returns:
        UTC-aware :class:`~datetime.datetime`.
    """
    # Seconds between 1601-01-01 and 1970-01-01
    win_epoch_delta_s = 11_644_473_600
    seconds = microseconds / 1_000_000 - win_epoch_delta_s
    return datetime.fromtimestamp(seconds, tz=UTC)


def _firefox_epoch_to_dt(microseconds: int) -> datetime:
    """Convert Firefox's epoch (microseconds since Unix epoch) to a UTC datetime.

    Args:
        microseconds: Firefox timestamp value (``last_visit_date``).

    Returns:
        UTC-aware :class:`~datetime.datetime`.
    """
    return datetime.fromtimestamp(microseconds / 1_000_000, tz=UTC)


def _query_chrome(db_copy: Path, browser: str) -> list[dict[str, Any]]:
    """Query a Chrome/Brave history database copy.

    Args:
        db_copy: Path to the *copied* (not live) database.
        browser: ``"chrome"`` or ``"brave"`` - stored in metadata.

    Returns:
        List of row dicts with keys ``url``, ``title``, ``visit_count``,
        ``last_visit``.
    """
    rows: list[dict[str, Any]] = []
    con = sqlite3.connect(str(db_copy))
    try:
        cur = con.execute(
            "SELECT url, title, visit_count, last_visit_time "
            "FROM urls "
            "WHERE visit_count >= ? "
            "ORDER BY last_visit_time DESC",
            (_MIN_VISIT_COUNT,),
        )
        for url, title, visit_count, last_visit_time in cur.fetchall():
            try:
                dt = _chrome_epoch_to_dt(int(last_visit_time))
            except (ValueError, OSError):
                dt = datetime(1970, 1, 1, tzinfo=UTC)
            rows.append(
                {
                    "url": url or "",
                    "title": title or "",
                    "visit_count": int(visit_count),
                    "last_visit": dt.isoformat(),
                    "browser": browser,
                }
            )
    except sqlite3.Error as exc:
        logger.warning(f"BrowserHistoryParser: SQLite error reading {browser} db: {exc}")
    finally:
        con.close()
    return rows


def _query_firefox(db_copy: Path) -> list[dict[str, Any]]:
    """Query a Firefox ``places.sqlite`` database copy.

    Args:
        db_copy: Path to the *copied* (not live) database.

    Returns:
        List of row dicts with keys ``url``, ``title``, ``visit_count``,
        ``last_visit``, ``browser``.
    """
    rows: list[dict[str, Any]] = []
    con = sqlite3.connect(str(db_copy))
    try:
        cur = con.execute(
            "SELECT url, title, visit_count, last_visit_date "
            "FROM moz_places "
            "WHERE visit_count >= ? AND last_visit_date IS NOT NULL "
            "ORDER BY last_visit_date DESC",
            (_MIN_VISIT_COUNT,),
        )
        for url, title, visit_count, last_visit_date in cur.fetchall():
            try:
                dt = _firefox_epoch_to_dt(int(last_visit_date))
            except (ValueError, OSError):
                dt = datetime(1970, 1, 1, tzinfo=UTC)
            rows.append(
                {
                    "url": url or "",
                    "title": title or "",
                    "visit_count": int(visit_count),
                    "last_visit": dt.isoformat(),
                    "browser": "firefox",
                }
            )
    except sqlite3.Error as exc:
        logger.warning(f"BrowserHistoryParser: SQLite error reading Firefox db: {exc}")
    finally:
        con.close()
    return rows


class BrowserHistoryParser(Parser):
    """Parser for browser history SQLite databases (Chrome, Firefox, Brave).

    Detects the correct browser by inspecting the column schema of the copied
    database (``urls`` table -> Chrome/Brave; ``moz_places`` -> Firefox).

    Activated for ``.db`` files when a source is configured with
    ``source.type: browser_history``, or when the source path points directly
    to the browser history DB.

    Because the standard ``Parser.parse`` interface is one-document-per-call,
    all history rows are packed into a single document with entries separated
    by newlines.  Use :meth:`parse_all` for per-URL documents.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: SQLite browser history databases."""
        return frozenset({".db", ".sqlite"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a browser history database file.

        Args:
            path: Absolute path to the history ``.db`` or ``.sqlite`` file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with all URLs
            (visit_count >= 2) packed as ``title\\nurl`` pairs, separated by
            newlines.
        """
        docs = self.parse_all(path)
        if not docs:
            return ParsedDocument(
                path=path,
                text="",
                file_type=".db",
                metadata={"url_count": 0},
            )

        lines: list[str] = []
        for d in docs:
            title = str(d.metadata.get("title", "")).strip()
            url = str(d.metadata.get("url", "")).strip()
            entry = f"{title}\n{url}" if title else url
            lines.append(entry)

        meta: dict[str, object] = {
            "url_count": len(docs),
            "browser": docs[0].metadata.get("browser", "unknown"),
        }
        return ParsedDocument(
            path=path,
            text="\n\n".join(lines),
            file_type=".db",
            encoding="utf-8",
            metadata=meta,
        )

    def parse_all(self, path: Path) -> list[ParsedDocument]:
        """Return one :class:`ParsedDocument` per visited URL.

        Args:
            path: Absolute path to the history database.

        Returns:
            List of documents, one per URL row.  Empty list on error.
        """
        db_copy: Path | None = None
        try:
            db_copy = _safe_copy(path)
            rows = self._query(db_copy)
        except OSError as exc:
            logger.warning(f"BrowserHistoryParser: cannot copy {path}: {exc}")
            return []
        except Exception as exc:
            logger.warning(f"BrowserHistoryParser: error processing {path}: {exc}")
            return []
        finally:
            if db_copy is not None:
                with contextlib.suppress(OSError):
                    db_copy.unlink(missing_ok=True)

        docs: list[ParsedDocument] = []
        for row in rows:
            url = row.get("url", "")
            title = row.get("title", "")
            text = f"{title}\n{url}".strip() if title else url
            meta: dict[str, object] = {
                "url": url,
                "visit_count": row.get("visit_count", 0),
                "last_visit": row.get("last_visit", ""),
                "browser": row.get("browser", "unknown"),
                "title": title,
            }
            docs.append(
                ParsedDocument(
                    path=path,
                    text=text,
                    file_type=".db",
                    encoding="utf-8",
                    metadata=meta,
                )
            )

        logger.debug(f"BrowserHistoryParser: {path.name!r} urls={len(docs)}")
        return docs

    @staticmethod
    def _query(db_copy: Path) -> list[dict[str, Any]]:
        """Detect browser type from schema and run the appropriate query.

        Args:
            db_copy: Path to the temporary database copy.

        Returns:
            List of row dicts from the appropriate query function.
        """
        con = sqlite3.connect(str(db_copy))
        try:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            con.close()

        if "urls" in tables:
            # Chrome or Brave - distinguish by path is optional; schema is the same
            # We can detect Brave by path later; for now label by DB content.
            return _query_chrome(db_copy, browser="chrome")

        if "moz_places" in tables:
            return _query_firefox(db_copy)

        logger.warning(
            f"BrowserHistoryParser: unrecognised schema in {db_copy.name!r} "
            "- expected 'urls' (Chrome/Brave) or 'moz_places' (Firefox)"
        )
        return []

    @classmethod
    def find_default_paths(cls) -> dict[str, Path]:
        """Find default browser history database paths for the current OS.

        Returns:
            Dict of ``browser_name -> Path`` for all browsers whose default
            history DB was found to exist.
        """
        system = platform.system()
        result: dict[str, Path] = {}

        for paths in _CHROME_PATHS.get(system, []):
            if paths.exists():
                result["chrome"] = paths
                break

        for paths in _BRAVE_PATHS.get(system, []):
            if paths.exists():
                result["brave"] = paths
                break

        ff = _find_firefox_db()
        if ff is not None:
            result["firefox"] = ff

        return result
