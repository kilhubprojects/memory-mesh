"""Letterboxd data export connector for MemoryMesh.

Parses Letterboxd data exports (Settings -> Import & Export -> Export
Your Data) and yields
:class:`~memorymesh.core.models.ParsedDocument` objects for watched
films and the watchlist.

Files parsed
------------
* ``diary.csv`` - viewing diary entries.
* ``reviews.csv`` - written film reviews.
* ``watchlist.csv`` - films to watch later.

``diary.csv`` and ``reviews.csv`` are merged on ``(Name, Watched Date)``
so that a review is attached to its diary entry when both exist.

All files are optional; missing files are silently skipped.

Usage
-----
::

    connector = LetterboxdConnector(LetterboxdConfig(
        export_path=Path("~/Downloads/letterboxd"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class LetterboxdConfig(BaseModel):
    """Configuration for a Letterboxd data export source.

    Args:
        export_path: Directory containing Letterboxd CSV export files.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    source_name: str = "letterboxd"


class LetterboxdConnector:
    """Parses Letterboxd CSV exports and yields ParsedDocuments.

    Args:
        config: Export path and source settings.
    """

    def __init__(self, config: LetterboxdConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse diary, reviews, and watchlist, yielding documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            watched film (diary + reviews merged) with
            ``file_type=".letterboxd"``, and one for the full watchlist.
        """
        base = self._cfg.export_path.expanduser()
        total = 0

        for doc in self._parse_diary_and_reviews(base):
            yield doc
            total += 1

        watchlist_doc = self._parse_watchlist(base)
        if watchlist_doc is not None:
            yield watchlist_doc
            total += 1

        logger.info(f"LetterboxdConnector: yielded {total} document(s)")

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        """Read a Letterboxd CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            List of row dicts, or an empty list if the file is absent or
            unreadable.
        """
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning(f"LetterboxdConnector: cannot read {path}: {exc}")
            return []
        reader = csv.DictReader(text.splitlines())
        return list(reader)

    def _parse_diary_and_reviews(self, base: Path) -> Iterator[ParsedDocument]:
        """Merge diary and review CSVs and yield one doc per film.

        Args:
            base: Export root directory.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per film.
        """
        diary_rows = self._read_csv(base / "diary.csv")
        review_rows = self._read_csv(base / "reviews.csv")

        # (Name, Watched Date) -> film dict
        merged: dict[tuple[str, str], dict[str, str]] = {}

        for row in diary_rows:
            key = (row.get("Name", ""), row.get("Watched Date", ""))
            merged[key] = {
                "name": row.get("Name", ""),
                "year": row.get("Year", ""),
                "rating": row.get("Rating", ""),
                "rewatch": row.get("Rewatch", ""),
                "tags": row.get("Tags", ""),
                "watched_date": row.get("Watched Date", ""),
                "review": "",
            }

        for row in review_rows:
            key = (row.get("Name", ""), row.get("Watched Date", ""))
            if key in merged:
                merged[key]["review"] = row.get("Review", "")
            else:
                merged[key] = {
                    "name": row.get("Name", ""),
                    "year": row.get("Year", ""),
                    "rating": row.get("Rating", ""),
                    "rewatch": "",
                    "tags": "",
                    "watched_date": row.get("Watched Date", ""),
                    "review": row.get("Review", ""),
                }

        source = self._cfg.source_name
        for (_name, watched_date), film in sorted(merged.items()):
            name = film["name"]
            if not name:
                continue

            text_parts = [
                f"Film: {name} ({film['year']})",
                f"Rating: {film['rating']}/5",
                f"Watched: {watched_date}",
                f"Rewatch: {film['rewatch']}",
                f"Tags: {film['tags']}",
            ]
            if film["review"]:
                text_parts.append(f"\n{film['review']}")

            safe_name = name.replace("/", "_").replace(" ", "_")
            safe_date = watched_date.replace("/", "-")
            yield ParsedDocument(
                path=Path(f"letterboxd://{safe_name}_{safe_date}.letterboxd"),
                text="\n".join(text_parts),
                file_type=".letterboxd",
                encoding="utf-8",
                metadata={
                    "name": name,
                    "year": film["year"],
                    "rating": film["rating"],
                    "watched_date": watched_date,
                    "has_review": bool(film["review"]),
                    "source": source,
                },
            )

    def _parse_watchlist(self, base: Path) -> ParsedDocument | None:
        """Yield a single document listing all watchlist films.

        Args:
            base: Export root directory.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if ``watchlist.csv`` is absent or empty.
        """
        rows = self._read_csv(base / "watchlist.csv")
        if not rows:
            return None

        lines = [
            f"{row.get('Name', '')} ({row.get('Year', '')})" for row in rows if row.get("Name")
        ]
        if not lines:
            return None

        return ParsedDocument(
            path=Path("letterboxd://watchlist.letterboxd"),
            text=f"Watchlist ({len(lines)} films)\n\n" + "\n".join(lines),
            file_type=".letterboxd",
            encoding="utf-8",
            metadata={
                "type": "watchlist",
                "film_count": len(lines),
                "source": self._cfg.source_name,
            },
        )
