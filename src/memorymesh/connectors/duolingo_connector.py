"""Duolingo connector for MemoryMesh.

Parses a Duolingo data export (JSON) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per learning language
plus one summary document.

Export format
-------------
Duolingo data exports can be requested from
https://www.duolingo.com/settings/privacy.
The export ZIP contains a ``duolingo_data.json`` (or similar) file with
streak info, XP per language, skill breakdowns, and lesson history.

Features
--------
* **Per-language documents** - one doc per language with XP, level,
  crown count, and recent skill names.
* **Summary document** - one aggregate doc with overall stats and streak.
* **ZIP support** - if the path points to a ``.zip`` file, the first
  JSON file inside is used automatically.

Usage
-----
::

    connector = DuolingoConnector(DuolingoConfig(
        export_path=Path("duolingo_data.json"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class DuolingoConfig(BaseModel):
    """Configuration for a Duolingo export source.

    Args:
        export_path: Path to the Duolingo JSON export file or ZIP.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    source_name: str = "duolingo"


class DuolingoConnector:
    """Parses a Duolingo JSON export and yields ParsedDocuments.

    Args:
        config: Export path and source settings.
    """

    def __init__(self, config: DuolingoConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Parse the export and yield language documents plus a summary.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            language and one summary, with ``file_type=".duolingo"``.
        """
        raw = self._load_json()
        if raw is None:
            return

        if not isinstance(raw, dict):
            logger.warning("DuolingoConnector: unexpected JSON structure")
            return

        total = 0

        languages = raw.get("languages") or raw.get("courses") or []
        username = raw.get("username", "")
        site_streak = raw.get("site_streak") or raw.get("streak_info", {}).get("site_streak", 0)
        total_xp = raw.get("total_xp", 0)

        for lang in languages:
            if not isinstance(lang, dict):
                continue
            doc = self._build_lang_doc(lang)
            if doc is not None:
                yield doc
                total += 1

        summary = self._build_summary_doc(
            username, int(site_streak or 0), int(total_xp or 0), len(languages)
        )
        if summary is not None:
            yield summary
            total += 1

        logger.info(f"DuolingoConnector: yielded {total} document(s)")

    def _load_json(self) -> Any | None:
        """Load the export JSON from file or ZIP.

        Returns:
            Parsed JSON value, or ``None`` on failure.
        """
        path = self._cfg.export_path
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    json_names = [n for n in zf.namelist() if n.endswith(".json")]
                    if not json_names:
                        logger.warning("DuolingoConnector: no JSON in ZIP")
                        return None
                    with zf.open(json_names[0]) as f:
                        return json.loads(f.read())
            else:
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"DuolingoConnector: failed to load export: {exc}")
            return None

    def _build_lang_doc(self, lang: dict[str, Any]) -> ParsedDocument | None:
        """Build a document for a single language.

        Args:
            lang: Language/course dict from the export.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the language name is missing.
        """
        language = lang.get("language_string") or lang.get("title") or lang.get("language") or ""
        if not language:
            return None

        xp = int(lang.get("xp", 0) or lang.get("points", 0) or 0)
        level = int(lang.get("level", 0) or 0)
        crowns = int(lang.get("crowns", 0) or 0)

        skills: list[str] = []
        for skill in (lang.get("skills") or [])[:10]:
            if isinstance(skill, dict):
                name = skill.get("name") or skill.get("title") or ""
                if name:
                    skills.append(name)

        text_parts = [
            f"Duolingo: {language}",
            f"XP: {xp}",
            f"Level: {level}",
            f"Crowns: {crowns}",
        ]
        if skills:
            text_parts.append(f"Recent skills: {', '.join(skills)}")

        safe_lang = language.replace(" ", "_").lower()
        return ParsedDocument(
            path=Path(f"duolingo://{safe_lang}.duolingo"),
            text="\n".join(text_parts),
            file_type=".duolingo",
            encoding="utf-8",
            metadata={
                "language": language,
                "xp": xp,
                "level": level,
                "crowns": crowns,
                "source": self._cfg.source_name,
            },
        )

    def _build_summary_doc(
        self,
        username: str,
        streak: int,
        total_xp: int,
        lang_count: int,
    ) -> ParsedDocument | None:
        """Build the aggregate summary document.

        Args:
            username: Duolingo username.
            streak: Current day streak.
            total_xp: Total XP across all languages.
            lang_count: Number of languages being learned.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`.
        """
        text_parts = [
            f"Duolingo summary for {username}" if username else "Duolingo summary",
            f"Streak: {streak} days",
            f"Total XP: {total_xp}",
            f"Languages: {lang_count}",
        ]
        return ParsedDocument(
            path=Path("duolingo://summary.duolingo"),
            text="\n".join(text_parts),
            file_type=".duolingo",
            encoding="utf-8",
            metadata={
                "username": username,
                "streak": streak,
                "total_xp": total_xp,
                "lang_count": lang_count,
                "source": self._cfg.source_name,
            },
        )
