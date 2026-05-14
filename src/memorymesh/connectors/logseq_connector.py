"""Logseq graph connector for MemoryMesh.

Reads a Logseq graph directory and yields one
:class:`~memorymesh.core.models.ParsedDocument` per Markdown file, after
stripping Logseq-specific syntax.

Logseq syntax handled
---------------------
* ``((uuid))`` block references -> ``[block]``
* ``{{embed [[Page]]}}`` embeds -> ``[embed: Page]``
* ``{{query ...}}`` blocks -> removed entirely
* ``key:: value`` property lines -> extracted into metadata; removed from
  text body
* ``- `` block bullets -> preserved as plain text
* ``[[wikilinks]]`` -> preserved as-is (useful for search)

Features
--------
* Globs ``pages/`` and ``journals/`` subdirectories (configurable).
* Journals are flagged ``is_journal=True`` in metadata.
* Extracted ``[[wikilinks]]`` are deduplicated and stored in metadata.

Usage
-----
::

    connector = LogseqConnector(LogseqConfig(
        vault_path=Path("~/Logseq/my-graph"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_RE_BLOCK_REF = re.compile(r"\(\([0-9a-f-]{36}\)\)")
_RE_EMBED = re.compile(r"\{\{embed\s+\[\[([^\]]+)\]\]\}\}")
_RE_QUERY = re.compile(r"\{\{query[^}]*\}\}")
_RE_PROPERTY = re.compile(r"^(\w[\w-]*)::\s*(.*)$")
_RE_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")


class LogseqConfig(BaseModel):
    """Configuration for a Logseq graph source.

    Args:
        vault_path: Root directory of the Logseq graph.
        pages_subdir: Subdirectory name for pages (default ``"pages"``).
        journals_subdir: Subdirectory name for journals
            (default ``"journals"``).
        source_name: Name used in the MemoryMesh source registry.
    """

    vault_path: Path
    pages_subdir: str = "pages"
    journals_subdir: str = "journals"
    source_name: str = "logseq"


class LogseqConnector:
    """Parses Logseq Markdown files and yields ParsedDocuments.

    Args:
        config: Vault path and subdirectory settings.
    """

    def __init__(self, config: LogseqConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Glob all Markdown files in pages and journals, yield documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            ``.md`` file, with ``file_type=".logseq"`` and metadata
            containing ``file_name``, ``is_journal``, ``properties``,
            and ``wikilinks``.
        """
        vault = self._cfg.vault_path.expanduser()
        subdirs = [
            (vault / self._cfg.pages_subdir, False),
            (vault / self._cfg.journals_subdir, True),
        ]
        total = 0
        for subdir, is_journal in subdirs:
            if not subdir.exists():
                continue
            for md_file in sorted(subdir.rglob("*.md")):
                doc = self._process_file(md_file, vault, is_journal)
                if doc is not None:
                    yield doc
                    total += 1

        logger.info(f"LogseqConnector: yielded {total} document(s)")

    def _process_file(
        self,
        path: Path,
        vault: Path,
        is_journal: bool,
    ) -> ParsedDocument | None:
        """Parse one Logseq Markdown file.

        Args:
            path: Absolute path to the ``.md`` file.
            vault: Vault root (used to compute the relative path).
            is_journal: ``True`` when the file lives in the journals dir.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the file is empty after stripping.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(f"LogseqConnector: cannot read {path}: {exc}")
            return None

        wikilinks: list[str] = _RE_WIKILINK.findall(raw)
        properties: dict[str, str] = {}
        content_lines: list[str] = []

        for line in raw.splitlines():
            m = _RE_PROPERTY.match(line.strip())
            if m:
                properties[m.group(1)] = m.group(2).strip()
            else:
                content_lines.append(line)

        text = "\n".join(content_lines)
        text = _RE_BLOCK_REF.sub("[block]", text)
        text = _RE_EMBED.sub(lambda m: f"[embed: {m.group(1)}]", text)
        text = _RE_QUERY.sub("", text)
        text = text.strip()

        if not text:
            return None

        try:
            rel_path = path.relative_to(vault)
        except ValueError:
            rel_path = Path(path.name)

        source = self._cfg.source_name
        return ParsedDocument(
            path=Path(f"logseq://{rel_path}.logseq"),
            text=text,
            file_type=".logseq",
            encoding="utf-8",
            metadata={
                "file_name": path.name,
                "is_journal": is_journal,
                "properties": properties,
                "wikilinks": list(set(wikilinks)),
                "source": source,
            },
        )
