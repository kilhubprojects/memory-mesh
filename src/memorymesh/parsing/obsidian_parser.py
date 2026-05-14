"""Obsidian Markdown parser.

Extends the base Markdown parse with:

* YAML frontmatter extraction (between leading ``---`` delimiters).
* Wikilink backlink extraction via ``[[link]]`` and ``[[link|alias]]`` patterns.
* Filtering of embedded image links (``![[img]]``).

The parser is registered for ``.md`` files when a source is configured with
``source.type: obsidian``.  For regular Markdown sources the standard
:class:`~memorymesh.parsing.markdown.MarkdownParser` is used instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser
from memorymesh.parsing.text import _read_with_fallback

# Matches YAML frontmatter: opening ``---``, content, closing ``---`` or ``...``.
_FRONTMATTER_RE = re.compile(
    r"^---\r?\n(.*?)\r?\n(?:---|\.\.\.)(?:\r?\n|$)",
    re.DOTALL,
)

# Matches [[link]] or [[link|alias]] - but NOT ![[embedded]] images.
_WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\[\]|#]+?)(?:\|[^\[\]]*?)?\]\]")

# Used for very simple YAML key: value extraction (no nesting needed here).
_YAML_KV_RE = re.compile(r"^(\w[\w -]*?)\s*:\s*(.+)$", re.MULTILINE)

# YAML list items under a key: ``  - value``
_YAML_LIST_ITEM_RE = re.compile(r"^\s{2,}-\s+(.+)$", re.MULTILINE)

# Full YAML block for a specific key with multi-line list values.
_YAML_LIST_BLOCK_RE = re.compile(
    r"^(?P<key>\w[\w -]*?)\s*:\s*\n(?P<items>(?:\s+-\s+.+\n?)+)",
    re.MULTILINE,
)


def _parse_frontmatter(raw_yaml: str) -> dict[str, object]:
    """Extract key-value pairs and simple list values from raw YAML text.

    Supports two patterns:
    - Scalar:  ``key: value``
    - List block::

          tags:
            - foo
            - bar

    Args:
        raw_yaml: The raw YAML string between the ``---`` delimiters.

    Returns:
        Dict of extracted key -> value (str or list[str]).
    """
    result: dict[str, object] = {}

    # First pass: list blocks
    list_keys: set[str] = set()
    for m in _YAML_LIST_BLOCK_RE.finditer(raw_yaml):
        key = m.group("key").strip()
        items_str = m.group("items")
        items = [
            item_m.group(1).strip().strip("\"'")
            for item_m in _YAML_LIST_ITEM_RE.finditer(items_str)
        ]
        result[key] = items
        list_keys.add(key)

    # Second pass: scalar values (skip keys already handled as lists)
    for m in _YAML_KV_RE.finditer(raw_yaml):
        key = m.group(1).strip()
        if key in list_keys:
            continue
        value = m.group(2).strip().strip("\"'")
        # Inline list: ``tags: [foo, bar]``
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            result[key] = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()]
        else:
            result[key] = value

    return result


def _extract_backlinks(text: str) -> list[str]:
    """Return all wikilink targets found in *text*, excluding image embeds.

    Args:
        text: The full file content (including frontmatter, if present).

    Returns:
        Deduplicated list of link targets in order of first occurrence.
    """
    seen: set[str] = set()
    links: list[str] = []
    for m in _WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            links.append(target)
    return links


class ObsidianParser(Parser):
    """Parser for Obsidian-flavoured Markdown files.

    Reads ``.md`` files written by Obsidian, extracting YAML frontmatter
    metadata (``tags``, ``aliases``, ``created``, ``modified``, plus any
    custom fields) and ``[[wikilink]]`` backlinks.

    The body text returned is the *full* file content (frontmatter included),
    so that downstream chunkers receive the complete document.  Frontmatter
    and backlinks are surfaced in :attr:`~memorymesh.core.models.ParsedDocument.metadata`.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: Obsidian Markdown vaults."""
        return frozenset({".md", ".mdx", ".markdown"})

    def parse(self, path: Path) -> ParsedDocument:
        """Parse an Obsidian Markdown file.

        Args:
            path: Absolute path to the ``.md`` file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with:

            * ``text`` - full file content (frontmatter + body).
            * ``metadata["frontmatter"]`` - dict of extracted YAML fields.
            * ``metadata["tags"]`` - list of tags (from ``tags:`` field or inline ``#tag``).
            * ``metadata["aliases"]`` - list of aliases.
            * ``metadata["created"]`` - value of ``created:`` field if present.
            * ``metadata["modified"]`` - value of ``modified:`` field if present.
            * ``metadata["backlinks"]`` - list of ``[[wikilink]]`` targets.
        """
        text, encoding, error = _read_with_fallback(path)
        meta: dict[str, object] = {}

        if error:
            meta["error"] = error
            return ParsedDocument(
                path=path,
                text=text,
                file_type=".md",
                encoding=encoding,
                metadata=meta,
            )

        fm_match = _FRONTMATTER_RE.match(text)
        if fm_match:
            try:
                fm_data = _parse_frontmatter(fm_match.group(1))
                meta["frontmatter"] = fm_data

                tags_raw = fm_data.get("tags", [])
                if isinstance(tags_raw, str):
                    tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
                meta["tags"] = tags_raw

                aliases_raw = fm_data.get("aliases", [])
                if isinstance(aliases_raw, str):
                    aliases_raw = [aliases_raw]
                meta["aliases"] = aliases_raw

                for key in ("created", "modified"):
                    if key in fm_data:
                        meta[key] = fm_data[key]

            except Exception as exc:
                logger.warning(f"ObsidianParser: frontmatter parse error in {path.name!r}: {exc}")
                meta["frontmatter_error"] = str(exc)
        else:
            meta["frontmatter"] = {}
            meta["tags"] = []
            meta["aliases"] = []

        backlinks = _extract_backlinks(text)
        meta["backlinks"] = backlinks
        logger.debug(
            f"ObsidianParser: {path.name!r} "
            f"tags={len(meta.get('tags', []))} "  # type: ignore[arg-type]
            f"backlinks={len(backlinks)}"
        )

        return ParsedDocument(
            path=path,
            text=text,
            file_type=".md",
            encoding=encoding,
            metadata=meta,
        )
