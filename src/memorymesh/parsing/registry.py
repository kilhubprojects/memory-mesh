"""Parser registry — maps file extensions to parser instances.

Usage::

    from memorymesh.parsing.registry import get_parser

    parser = get_parser(".pdf", ocr_config=config.ocr)
    if parser is None:
        # unsupported extension
        ...
    doc = parser.parse(path)

Source-type-specific parsers (Obsidian, Notion, Conversations, Email,
Calendar, Browser history) are registered in addition to the default set.
Their extensions overlap with the generic parsers (e.g. ``.md``, ``.html``,
``.json``, ``.db``).  When ``source_type`` is provided to
:func:`build_registry` or :func:`get_parser`, the source-type-specific parser
takes precedence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memorymesh.parsing.base import Parser
from memorymesh.parsing.browser_history_parser import BrowserHistoryParser
from memorymesh.parsing.calendar_parser import CalendarParser
from memorymesh.parsing.conversation_parser import ConversationParser
from memorymesh.parsing.docx import DocxParser
from memorymesh.parsing.email_parser import EmailParser
from memorymesh.parsing.markdown import MarkdownParser
from memorymesh.parsing.notion_parser import NotionParser
from memorymesh.parsing.obsidian_parser import ObsidianParser
from memorymesh.parsing.pdf import PdfParser
from memorymesh.parsing.text import TextParser

if TYPE_CHECKING:
    from memorymesh.core.models import EmailSourceConfig, OcrConfig

# Maps ``source.type`` values to the parser class that handles that source.
_SOURCE_TYPE_PARSERS: dict[str, type[Parser]] = {
    "obsidian": ObsidianParser,
    "notion": NotionParser,
    "conversations": ConversationParser,
    "email": EmailParser,
    "calendar": CalendarParser,
    "browser_history": BrowserHistoryParser,
}


def build_registry(
    ocr_config: OcrConfig | None = None,
    source_type: str | None = None,
    email_config: EmailSourceConfig | None = None,
) -> dict[str, Parser]:
    """Create a mapping from lowercase extension → parser instance.

    When *source_type* is given, the corresponding specialised parser is added
    to the registry and will shadow any generic parser for overlapping
    extensions.

    Args:
        ocr_config: OCR settings forwarded to :class:`~memorymesh.parsing.pdf.PdfParser`.
        source_type: Optional ``source.type`` value from config.  When provided,
            the matching specialised parser is instantiated and its extensions
            override the defaults.
        email_config: Optional email configuration forwarded to
            :class:`~memorymesh.parsing.email_parser.EmailParser` when
            ``source_type="email"``.

    Returns:
        Dict mapping ``".ext"`` strings to parser singletons.
    """
    parsers: list[Parser] = [
        TextParser(),
        MarkdownParser(),
        PdfParser(ocr_config=ocr_config),
        DocxParser(),
        CalendarParser(),
        EmailParser(),
        BrowserHistoryParser(),
    ]

    registry: dict[str, Parser] = {}
    for parser in parsers:
        for ext in parser.supported_extensions:
            registry[ext] = parser

    # Source-type override: specialised parser shadows the generic one.
    if source_type and source_type in _SOURCE_TYPE_PARSERS:
        parser_cls = _SOURCE_TYPE_PARSERS[source_type]

        if source_type == "email" and email_config is not None:
            specialised: Parser = EmailParser(max_messages=email_config.max_messages)
        else:
            specialised = parser_cls()  # type: ignore[assignment]

        for ext in specialised.supported_extensions:
            registry[ext] = specialised

    return registry


def get_parser(
    extension: str,
    ocr_config: OcrConfig | None = None,
    source_type: str | None = None,
    *,
    _registry: dict[str, Parser] | None = None,
) -> Parser | None:
    """Return the parser for *extension*, or ``None`` if unsupported.

    Args:
        extension: Lowercase file extension including the dot (e.g. ``".pdf"``).
        ocr_config: Forwarded to :func:`build_registry` if no cached registry is provided.
        source_type: Optional ``source.type`` value for specialised parser selection.
        _registry: Pre-built registry (avoids re-instantiation in hot paths).

    Returns:
        A :class:`~memorymesh.parsing.base.Parser` or ``None``.
    """
    reg = _registry if _registry is not None else build_registry(ocr_config, source_type)
    return reg.get(extension.lower())
