"""Plain-text and source-code parser.

Handles any file that can be read as UTF-8 text.  Encoding detection tries
utf-8 → utf-8-sig → cp1252 → latin-1 in order; the first codec that decodes
without error is used.  The daemon never crashes due to encoding issues.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".sql",
        ".r",
        ".m",
        ".tex",
        ".rst",
        ".csv",
        ".tsv",
        ".log",
    }
)


class TextParser(Parser):
    """Parser for plain text and source code files."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: plain-text and source-code files."""
        return _TEXT_EXTENSIONS

    def parse(self, path: Path) -> ParsedDocument:
        """Read and return the file's text content.

        Args:
            path: Absolute path to the text file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with raw text.
        """
        file_type = path.suffix.lower() or ".txt"
        text, encoding, error = _read_with_fallback(path)
        meta: dict[str, object] = {}
        if error:
            meta["error"] = error
        return ParsedDocument(
            path=path,
            text=text,
            file_type=file_type,
            encoding=encoding,
            metadata=meta,
        )


def _read_with_fallback(path: Path) -> tuple[str, str, str]:
    """Try reading *path* with several encodings.

    Returns:
        ``(text, encoding_used, error_message)`` — error is empty on success.
    """
    last_exc: Exception | None = None
    for enc in _ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
            logger.debug(f"TextParser: read {path.name!r} as {enc}")
            return text, enc, ""
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_exc = exc
        except OSError as exc:
            logger.warning(f"TextParser: OS error reading {path}: {exc}")
            return "", "utf-8", str(exc)

    msg = f"Could not decode {path} with any of {_ENCODINGS}: {last_exc}"
    logger.warning(msg)
    return "", "utf-8", msg
