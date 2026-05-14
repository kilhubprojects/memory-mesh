"""Parent Document Retriever helper.

Expands a narrow chunk window to a wider context slice so the MCP client
receives more surrounding text without changing the dense index.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def expand_context(
    path: str,
    start_char: int,
    end_char: int,
    window: int,
) -> str:
    """Read a wider context window around ``[start_char, end_char]`` from disk.

    Args:
        path: Absolute path to the source file.
        start_char: Start character offset of the original chunk.
        end_char: End character offset of the original chunk.
        window: Number of extra characters to include before and after the
            chunk boundaries.

    Returns:
        Extended text string containing the chunk plus up to *window* extra
        characters on each side.  Returns an empty string if the file cannot
        be read or does not exist.
    """
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        lo = max(0, start_char - window)
        hi = min(len(content), end_char + window)
        return content[lo:hi]
    except OSError as exc:
        logger.debug(f"expand_context: could not read {path!r}: {exc}")
        return ""
