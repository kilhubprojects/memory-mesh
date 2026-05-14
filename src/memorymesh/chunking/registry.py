"""Chunker registry — selects the right chunker for a given file type.

Usage::

    from memorymesh.chunking.registry import get_chunker

    chunker = get_chunker(".md", chunking_config=config.chunking)
    chunks = chunker.chunk(doc)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memorymesh.chunking.base import Chunker
from memorymesh.chunking.code import CodeChunker
from memorymesh.chunking.markdown import MarkdownChunker
from memorymesh.chunking.recursive import RecursiveChunker

if TYPE_CHECKING:
    from memorymesh.core.models import ChunkingConfig

_MARKDOWN_EXTENSIONS: frozenset[str] = frozenset({".md", ".mdx", ".markdown"})

_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
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
    }
)


def get_chunker(
    extension: str,
    chunking_config: ChunkingConfig | None = None,
) -> Chunker:
    """Return the appropriate chunker for *extension*.

    Priority:
    1. Markdown extensions → :class:`~memorymesh.chunking.markdown.MarkdownChunker`
    2. Code extensions → :class:`~memorymesh.chunking.code.CodeChunker` (with tree-sitter fallback)
    3. Everything else → :class:`~memorymesh.chunking.recursive.RecursiveChunker`

    Args:
        extension: Lowercase file extension including the dot (e.g. ``".py"``).
        chunking_config: Config from ``config.yaml``.  Uses defaults when ``None``.

    Returns:
        A :class:`~memorymesh.chunking.base.Chunker` instance.
    """
    ext = extension.lower()

    if chunking_config is not None:
        md_cfg = chunking_config.markdown
        code_cfg = chunking_config.code
        default_cfg = chunking_config.default
    else:
        from memorymesh.core.models import (
            CodeChunkingConfig,
            MarkdownChunkingConfig,
            RecursiveChunkingConfig,
        )

        md_cfg = MarkdownChunkingConfig()
        code_cfg = CodeChunkingConfig()
        default_cfg = RecursiveChunkingConfig()

    if ext in _MARKDOWN_EXTENSIONS:
        return MarkdownChunker(max_chunk_size=md_cfg.max_chunk_size)

    if ext in _CODE_EXTENSIONS:
        return CodeChunker(max_chunk_size=code_cfg.max_chunk_size)

    return RecursiveChunker(
        chunk_size=default_cfg.chunk_size,
        chunk_overlap=default_cfg.chunk_overlap,
    )
