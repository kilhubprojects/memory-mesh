"""AST-aware code chunker using tree-sitter.

Splits source code at top-level function and class definitions.  Falls back
to :class:`~memorymesh.chunking.recursive.RecursiveChunker` when:
- ``tree-sitter-languages`` is not installed.
- The language is not supported by the installed grammar pack.
- Any tree-sitter error occurs at runtime.

The ``language``, ``function_name``, and ``class_name`` metadata fields are
populated so search hits can reference the exact symbol.
"""

from __future__ import annotations

import os

from loguru import logger

from memorymesh.chunking.base import Chunker
from memorymesh.chunking.recursive import RecursiveChunker
from memorymesh.core.models import Chunk, ChunkMetadata, ParsedDocument

# Maps lowercase file extension -> tree-sitter language name
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
}

# Node types that represent top-level chunks worth splitting on
_CHUNK_NODE_TYPES: frozenset[str] = frozenset(
    {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "class_definition",
        "class_declaration",
        "impl_item",  # Rust
        "function_item",  # Rust
    }
)


class CodeChunker(Chunker):
    """Tree-sitter AST-aware code chunker with recursive fallback.

    Args:
        max_chunk_size: Nodes larger than this are split further by the
            :class:`~memorymesh.chunking.recursive.RecursiveChunker`.
    """

    def __init__(self, max_chunk_size: int = 1024) -> None:
        self._max_chunk_size = max_chunk_size
        self._fallback = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=50)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split *doc* by AST boundaries, falling back to recursive splitting.

        Args:
            doc: Parsed source code document.

        Returns:
            List of :class:`~memorymesh.core.models.Chunk` objects.
        """
        ext = doc.file_type.lower()
        lang_name = _EXT_TO_LANG.get(ext)

        if lang_name is None:
            logger.debug(f"CodeChunker: no tree-sitter grammar for {ext!r}; using fallback")
            return self._fallback.chunk(doc)

        try:
            return self._tree_sitter_chunk(doc, lang_name)
        except Exception as exc:
            logger.warning(
                f"CodeChunker: tree-sitter failed for {doc.path.name!r} ({lang_name}): {exc}; "
                "falling back to RecursiveChunker"
            )
            return self._fallback.chunk(doc)

    def _tree_sitter_chunk(self, doc: ParsedDocument, lang_name: str) -> list[Chunk]:
        from tree_sitter_languages import get_parser  # type: ignore[import-untyped]

        parser = get_parser(lang_name)

        source_bytes = doc.text.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        top_level_nodes = [child for child in root.children if child.type in _CHUNK_NODE_TYPES]

        if not top_level_nodes:
            # No recognised top-level nodes - split the whole file recursively
            return self._fallback.chunk(doc)

        chunks: list[Chunk] = []
        idx = 0
        mtime = float(os.path.getmtime(doc.path)) if doc.path.exists() else 0.0

        for node in top_level_nodes:
            start = node.start_byte
            end = node.end_byte
            node_text = source_bytes[start:end].decode("utf-8", errors="replace")
            func_name, class_name = _extract_names(node, source_bytes)

            if len(node_text) <= self._max_chunk_size:
                chunks.append(
                    Chunk(
                        path=doc.path,
                        chunk_index=idx,
                        text=node_text,
                        start_char=start,
                        end_char=end,
                        file_type=doc.file_type,
                        mtime=mtime,
                        source_root="",
                        metadata=ChunkMetadata(
                            language=lang_name,
                            function_name=func_name,
                            class_name=class_name,
                        ),
                    )
                )
                idx += 1
            else:
                synthetic = ParsedDocument(
                    path=doc.path,
                    text=node_text,
                    file_type=doc.file_type,
                    encoding=doc.encoding,
                )
                sub_chunks = self._fallback.chunk(synthetic)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            path=doc.path,
                            chunk_index=idx,
                            text=sub.text,
                            start_char=start + sub.start_char,
                            end_char=start + sub.end_char,
                            file_type=doc.file_type,
                            mtime=mtime,
                            source_root="",
                            metadata=ChunkMetadata(
                                language=lang_name,
                                function_name=func_name,
                                class_name=class_name,
                            ),
                        )
                    )
                    idx += 1

        # Collect any top-level code not inside a recognised node
        residual_parts: list[str] = []
        prev_end = 0
        for node in sorted(top_level_nodes, key=lambda n: n.start_byte):
            gap = source_bytes[prev_end : node.start_byte].decode("utf-8", errors="replace").strip()
            if gap:
                residual_parts.append(gap)
            prev_end = node.end_byte
        tail = source_bytes[prev_end:].decode("utf-8", errors="replace").strip()
        if tail:
            residual_parts.append(tail)

        for part in residual_parts:
            if part.strip():
                chunks.append(
                    Chunk(
                        path=doc.path,
                        chunk_index=idx,
                        text=part,
                        start_char=doc.text.find(part),
                        end_char=doc.text.find(part) + len(part),
                        file_type=doc.file_type,
                        mtime=mtime,
                        source_root="",
                        metadata=ChunkMetadata(language=lang_name),
                    )
                )
                idx += 1

        return chunks if chunks else self._fallback.chunk(doc)


def _extract_names(node: object, source_bytes: bytes) -> tuple[str | None, str | None]:
    """Try to extract function and class names from a tree-sitter node."""
    try:
        node_type: str = node.type  # type: ignore[attr-defined]
        children = node.children  # type: ignore[attr-defined]

        func_name: str | None = None
        class_name: str | None = None

        name_node = next(
            (c for c in children if c.type == "identifier"),
            None,
        )
        if name_node is not None:
            raw_name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if "class" in node_type:
                class_name = raw_name
            else:
                func_name = raw_name

        return func_name, class_name
    except Exception:
        return None, None
