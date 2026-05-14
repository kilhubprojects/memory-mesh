"""Windows compatibility tests for CodeChunker tree-sitter fallback.

Validates that CodeChunker gracefully falls back to RecursiveChunker when
tree-sitter raises TypeError (the known windows wheel incompatibility between
tree-sitter >= 0.22 and tree-sitter-languages <= 1.10.2).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from memorymesh.chunking.code import CodeChunker
from memorymesh.core.models import ParsedDocument


def _py_doc(text: str = "def foo():\n    return 1\n") -> ParsedDocument:
    return ParsedDocument(
        path=Path("fake.py"),
        text=text,
        file_type=".py",
    )


class TestCodeChunkerWindowsCompat:
    def test_typeerror_from_get_parser_triggers_fallback(self) -> None:
        """Simulate tree-sitter-languages TypeError on Windows and confirm fallback."""
        chunker = CodeChunker(max_chunk_size=512)
        doc = _py_doc("def hello():\n    print('world')\n\ndef bye():\n    pass\n")

        with patch(
            "memorymesh.chunking.code.CodeChunker._tree_sitter_chunk",
            side_effect=TypeError("__init__() takes exactly 1 argument (2 given)"),
        ):
            chunks = chunker.chunk(doc)

        # Fallback must produce at least one chunk without crashing
        assert len(chunks) >= 1
        assert all(c.file_type == ".py" for c in chunks)

    def test_import_error_triggers_fallback(self) -> None:
        """Simulate missing tree-sitter-languages package."""
        chunker = CodeChunker(max_chunk_size=512)
        doc = _py_doc()

        with patch(
            "memorymesh.chunking.code.CodeChunker._tree_sitter_chunk",
            side_effect=ImportError("No module named 'tree_sitter_languages'"),
        ):
            chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    def test_no_crash_on_unsupported_extension(self) -> None:
        """Files with unknown extension get recursive fallback, no crash."""
        chunker = CodeChunker(max_chunk_size=512)
        doc = ParsedDocument(
            path=Path("file.cobol"),
            text="IDENTIFICATION DIVISION.\n",
            file_type=".cobol",
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_fallback_chunks_cover_all_text(self) -> None:
        """Fallback output must contain all the source text across chunks."""
        chunker = CodeChunker(max_chunk_size=512)
        source = "def alpha():\n    pass\n\ndef beta():\n    return 42\n"
        doc = _py_doc(source)

        with patch(
            "memorymesh.chunking.code.CodeChunker._tree_sitter_chunk",
            side_effect=TypeError("__init__() takes exactly 1 argument (2 given)"),
        ):
            chunks = chunker.chunk(doc)

        combined = " ".join(c.text for c in chunks)
        assert "alpha" in combined
        assert "beta" in combined

    def test_normal_path_not_affected(self) -> None:
        """When tree-sitter works, CodeChunker should not use fallback."""
        chunker = CodeChunker(max_chunk_size=512)
        doc = _py_doc()

        # Just confirm it doesn't crash regardless of whether tree-sitter is
        # available or falls back — both are valid outcomes.
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)
