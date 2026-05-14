"""Unit tests for the chunking layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorymesh.chunking.markdown import MarkdownChunker
from memorymesh.chunking.recursive import RecursiveChunker
from memorymesh.chunking.registry import get_chunker
from memorymesh.core.models import ParsedDocument


def _doc(
    text: str,
    file_type: str = ".txt",
    path: str = "/tmp/test.txt",
) -> ParsedDocument:
    return ParsedDocument(path=Path(path), text=text, file_type=file_type)


class TestRecursiveChunker:
    def test_short_text_produces_one_chunk(self) -> None:
        doc = _doc("Hello world.")
        chunks = RecursiveChunker(chunk_size=512).chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_empty_text_produces_one_chunk(self) -> None:
        doc = _doc("")
        chunks = RecursiveChunker().chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_long_text_is_split(self) -> None:
        text = "word " * 200  # 1000 chars
        doc = _doc(text)
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=10).chunk(doc)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self) -> None:
        text = "paragraph one.\n\n" * 20
        chunks = RecursiveChunker(chunk_size=50, chunk_overlap=5).chunk(doc=_doc(text))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_chunk_path_matches_doc(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("content", encoding="utf-8")
        doc = ParsedDocument(path=p, text="content", file_type=".txt")
        chunks = RecursiveChunker().chunk(doc)
        for c in chunks:
            assert c.path == p

    def test_chunk_size_respected(self) -> None:
        text = "a" * 1000
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0).chunk(_doc(text))
        for c in chunks:
            assert len(c.text) <= 110  # allow small overrun at boundary

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=50, chunk_overlap=50)


class TestMarkdownChunker:
    def test_splits_at_headings(self) -> None:
        text = "# Heading One\n\nContent one.\n\n## Heading Two\n\nContent two.\n"
        chunks = MarkdownChunker().chunk(_doc(text, ".md"))
        assert len(chunks) == 2

    def test_heading_path_populated(self) -> None:
        text = "# Top\n\nBody.\n\n## Sub\n\nSub body.\n"
        chunks = MarkdownChunker().chunk(_doc(text, ".md"))
        top = next(c for c in chunks if "Top" in c.text)
        assert top.metadata.heading_path is not None
        assert "Top" in top.metadata.heading_path

    def test_empty_text_returns_single_chunk(self) -> None:
        chunks = MarkdownChunker().chunk(_doc("", ".md"))
        assert len(chunks) == 1

    def test_no_headings_returns_single_chunk(self) -> None:
        text = "Just plain text without any headings."
        chunks = MarkdownChunker().chunk(_doc(text, ".md"))
        assert len(chunks) == 1

    def test_oversized_section_is_split_further(self) -> None:
        # section > max_chunk_size → recursive fallback
        body = "word " * 200  # ~1000 chars
        text = f"# Big Section\n\n{body}"
        chunks = MarkdownChunker(max_chunk_size=200).chunk(_doc(text, ".md"))
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self) -> None:
        text = "\n\n".join(f"# H{i}\n\nBody {i}." for i in range(5))
        chunks = MarkdownChunker().chunk(_doc(text, ".md"))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestCodeChunker:
    def test_fallback_when_no_grammar(self) -> None:
        from memorymesh.chunking.code import CodeChunker

        doc = _doc("print('hello')", ".unknownlang")
        chunks = CodeChunker().chunk(doc)
        assert len(chunks) >= 1

    def test_python_file_uses_fallback_or_ast(self, sample_py_file: Path) -> None:
        from memorymesh.chunking.code import CodeChunker

        doc = ParsedDocument(
            path=sample_py_file,
            text=sample_py_file.read_text(encoding="utf-8"),
            file_type=".py",
        )
        chunks = CodeChunker().chunk(doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.file_type == ".py"

    def test_empty_code_produces_chunk(self) -> None:
        from memorymesh.chunking.code import CodeChunker

        chunks = CodeChunker().chunk(_doc("", ".py"))
        assert len(chunks) >= 1


class TestRegistry:
    def test_md_gets_markdown_chunker(self) -> None:
        from memorymesh.chunking.markdown import MarkdownChunker

        assert isinstance(get_chunker(".md"), MarkdownChunker)

    def test_py_gets_code_chunker(self) -> None:
        from memorymesh.chunking.code import CodeChunker

        assert isinstance(get_chunker(".py"), CodeChunker)

    def test_txt_gets_recursive_chunker(self) -> None:
        assert isinstance(get_chunker(".txt"), RecursiveChunker)

    def test_unknown_ext_gets_recursive_chunker(self) -> None:
        assert isinstance(get_chunker(".xyz"), RecursiveChunker)

    def test_uppercase_extension_normalised(self) -> None:
        from memorymesh.chunking.markdown import MarkdownChunker

        assert isinstance(get_chunker(".MD"), MarkdownChunker)
