"""Unit tests for the parsing layer."""

from __future__ import annotations

from pathlib import Path

from memorymesh.parsing.markdown import MarkdownParser
from memorymesh.parsing.registry import build_registry, get_parser
from memorymesh.parsing.text import TextParser


class TestTextParser:
    def test_parse_returns_text(self, sample_text_file: Path) -> None:
        doc = TextParser().parse(sample_text_file)
        assert "Hello world" in doc.text
        assert doc.file_type == ".txt"

    def test_parse_sets_encoding(self, sample_text_file: Path) -> None:
        doc = TextParser().parse(sample_text_file)
        assert doc.encoding in ("utf-8", "utf-8-sig")

    def test_parse_path_matches(self, sample_text_file: Path) -> None:
        doc = TextParser().parse(sample_text_file)
        assert doc.path == sample_text_file

    def test_parse_py_file(self, sample_py_file: Path) -> None:
        doc = TextParser().parse(sample_py_file)
        assert "def greet" in doc.text
        assert doc.file_type == ".py"

    def test_parse_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost.txt"
        doc = TextParser().parse(ghost)
        assert doc.text == ""
        assert "error" in doc.metadata

    def test_supported_extensions_includes_py_and_txt(self) -> None:
        exts = TextParser().supported_extensions
        assert ".py" in exts
        assert ".txt" in exts


class TestMarkdownParser:
    def test_parse_returns_markdown_text(self, sample_markdown_file: Path) -> None:
        doc = MarkdownParser().parse(sample_markdown_file)
        assert "Introduction" in doc.text
        assert doc.file_type == ".md"

    def test_supported_extensions(self) -> None:
        exts = MarkdownParser().supported_extensions
        assert ".md" in exts
        assert ".mdx" in exts

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        doc = MarkdownParser().parse(p)
        assert doc.text == ""
        assert doc.file_type == ".md"


class TestPdfParser:
    def test_parse_pdf(self, sample_pdf_path: Path) -> None:
        from memorymesh.parsing.pdf import PdfParser

        doc = PdfParser().parse(sample_pdf_path)
        assert doc.file_type == ".pdf"
        assert "page_count" in doc.metadata

    def test_parse_pdf_text_contains_fixture_text(self, sample_pdf_path: Path) -> None:
        from memorymesh.parsing.pdf import PdfParser

        doc = PdfParser().parse(sample_pdf_path)
        assert "MemoryMesh" in doc.text or doc.text == ""  # text may be empty for image PDFs

    def test_parse_corrupt_pdf_does_not_raise(self, tmp_path: Path) -> None:
        from memorymesh.parsing.pdf import PdfParser

        p = tmp_path / "corrupt.pdf"
        p.write_bytes(b"not a real pdf")
        doc = PdfParser().parse(p)
        assert doc.text == ""
        assert "error" in doc.metadata

    def test_ocr_not_triggered_when_disabled(self, sample_pdf_path: Path) -> None:
        from memorymesh.core.models import OcrConfig
        from memorymesh.parsing.pdf import PdfParser

        cfg = OcrConfig(enabled=False)
        doc = PdfParser(ocr_config=cfg).parse(sample_pdf_path)
        assert doc.metadata.get("ocr") is not True


class TestDocxParser:
    def test_parse_docx(self, sample_docx_path: Path) -> None:
        from memorymesh.parsing.docx import DocxParser

        doc = DocxParser().parse(sample_docx_path)
        assert "MemoryMesh" in doc.text
        assert doc.file_type == ".docx"

    def test_paragraph_count_in_metadata(self, sample_docx_path: Path) -> None:
        from memorymesh.parsing.docx import DocxParser

        doc = DocxParser().parse(sample_docx_path)
        assert doc.metadata.get("paragraph_count", 0) >= 1

    def test_corrupt_docx_does_not_raise(self, tmp_path: Path) -> None:
        from memorymesh.parsing.docx import DocxParser

        p = tmp_path / "corrupt.docx"
        p.write_bytes(b"not a docx")
        doc = DocxParser().parse(p)
        assert doc.text == ""
        assert "error" in doc.metadata


class TestRegistry:
    def test_get_parser_txt(self) -> None:
        assert get_parser(".txt") is not None

    def test_get_parser_pdf(self) -> None:
        assert get_parser(".pdf") is not None

    def test_get_parser_md(self) -> None:
        assert get_parser(".md") is not None

    def test_get_parser_docx(self) -> None:
        assert get_parser(".docx") is not None

    def test_get_parser_unsupported_returns_none(self) -> None:
        assert get_parser(".exe") is None

    def test_case_insensitive(self) -> None:
        assert get_parser(".PDF") is not None

    def test_build_registry_covers_common_extensions(self) -> None:
        reg = build_registry()
        for ext in (".py", ".txt", ".md", ".pdf", ".docx"):
            assert ext in reg, f"{ext!r} missing from registry"
