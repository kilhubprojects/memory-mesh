"""Unit tests for memorymesh.search.context.expand_context."""

from __future__ import annotations

from memorymesh.search.context import expand_context


def test_expand_context_returns_wider_window(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("AAAA BBBB CCCC DDDD EEEE", encoding="utf-8")
    result = expand_context(str(f), start_char=5, end_char=9, window=5)
    assert "AAAA" in result
    assert "BBBB" in result
    assert "CCCC" in result


def test_expand_context_handles_missing_file():
    result = expand_context("/nonexistent/file.txt", 0, 10, 100)
    assert result == ""


def test_expand_context_clamps_to_file_boundaries(tmp_path):
    f = tmp_path / "short.txt"
    text = "hello"
    f.write_text(text, encoding="utf-8")
    result = expand_context(str(f), start_char=1, end_char=4, window=9999)
    assert result == text


def test_expand_context_exact_slice(tmp_path):
    f = tmp_path / "slice.txt"
    f.write_text("0123456789", encoding="utf-8")
    result = expand_context(str(f), start_char=4, end_char=6, window=2)
    assert result == "234567"
