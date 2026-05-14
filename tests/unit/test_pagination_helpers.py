"""Unit tests for memorymesh.connectors._pagination."""

from __future__ import annotations

import pytest

from memorymesh.connectors._pagination import paginate_cursor, paginate_pages


class TestPaginatePages:
    def test_yields_all_items_across_pages(self) -> None:
        pages = [[{"id": 1}, {"id": 2}], [{"id": 3}], []]
        calls: list[int] = []

        def fetcher(page: int) -> list[dict]:
            calls.append(page)
            return pages[page - 1] if page <= len(pages) else []

        result = list(paginate_pages(fetcher, max_pages=10))
        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert calls == [1, 2, 3]

    def test_stops_on_empty_page(self) -> None:
        def fetcher(page: int) -> list[dict]:
            return [{"x": page}] if page <= 2 else []

        result = list(paginate_pages(fetcher, max_pages=100))
        assert len(result) == 2

    def test_respects_max_pages(self) -> None:
        def fetcher(page: int) -> list[dict]:
            return [{"x": page}]

        result = list(paginate_pages(fetcher, max_pages=3))
        assert len(result) == 3

    def test_empty_first_page_returns_nothing(self) -> None:
        result = list(paginate_pages(lambda p: [], max_pages=5))
        assert result == []

    def test_error_in_fetcher_propagates(self) -> None:
        def fetcher(page: int) -> list[dict]:
            raise ValueError("network error")

        with pytest.raises(ValueError, match="network error"):
            list(paginate_pages(fetcher, max_pages=5))


class TestPaginateCursor:
    def test_yields_all_items_with_cursor_chain(self) -> None:
        responses = [
            ([{"a": 1}], "cursor2"),
            ([{"a": 2}], "cursor3"),
            ([{"a": 3}], None),
        ]
        call_cursors: list[str | None] = []

        def fetcher(cursor: str | None) -> tuple[list[dict], str | None]:
            call_cursors.append(cursor)
            return responses[len(call_cursors) - 1]

        result = list(paginate_cursor(fetcher, max_pages=10))
        assert result == [{"a": 1}, {"a": 2}, {"a": 3}]
        assert call_cursors == [None, "cursor2", "cursor3"]

    def test_stops_when_cursor_is_none(self) -> None:
        def fetcher(cursor: str | None) -> tuple[list[dict], str | None]:
            return [{"x": 1}], None

        result = list(paginate_cursor(fetcher, max_pages=10))
        assert result == [{"x": 1}]

    def test_respects_max_pages_cap(self) -> None:
        def fetcher(cursor: str | None) -> tuple[list[dict], str | None]:
            return [{"x": 1}], "next"

        result = list(paginate_cursor(fetcher, max_pages=4))
        assert len(result) == 4

    def test_empty_first_batch_yields_nothing(self) -> None:
        result = list(paginate_cursor(lambda c: ([], None), max_pages=5))
        assert result == []
