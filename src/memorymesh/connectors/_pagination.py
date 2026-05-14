"""Shared pagination helpers for MemoryMesh connectors.

Two patterns are supported:

- **Page-based**: ``fetcher(page)`` is called with ``page=1, 2, 3…`` until it
  returns an empty list or *max_pages* is reached.
- **Cursor-based**: ``fetcher(cursor)`` is called with ``None`` on the first
  call and with the cursor string returned by the previous call; iteration
  stops when the fetcher returns ``(items, None)``.

Both helpers ``yield`` individual ``dict`` items so connectors can process
results lazily without loading all pages into memory at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator


def paginate_pages(
    fetcher: Callable[[int], list[dict]],
    max_pages: int = 50,
) -> Iterator[dict]:
    """Iterate page-based API results.

    Calls ``fetcher(page)`` starting at ``page=1``.  Stops when the
    response is empty or *max_pages* pages have been consumed.

    Args:
        fetcher: Callable that accepts an integer page number and returns a
            list of result dicts.  An empty list signals the last page.
        max_pages: Safety cap on the number of pages to fetch.

    Yields:
        Individual result dicts from each page, in order.

    Example::

        def _fetch(page: int) -> list[dict]:
            return api_get(f"/items?page={page}&per_page=100")

        for item in paginate_pages(_fetch, max_pages=20):
            process(item)
    """
    for page in range(1, max_pages + 1):
        batch = fetcher(page)
        if not batch:
            break
        yield from batch


def paginate_cursor(
    fetcher: Callable[[str | None], tuple[list[dict], str | None]],
    max_pages: int = 50,
) -> Iterator[dict]:
    """Iterate cursor-based API results.

    Calls ``fetcher(cursor)`` with ``cursor=None`` on the first request.
    Stops when the fetcher returns an empty cursor or *max_pages* calls
    have been made.

    Args:
        fetcher: Callable that accepts an optional cursor string and returns
            a ``(items, next_cursor)`` tuple.  A ``None`` *next_cursor*
            signals the final page.
        max_pages: Safety cap on the number of pages to fetch.

    Yields:
        Individual result dicts from each page, in order.

    Example::

        def _fetch(cursor: str | None) -> tuple[list[dict], str | None]:
            data = api_get("/items", params={"cursor": cursor})
            return data["items"], data.get("next_cursor")

        for item in paginate_cursor(_fetch):
            process(item)
    """
    cursor: str | None = None
    for _ in range(max_pages):
        items, next_cursor = fetcher(cursor)
        yield from items
        if not next_cursor:
            break
        cursor = next_cursor
