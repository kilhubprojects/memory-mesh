"""Unit tests for Tier 1 MemoryMesh connectors.

All I/O (filesystem, SQLite, HTTP) is mocked - no real external resources are
needed to run these tests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import textwrap
import time
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest


class TestHtmlToText:
    def test_strips_tags(self) -> None:
        from memorymesh.connectors._html import html_to_text

        assert html_to_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self) -> None:
        from memorymesh.connectors._html import html_to_text

        assert "&amp;" not in html_to_text("A &amp; B")
        assert html_to_text("A &amp; B") == "A & B"

    def test_normalises_whitespace(self) -> None:
        from memorymesh.connectors._html import html_to_text

        result = html_to_text("<p>one</p>   <p>two</p>")
        assert "  " not in result

    def test_empty_string_returns_empty(self) -> None:
        from memorymesh.connectors._html import html_to_text

        assert html_to_text("") == ""

    def test_plain_text_unchanged(self) -> None:
        from memorymesh.connectors._html import html_to_text

        assert html_to_text("just text") == "just text"


class TestBrowserHistoryConnector:
    def _make_chrome_db(self, path: Path) -> None:
        """Create a minimal Chrome-style history SQLite in *path*."""
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE urls "
            "(id INTEGER PRIMARY KEY, url TEXT, title TEXT, "
            "visit_count INTEGER, last_visit_time INTEGER)"
        )
        # Timestamp: now in Chrome micros
        from memorymesh.connectors.browser_history_connector import _unix_to_chrome_micros

        now_micros = _unix_to_chrome_micros(time.time())
        conn.execute(
            "INSERT INTO urls VALUES (1, 'https://example.com', 'Example', 5, ?)",
            (now_micros,),
        )
        conn.execute(
            "INSERT INTO urls VALUES (2, 'https://python.org', 'Python', 2, ?)",
            (now_micros - 1000,),
        )
        conn.commit()
        conn.close()

    def _make_firefox_db(self, path: Path) -> None:
        """Create a minimal Firefox-style places.sqlite in *path*."""
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE moz_places "
            "(id INTEGER PRIMARY KEY, url TEXT, title TEXT, "
            "visit_count INTEGER, last_visit_date INTEGER)"
        )
        now_micros = int(time.time() * 1_000_000)
        conn.execute(
            "INSERT INTO moz_places VALUES (1, 'https://firefox.com', 'Firefox', 3, ?)",
            (now_micros,),
        )
        conn.commit()
        conn.close()

    def test_chrome_yields_documents(self, tmp_path: Path) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
        )

        db = tmp_path / "History"
        self._make_chrome_db(db)

        cfg = BrowserHistoryConfig(browser="chrome", profile_path=db, days_past=1)
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())

        assert len(docs) == 2
        assert all(d.file_type == ".url" for d in docs)
        assert any("example.com" in d.metadata["url"] for d in docs)

    def test_firefox_yields_documents(self, tmp_path: Path) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
        )

        db = tmp_path / "places.sqlite"
        self._make_firefox_db(db)

        cfg = BrowserHistoryConfig(browser="firefox", profile_path=db, days_past=1)
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())

        assert len(docs) == 1
        assert docs[0].metadata["browser"] == "firefox"
        assert docs[0].file_type == ".url"

    def test_missing_db_yields_nothing(self) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
        )

        cfg = BrowserHistoryConfig(
            browser="chrome",
            profile_path=Path("/nonexistent/path/History"),
        )
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())
        assert docs == []

    def test_max_urls_respected(self, tmp_path: Path) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
        )

        db = tmp_path / "History"
        self._make_chrome_db(db)

        cfg = BrowserHistoryConfig(browser="chrome", profile_path=db, max_urls=1)
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())
        assert len(docs) == 1

    def test_synthetic_path_contains_hash(self, tmp_path: Path) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
            _url_hash,
        )

        db = tmp_path / "History"
        self._make_chrome_db(db)

        cfg = BrowserHistoryConfig(browser="chrome", profile_path=db)
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())
        url = docs[0].metadata["url"]
        expected_hash = _url_hash(url)
        assert expected_hash in str(docs[0].path)

    def test_epoch_conversion(self) -> None:
        from memorymesh.connectors.browser_history_connector import (
            _chrome_ts_to_unix,
            _unix_to_chrome_micros,
        )

        unix_ts = 1_700_000_000.0
        chrome_micros = _unix_to_chrome_micros(unix_ts)
        recovered = _chrome_ts_to_unix(chrome_micros)
        assert abs(recovered - unix_ts) < 1.0

    def test_text_contains_title_and_url(self, tmp_path: Path) -> None:
        from memorymesh.connectors.browser_history_connector import (
            BrowserHistoryConfig,
            BrowserHistoryConnector,
        )

        db = tmp_path / "History"
        self._make_chrome_db(db)

        cfg = BrowserHistoryConfig(browser="chrome", profile_path=db)
        docs = list(BrowserHistoryConnector(cfg).fetch_documents())
        assert any("Example" in d.text and "https://example.com" in d.text for d in docs)


_CLIPPINGS_SAMPLE = textwrap.dedent("""\
==========
The Pragmatic Programmer (David Thomas, Andrew Hunt)

Your Highlight on page 42 | location 650-655 | Added on Monday, January 1, 2024 10:00:00 AM

Orthogonality reduces the risk involved in any change.

==========
Clean Code (Robert C. Martin)

Your Highlight on page 7 | location 102-104 | Added on Tuesday, January 2, 2024 09:30:00 AM

Functions should do one thing.

==========
""")

_CLIPPINGS_WITH_NOTE = textwrap.dedent("""\
==========
Some Book (Some Author)

Your Note on page 5 | location 80 | Added on Wednesday, January 3, 2024 08:00:00 AM

This is a note, not a highlight.

==========
""")

_CLIPPINGS_EMPTY_TEXT = textwrap.dedent("""\
==========
Empty Book (No Author)

Your Bookmark on page 1 | location 1 | Added on Thursday, January 4, 2024 07:00:00 AM


==========
""")


class TestKindleHighlightsConnector:
    def test_yields_one_doc_per_highlight(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_SAMPLE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        assert len(docs) == 2

    def test_file_type_is_kindle(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_SAMPLE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        assert all(d.file_type == ".kindle" for d in docs)

    def test_metadata_contains_title_and_author(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_SAMPLE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        titles = {d.metadata["title"] for d in docs}
        assert "The Pragmatic Programmer" in titles

    def test_highlight_text_in_document(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_SAMPLE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        combined = " ".join(d.text for d in docs)
        assert "Orthogonality" in combined
        assert "one thing" in combined

    def test_missing_file_yields_nothing(self) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        docs = list(
            KindleHighlightsConnector(
                KindleConfig(clippings_path=Path("/no/such/file.txt"))
            ).fetch_documents()
        )
        assert docs == []

    def test_bookmark_without_text_skipped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_EMPTY_TEXT, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        # Empty text -> skipped
        assert docs == []

    def test_note_entry_parsed(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_WITH_NOTE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        assert len(docs) == 1
        assert "note" in docs[0].text.lower()

    def test_synthetic_path_contains_title(self, tmp_path: Path) -> None:
        from memorymesh.connectors.kindle_connector import (
            KindleConfig,
            KindleHighlightsConnector,
        )

        f = tmp_path / "My Clippings.txt"
        f.write_text(_CLIPPINGS_SAMPLE, encoding="utf-8")
        docs = list(KindleHighlightsConnector(KindleConfig(clippings_path=f)).fetch_documents())
        assert any("Pragmatic" in str(d.path) for d in docs)


_MOCK_ENTRY_RECENT = mock.MagicMock()
_MOCK_ENTRY_RECENT.title = "New Article"
_MOCK_ENTRY_RECENT.link = "https://example.com/new"
_MOCK_ENTRY_RECENT.id = "tag:example.com,2024:new"
_MOCK_ENTRY_RECENT.summary = "<p>Great article about Python.</p>"
_MOCK_ENTRY_RECENT.content = []
_MOCK_ENTRY_RECENT.published_parsed = time.gmtime(time.time() - 3600)  # 1 hour ago
_MOCK_ENTRY_RECENT.updated_parsed = None

_MOCK_ENTRY_OLD = mock.MagicMock()
_MOCK_ENTRY_OLD.title = "Old Article"
_MOCK_ENTRY_OLD.link = "https://example.com/old"
_MOCK_ENTRY_OLD.id = "tag:example.com,2020:old"
_MOCK_ENTRY_OLD.summary = "Old content"
_MOCK_ENTRY_OLD.content = []
_MOCK_ENTRY_OLD.published_parsed = time.gmtime(1_577_836_800)  # 2020-01-01
_MOCK_ENTRY_OLD.updated_parsed = None


class TestRSSConnector:
    def _make_parsed_feed(self, entries: list[object]) -> mock.MagicMock:
        feed_meta = mock.MagicMock()
        feed_meta.title = "Test Feed"
        parsed = mock.MagicMock()
        parsed.entries = entries
        parsed.feed = feed_meta
        return parsed

    def test_yields_document_per_entry(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        parsed = self._make_parsed_feed([_MOCK_ENTRY_RECENT])
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"], days_past=0)
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert len(docs) == 1
        assert docs[0].file_type == ".rss"

    def test_filters_old_entries(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        parsed = self._make_parsed_feed([_MOCK_ENTRY_RECENT, _MOCK_ENTRY_OLD])
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"], days_past=7)
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert len(docs) == 1
        assert docs[0].metadata["title"] == "New Article"

    def test_max_items_per_feed_respected(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        entries = [_MOCK_ENTRY_RECENT, _MOCK_ENTRY_RECENT]
        parsed = self._make_parsed_feed(entries)
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"], days_past=0, max_items_per_feed=1)
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert len(docs) == 1

    def test_feedparser_not_installed_yields_nothing(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"])
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": None}):
            docs = list(connector.fetch_documents())

        assert docs == []

    def test_html_stripped_from_summary(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        parsed = self._make_parsed_feed([_MOCK_ENTRY_RECENT])
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"], days_past=0)
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert "<p>" not in docs[0].text
        assert "Python" in docs[0].text

    def test_synthetic_path_contains_domain(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        parsed = self._make_parsed_feed([_MOCK_ENTRY_RECENT])
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(feeds=["https://example.com/feed.xml"], days_past=0)
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert "example.com" in str(docs[0].path)

    def test_multiple_feeds_aggregated(self) -> None:
        from memorymesh.connectors.rss_connector import RSSConfig, RSSConnector

        parsed = self._make_parsed_feed([_MOCK_ENTRY_RECENT])
        feedparser_mod = mock.MagicMock()
        feedparser_mod.parse.return_value = parsed

        cfg = RSSConfig(
            feeds=["https://a.com/feed", "https://b.com/feed"],
            days_past=0,
        )
        connector = RSSConnector(cfg)

        with mock.patch.dict("sys.modules", {"feedparser": feedparser_mod}):
            docs = list(connector.fetch_documents())

        assert len(docs) == 2

    def test_entry_hash_stable(self) -> None:
        from memorymesh.connectors.rss_connector import _entry_hash

        h1 = _entry_hash("https://example.com/article-123")
        h2 = _entry_hash("https://example.com/article-123")
        assert h1 == h2
        assert len(h1) == 16


def _make_zotero_db(path: Path) -> None:
    """Create a minimal Zotero-compatible SQLite database."""
    conn = sqlite3.connect(str(path))

    conn.executescript("""
        CREATE TABLE itemTypes (
            itemTypeID INTEGER PRIMARY KEY,
            typeName TEXT,
            templateItemTypeID INTEGER,
            display INTEGER
        );
        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            itemTypeID INTEGER,
            dateAdded TEXT,
            dateModified TEXT,
            clientDateModified TEXT,
            libraryID INTEGER,
            key TEXT
        );
        CREATE TABLE fields (
            fieldID INTEGER PRIMARY KEY,
            fieldName TEXT,
            fieldFormatID INTEGER
        );
        CREATE TABLE itemDataValues (
            valueID INTEGER PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE itemData (
            itemID INTEGER,
            fieldID INTEGER,
            valueID INTEGER
        );
        CREATE TABLE creatorTypes (
            creatorTypeID INTEGER PRIMARY KEY,
            creatorType TEXT
        );
        CREATE TABLE creators (
            creatorID INTEGER PRIMARY KEY,
            firstName TEXT,
            lastName TEXT,
            fieldMode INTEGER
        );
        CREATE TABLE itemCreators (
            itemID INTEGER,
            creatorID INTEGER,
            creatorTypeID INTEGER,
            orderIndex INTEGER
        );
        CREATE TABLE itemNotes (
            itemID INTEGER,
            sourceItemID INTEGER,
            note TEXT,
            title TEXT
        );
    """)

    # Item types
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle', NULL, 1)")
    conn.execute("INSERT INTO itemTypes VALUES (2, 'book', NULL, 1)")
    conn.execute("INSERT INTO itemTypes VALUES (3, 'note', NULL, 0)")

    # Items
    conn.execute(
        "INSERT INTO items VALUES (1, 1, '2024-01-01', '2024-01-01', '2024-01-01', 1, 'ABCD1234')"
    )
    conn.execute(
        "INSERT INTO items VALUES (2, 2, '2024-01-02', '2024-01-02', '2024-01-02', 1, 'EFGH5678')"
    )

    # Fields
    conn.execute("INSERT INTO fields VALUES (1, 'title', NULL)")
    conn.execute("INSERT INTO fields VALUES (2, 'abstractNote', NULL)")
    conn.execute("INSERT INTO fields VALUES (3, 'date', NULL)")
    conn.execute("INSERT INTO fields VALUES (4, 'DOI', NULL)")

    # Values
    conn.execute("INSERT INTO itemDataValues VALUES (1, 'Attention Is All You Need')")
    conn.execute("INSERT INTO itemDataValues VALUES (2, 'We propose a new architecture...')")
    conn.execute("INSERT INTO itemDataValues VALUES (3, '2017')")
    conn.execute("INSERT INTO itemDataValues VALUES (4, '10.48550/arXiv.1706.03762')")
    conn.execute("INSERT INTO itemDataValues VALUES (5, 'The Deep Learning Book')")
    conn.execute("INSERT INTO itemDataValues VALUES (6, 'An introduction to deep learning.')")
    conn.execute("INSERT INTO itemDataValues VALUES (7, '2016')")

    # Item data
    conn.execute("INSERT INTO itemData VALUES (1, 1, 1)")
    conn.execute("INSERT INTO itemData VALUES (1, 2, 2)")
    conn.execute("INSERT INTO itemData VALUES (1, 3, 3)")
    conn.execute("INSERT INTO itemData VALUES (1, 4, 4)")
    conn.execute("INSERT INTO itemData VALUES (2, 1, 5)")
    conn.execute("INSERT INTO itemData VALUES (2, 2, 6)")
    conn.execute("INSERT INTO itemData VALUES (2, 3, 7)")

    # Creator types
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")

    # Creators
    conn.execute("INSERT INTO creators VALUES (1, 'Ashish', 'Vaswani', 0)")
    conn.execute("INSERT INTO creators VALUES (2, 'Ian', 'Goodfellow', 0)")

    # Item creators
    conn.execute("INSERT INTO itemCreators VALUES (1, 1, 1, 0)")
    conn.execute("INSERT INTO itemCreators VALUES (2, 2, 1, 0)")

    # Notes
    conn.execute(
        "INSERT INTO itemNotes VALUES"
        " (10, 1, 'Key insight: multi-head attention.', 'Note on Attention')"
    )

    conn.commit()
    conn.close()


class TestZoteroConnector:
    def test_yields_one_doc_per_item(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        assert len(docs) == 2

    def test_file_type_is_zotero(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        assert all(d.file_type == ".zotero" for d in docs)

    def test_title_in_text(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        texts = [d.text for d in docs]
        assert any("Attention Is All You Need" in t for t in texts)

    def test_abstract_included(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(
            ZoteroConnector(ZoteroConfig(db_path=db, include_abstracts=True)).fetch_documents()
        )
        combined = " ".join(d.text for d in docs)
        assert "new architecture" in combined

    def test_abstract_excluded(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(
            ZoteroConnector(ZoteroConfig(db_path=db, include_abstracts=False)).fetch_documents()
        )
        combined = " ".join(d.text for d in docs)
        assert "new architecture" not in combined

    def test_notes_included(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db, include_notes=True)).fetch_documents())
        combined = " ".join(d.text for d in docs)
        assert "multi-head attention" in combined

    def test_notes_excluded(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(
            ZoteroConnector(ZoteroConfig(db_path=db, include_notes=False)).fetch_documents()
        )
        combined = " ".join(d.text for d in docs)
        assert "multi-head attention" not in combined

    def test_metadata_contains_authors(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        all_authors = [a for d in docs for a in d.metadata["authors"]]
        assert any("Vaswani" in a for a in all_authors)

    def test_missing_db_yields_nothing(self) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        docs = list(
            ZoteroConnector(ZoteroConfig(db_path=Path("/no/such/zotero.sqlite"))).fetch_documents()
        )
        assert docs == []

    def test_synthetic_path_contains_item_key(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        keys = {d.metadata["item_key"] for d in docs}
        assert "ABCD1234" in keys
        assert any("ABCD1234" in str(d.path) for d in docs)

    def test_doi_in_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.zotero_connector import ZoteroConfig, ZoteroConnector

        db = tmp_path / "zotero.sqlite"
        _make_zotero_db(db)

        docs = list(ZoteroConnector(ZoteroConfig(db_path=db)).fetch_documents())
        dois = {d.metadata["doi"] for d in docs}
        assert "10.48550/arXiv.1706.03762" in dois


class TestUrlHash:
    def test_stable_across_calls(self) -> None:
        from memorymesh.connectors.browser_history_connector import _url_hash

        assert _url_hash("https://example.com") == _url_hash("https://example.com")

    def test_different_urls_different_hashes(self) -> None:
        from memorymesh.connectors.browser_history_connector import _url_hash

        assert _url_hash("https://a.com") != _url_hash("https://b.com")

    def test_length_is_16(self) -> None:
        from memorymesh.connectors.browser_history_connector import _url_hash

        assert len(_url_hash("https://example.com/path?q=1")) == 16

    def test_matches_md5_prefix(self) -> None:
        from memorymesh.connectors.browser_history_connector import _url_hash

        url = "https://example.com"
        expected = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:16]
        assert _url_hash(url) == expected


class TestIMAPConnectorHTMLImport:
    def test_html_to_text_imported_from_shared_module(self) -> None:
        import memorymesh.connectors.imap_connector as imap_mod
        from memorymesh.connectors._html import html_to_text

        # The function used in _extract_body should be the shared one
        assert imap_mod.html_to_text is html_to_text


_WA_ANDROID = textwrap.dedent("""\
    12/31/2024, 14:30 - Alice: Hello there
    12/31/2024, 14:31 - Bob: Hey!
    12/31/2024, 14:32 - Alice: <Media omitted>
    12/31/2024, 14:33 - Bob: That's cool
""")

_WA_IOS = textwrap.dedent("""\
    [31/12/2024, 14:30:00] Alice: Hello there
    [31/12/2024, 14:31:00] Bob: Hey!
    [31/12/2024, 14:32:00] Alice: image omitted
    [31/12/2024, 14:33:00] Bob: Got it
""")

_WA_MULTILINE = textwrap.dedent("""\
    12/31/2024, 09:00 - Alice: First line
    continuation here
    12/31/2024, 09:01 - Bob: Response
""")


class TestWhatsAppConnector:
    def test_android_format_parses_messages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        f = tmp_path / "chat.txt"
        f.write_text(_WA_ANDROID, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=f)).fetch_documents())
        # 4 lines minus 1 media = 3 messages -> 1 chunk
        assert len(docs) == 1
        assert docs[0].metadata["message_count"] == 3

    def test_ios_format_parses_messages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        f = tmp_path / "chat.txt"
        f.write_text(_WA_IOS, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=f)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].metadata["message_count"] == 3

    def test_media_omitted_skipped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        f = tmp_path / "chat.txt"
        f.write_text(_WA_ANDROID, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=f)).fetch_documents())
        assert "<Media omitted>" not in docs[0].text
        assert "image omitted" not in docs[0].text

    def test_multiline_message_joined(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        f = tmp_path / "chat.txt"
        f.write_text(_WA_MULTILINE, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=f)).fetch_documents())
        assert docs[0].metadata["message_count"] == 2
        assert "continuation here" in docs[0].text

    def test_chunk_grouping(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        # 60 messages -> 2 chunks of 50/10 with chunk_size=50
        lines = [f"01/01/2024, 10:{i:02d} - Alice: msg {i}" for i in range(60)]
        f = tmp_path / "chat.txt"
        f.write_text("\n".join(lines), encoding="utf-8")
        docs = list(
            WhatsAppConnector(WhatsAppConfig(export_path=f, chunk_size=50)).fetch_documents()
        )
        assert len(docs) == 2
        assert docs[0].metadata["message_count"] == 50
        assert docs[1].metadata["message_count"] == 10

    def test_directory_of_files(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        (tmp_path / "chat1.txt").write_text(_WA_ANDROID, encoding="utf-8")
        (tmp_path / "chat2.txt").write_text(_WA_IOS, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=tmp_path)).fetch_documents())
        assert len(docs) == 2

    def test_missing_path_yields_nothing(self) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        docs = list(
            WhatsAppConnector(
                WhatsAppConfig(export_path=Path("/no/such/file.txt"))
            ).fetch_documents()
        )
        assert docs == []

    def test_metadata_and_path_format(self, tmp_path: Path) -> None:
        from memorymesh.connectors.whatsapp_connector import (
            WhatsAppConfig,
            WhatsAppConnector,
        )

        f = tmp_path / "Alice Chat.txt"
        f.write_text(_WA_ANDROID, encoding="utf-8")
        docs = list(WhatsAppConnector(WhatsAppConfig(export_path=f)).fetch_documents())
        doc = docs[0]
        assert doc.file_type == ".whatsapp"
        assert "chat_name" in doc.metadata
        assert "participants" in doc.metadata
        assert "start_date" in doc.metadata
        assert "end_date" in doc.metadata
        assert ".whatsapp" in str(doc.path)


_TG_EXPORT = {
    "chats": {
        "list": [
            {
                "name": "Alice",
                "type": "personal_chat",
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "date": "2024-01-01T10:00:00",
                        "from": "Me",
                        "text": "Hello",
                    },
                    {
                        "id": 2,
                        "type": "service",
                        "date": "2024-01-01T10:01:00",
                        "from": "Me",
                        "text": "",
                    },
                    {
                        "id": 3,
                        "type": "message",
                        "date": "2024-01-01T10:02:00",
                        "from": "Alice",
                        "text": [{"type": "plain", "text": "World"}],
                    },
                ],
            },
            {
                "name": "Work Group",
                "type": "private_supergroup",
                "messages": [
                    {
                        "id": 10,
                        "type": "message",
                        "date": "2024-01-02T09:00:00",
                        "from": "Bob",
                        "text": "Meeting at noon",
                    },
                ],
            },
        ]
    }
}


class TestTelegramConnector:
    def _write_export(self, tmp_path: Path) -> Path:
        import json

        f = tmp_path / "result.json"
        f.write_text(json.dumps(_TG_EXPORT), encoding="utf-8")
        return f

    def test_parses_messages_from_file(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        f = self._write_export(tmp_path)
        docs = list(TelegramConnector(TelegramConfig(export_path=f)).fetch_documents())
        # 2 messages in Alice chat, 1 in Work Group -> 2 chunks
        assert len(docs) == 2

    def test_skips_service_messages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        f = self._write_export(tmp_path)
        docs = list(
            TelegramConnector(
                TelegramConfig(export_path=f, chat_types=["personal_chat"])
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].metadata["message_count"] == 2

    def test_text_as_list_flattened(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        f = self._write_export(tmp_path)
        docs = list(
            TelegramConnector(
                TelegramConfig(export_path=f, chat_types=["personal_chat"])
            ).fetch_documents()
        )
        assert "World" in docs[0].text

    def test_filters_by_chat_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        f = self._write_export(tmp_path)
        # Only supergroups
        docs = list(
            TelegramConnector(
                TelegramConfig(export_path=f, chat_types=["private_supergroup"])
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].metadata["chat_name"] == "Work Group"

    def test_directory_finds_result_json(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        self._write_export(tmp_path)
        docs = list(TelegramConnector(TelegramConfig(export_path=tmp_path)).fetch_documents())
        assert len(docs) == 2

    def test_missing_file_yields_nothing(self) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        docs = list(
            TelegramConnector(
                TelegramConfig(export_path=Path("/no/such/result.json"))
            ).fetch_documents()
        )
        assert docs == []

    def test_metadata_keys(self, tmp_path: Path) -> None:
        from memorymesh.connectors.telegram_connector import (
            TelegramConfig,
            TelegramConnector,
        )

        f = self._write_export(tmp_path)
        docs = list(
            TelegramConnector(
                TelegramConfig(export_path=f, chat_types=["personal_chat"])
            ).fetch_documents()
        )
        doc = docs[0]
        assert doc.file_type == ".telegram"
        assert "chat_name" in doc.metadata
        assert "chat_type" in doc.metadata
        assert "participants" in doc.metadata
        assert ".telegram" in str(doc.path)


_TWEETS_JS_CONTENT = (
    "window.YTD.tweets.part0 = "
    '[{"tweet": {"id_str": "111", "full_text": "Hello world",'
    ' "created_at": "Mon Jan 01 10:00:00 +0000 2024",'
    ' "retweet_count": "5", "favorite_count": "10"}},'
    ' {"tweet": {"id_str": "222", "full_text": "RT @user: retweet",'
    ' "created_at": "Mon Jan 01 11:00:00 +0000 2024",'
    ' "retweet_count": "0", "favorite_count": "0"}},'
    ' {"tweet": {"id_str": "333", "full_text": "Another tweet",'
    ' "created_at": "Mon Jan 02 10:00:00 +0000 2024",'
    ' "retweet_count": "2", "favorite_count": "7"}}]'
)

_LIKES_JS_CONTENT = (
    "window.YTD.like.part0 = "
    '[{"like": {"tweetId": "999", "fullText": "Liked tweet text",'
    ' "expandedUrl": "https://twitter.com/user/status/999"}}]'
)


def _make_twitter_archive(base: Path) -> None:
    """Create a minimal Twitter archive directory structure."""
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "tweets.js").write_text(_TWEETS_JS_CONTENT, encoding="utf-8")
    (data_dir / "likes.js").write_text(_LIKES_JS_CONTENT, encoding="utf-8")


class TestTwitterArchiveConnector:
    def test_parses_tweets_js(self, tmp_path: Path) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        _make_twitter_archive(tmp_path)
        docs = list(
            TwitterArchiveConnector(TwitterArchiveConfig(archive_path=tmp_path)).fetch_documents()
        )
        # 3 items, 1 retweet skipped -> 2 docs
        assert len(docs) == 2

    def test_skips_retweets(self, tmp_path: Path) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        _make_twitter_archive(tmp_path)
        docs = list(
            TwitterArchiveConnector(TwitterArchiveConfig(archive_path=tmp_path)).fetch_documents()
        )
        assert all("RT @" not in d.text for d in docs)

    def test_include_likes(self, tmp_path: Path) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        _make_twitter_archive(tmp_path)
        docs = list(
            TwitterArchiveConnector(
                TwitterArchiveConfig(archive_path=tmp_path, include_likes=True)
            ).fetch_documents()
        )
        # 2 original tweets + 1 liked tweet
        assert len(docs) == 3
        assert any("Liked tweet text" in d.text for d in docs)

    def test_synthetic_path_contains_tweet_id(self, tmp_path: Path) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        _make_twitter_archive(tmp_path)
        docs = list(
            TwitterArchiveConnector(TwitterArchiveConfig(archive_path=tmp_path)).fetch_documents()
        )
        paths = {str(d.path) for d in docs}
        assert any("111" in p for p in paths)
        assert all(p.endswith(".tweet") for p in paths)

    def test_zip_archive(self, tmp_path: Path) -> None:
        import zipfile

        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        # Build archive dir then zip it
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        _make_twitter_archive(archive_dir)
        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in archive_dir.rglob("*"):
                zf.write(f, f.relative_to(archive_dir))
        docs = list(
            TwitterArchiveConnector(TwitterArchiveConfig(archive_path=zip_path)).fetch_documents()
        )
        assert len(docs) == 2

    def test_missing_path_yields_nothing(self) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        docs = list(
            TwitterArchiveConnector(
                TwitterArchiveConfig(archive_path=Path("/no/such/dir"))
            ).fetch_documents()
        )
        assert docs == []

    def test_metadata_fields(self, tmp_path: Path) -> None:
        from memorymesh.connectors.twitter_connector import (
            TwitterArchiveConfig,
            TwitterArchiveConnector,
        )

        _make_twitter_archive(tmp_path)
        docs = list(
            TwitterArchiveConnector(TwitterArchiveConfig(archive_path=tmp_path)).fetch_documents()
        )
        doc = docs[0]
        assert doc.file_type == ".tweet"
        assert "tweet_id" in doc.metadata
        assert "created_at" in doc.metadata
        assert "retweets" in doc.metadata
        assert "likes" in doc.metadata


_SPOTIFY_RECORDS_JAN = [
    {
        "ts": "2024-01-15T10:00:00Z",
        "master_metadata_track_name": "Song A",
        "master_metadata_album_artist_name": "Artist A",
        "master_metadata_album_album_name": "Album A",
        "ms_played": 240_000,
        "reason_end": "trackdone",
    },
    {
        "ts": "2024-01-16T11:00:00Z",
        "master_metadata_track_name": "Short clip",
        "master_metadata_album_artist_name": "Artist B",
        "master_metadata_album_album_name": "Album B",
        "ms_played": 5_000,  # below 30 s threshold
        "reason_end": "fwdbtn",
    },
]

_SPOTIFY_RECORDS_FEB = [
    {
        "ts": "2024-02-10T08:00:00Z",
        "master_metadata_track_name": "Song B",
        "master_metadata_album_artist_name": "Artist C",
        "master_metadata_album_album_name": "Album C",
        "ms_played": 180_000,
        "reason_end": "trackdone",
    },
]


class TestSpotifyHistoryConnector:
    def _write_files(self, tmp_path: Path) -> None:
        import json

        (tmp_path / "Streaming_History_Audio_0.json").write_text(
            json.dumps(_SPOTIFY_RECORDS_JAN), encoding="utf-8"
        )
        (tmp_path / "Streaming_History_Audio_1.json").write_text(
            json.dumps(_SPOTIFY_RECORDS_FEB), encoding="utf-8"
        )

    def test_parses_history(self, tmp_path: Path) -> None:
        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        self._write_files(tmp_path)
        docs = list(SpotifyHistoryConnector(SpotifyConfig(history_path=tmp_path)).fetch_documents())
        # Jan has 1 qualifying play, Feb has 1 -> 2 monthly docs
        assert len(docs) == 2

    def test_filters_by_min_ms_played(self, tmp_path: Path) -> None:
        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        self._write_files(tmp_path)
        docs = list(SpotifyHistoryConnector(SpotifyConfig(history_path=tmp_path)).fetch_documents())
        jan_doc = next(d for d in docs if d.metadata["year_month"] == "2024-01")
        assert "Short clip" not in jan_doc.text

    def test_groups_by_month(self, tmp_path: Path) -> None:
        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        self._write_files(tmp_path)
        docs = list(SpotifyHistoryConnector(SpotifyConfig(history_path=tmp_path)).fetch_documents())
        months = {d.metadata["year_month"] for d in docs}
        assert months == {"2024-01", "2024-02"}

    def test_multiple_json_files_merged(self, tmp_path: Path) -> None:
        import json

        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        # Both files cover January -> merged into one doc
        (tmp_path / "Streaming_History_Audio_0.json").write_text(
            json.dumps([_SPOTIFY_RECORDS_JAN[0]]), encoding="utf-8"
        )
        (tmp_path / "Streaming_History_Audio_1.json").write_text(
            json.dumps([_SPOTIFY_RECORDS_JAN[0]]), encoding="utf-8"
        )
        docs = list(SpotifyHistoryConnector(SpotifyConfig(history_path=tmp_path)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].metadata["track_count"] == 2

    def test_missing_directory_yields_nothing(self) -> None:
        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        docs = list(
            SpotifyHistoryConnector(
                SpotifyConfig(history_path=Path("/no/such/dir"))
            ).fetch_documents()
        )
        assert docs == []

    def test_metadata_fields(self, tmp_path: Path) -> None:
        from memorymesh.connectors.spotify_connector import (
            SpotifyConfig,
            SpotifyHistoryConnector,
        )

        self._write_files(tmp_path)
        docs = list(SpotifyHistoryConnector(SpotifyConfig(history_path=tmp_path)).fetch_documents())
        doc = docs[0]
        assert doc.file_type == ".spotify"
        assert "year_month" in doc.metadata
        assert "track_count" in doc.metadata
        assert "total_hours" in doc.metadata
        assert doc.metadata["total_hours"] > 0
        assert ".spotify" in str(doc.path)


class _MockHeaders:
    """Minimal mock of an http.client.HTTPMessage for response header access."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._h = headers

    def items(self) -> list[tuple[str, str]]:
        """Return headers as (name, value) pairs."""
        return list(self._h.items())


class _MockResp:
    """Minimal mock of a urllib HTTP response (context manager)."""

    def __init__(
        self,
        data: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = json.dumps(data).encode()
        self.headers = _MockHeaders(headers or {})

    def read(self) -> bytes:
        """Return serialised response body."""
        return self._body

    def __enter__(self) -> _MockResp:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


_NOTION_SEARCH_RESP = {
    "results": [
        {
            "id": "page-uuid-1",
            "created_time": "2024-01-01T10:00:00.000Z",
            "last_edited_time": "2024-01-02T10:00:00.000Z",
            "url": "https://notion.so/page-uuid-1",
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "My Page"}],
                }
            },
        }
    ],
    "has_more": False,
}

_NOTION_BLOCKS_RESP = {
    "results": [
        {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": "Hello world"}]},
        },
        {
            "type": "heading_1",
            "heading_1": {"rich_text": [{"plain_text": "Title"}]},
        },
    ],
    "has_more": False,
}


class TestNotionConnector:
    def test_search_yields_page(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.notion_connector import (
            NotionConfig,
            NotionConnector,
        )

        responses = [_MockResp(_NOTION_SEARCH_RESP), _MockResp(_NOTION_BLOCKS_RESP)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                NotionConnector(NotionConfig(api_key=SecretStr("secret_test"))).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "My Page"
        assert docs[0].metadata["page_id"] == "page-uuid-1"

    def test_database_query_yields_page(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.notion_connector import (
            NotionConfig,
            NotionConnector,
        )

        db_resp = {
            "results": [
                {
                    "id": "db-page-1",
                    "created_time": "2024-01-01T00:00:00.000Z",
                    "last_edited_time": "2024-01-01T00:00:00.000Z",
                    "url": "",
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "DB Entry"}],
                        }
                    },
                }
            ],
            "has_more": False,
        }
        responses = [_MockResp(db_resp), _MockResp(_NOTION_BLOCKS_RESP)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                NotionConnector(
                    NotionConfig(
                        api_key=SecretStr("secret_test"),
                        database_ids=["db-uuid-1"],
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "DB Entry"

    def test_block_content_in_text(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.notion_connector import (
            NotionConfig,
            NotionConnector,
        )

        responses = [_MockResp(_NOTION_SEARCH_RESP), _MockResp(_NOTION_BLOCKS_RESP)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                NotionConnector(NotionConfig(api_key=SecretStr("secret_test"))).fetch_documents()
            )
        assert "Hello world" in docs[0].text
        assert "# Title" in docs[0].text

    def test_pagination_fetches_all_pages(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.notion_connector import (
            NotionConfig,
            NotionConnector,
        )

        page_factory = lambda pid: {  # noqa: E731
            "id": pid,
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-01T00:00:00.000Z",
            "url": "",
            "properties": {"title": {"type": "title", "title": [{"plain_text": pid}]}},
        }
        first_page = {
            "results": [page_factory("p1")],
            "has_more": True,
            "next_cursor": "cursor-2",
        }
        second_page = {
            "results": [page_factory("p2")],
            "has_more": False,
        }
        empty_blocks = {"results": [], "has_more": False}
        responses = [
            _MockResp(first_page),
            _MockResp(empty_blocks),
            _MockResp(second_page),
            _MockResp(empty_blocks),
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                NotionConnector(NotionConfig(api_key=SecretStr("secret_test"))).fetch_documents()
            )
        assert len(docs) == 2

    def test_block_types_converted(self) -> None:
        from memorymesh.connectors.notion_connector import _block_to_text

        cases = [
            ({"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "H"}]}}, "# H"),
            ({"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "H"}]}}, "## H"),
            (
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"plain_text": "x"}]},
                },
                "- x",
            ),
            ({"type": "quote", "quote": {"rich_text": [{"plain_text": "q"}]}}, "> q"),
            (
                {
                    "type": "code",
                    "code": {"language": "python", "rich_text": [{"plain_text": "pass"}]},
                },
                "```python\npass\n```",
            ),
        ]
        for block, expected in cases:
            assert _block_to_text(block) == expected

    def test_file_type_and_path(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.notion_connector import (
            NotionConfig,
            NotionConnector,
        )

        responses = [_MockResp(_NOTION_SEARCH_RESP), _MockResp(_NOTION_BLOCKS_RESP)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                NotionConnector(NotionConfig(api_key=SecretStr("secret_test"))).fetch_documents()
            )
        doc = docs[0]
        assert doc.file_type == ".notion"
        assert "page-uuid-1" in str(doc.path)
        assert str(doc.path).endswith(".notion")


_GH_ISSUE = {
    "number": 42,
    "title": "Fix the bug",
    "body": "This is broken.",
    "state": "open",
    "user": {"login": "alice"},
    "created_at": "2099-01-01T00:00:00Z",  # far future -> always in days_past
    "html_url": "https://github.com/owner/repo/issues/42",
    "comments": 0,
}

_GH_PR = {
    "number": 10,
    "title": "Add feature",
    "body": "New feature.",
    "state": "open",
    "user": {"login": "bob"},
    "created_at": "2099-01-01T00:00:00Z",
    "html_url": "https://github.com/owner/repo/pull/10",
    "comments": 0,
    "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/10"},
}

_GH_HEADERS = {"x-ratelimit-remaining": "60", "x-ratelimit-reset": "0"}


def _gh_resp(items: list, headers: dict | None = None):
    return _MockResp(items, headers or _GH_HEADERS)


class TestGitHubConnector:
    def test_fetches_issues(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        responses = [_gh_resp([_GH_ISSUE]), _gh_resp([])]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_prs=False,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "issue"
        assert docs[0].metadata["number"] == 42

    def test_fetches_prs(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        responses = [_gh_resp([_GH_PR]), _gh_resp([])]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_issues=False,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "pr"

    def test_skips_prs_when_disabled(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        responses = [_gh_resp([_GH_ISSUE, _GH_PR]), _gh_resp([])]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_prs=False,
                    )
                ).fetch_documents()
            )
        assert all(d.metadata["type"] == "issue" for d in docs)

    def test_days_past_filter(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        old_issue = dict(_GH_ISSUE, created_at="2000-01-01T00:00:00Z", number=99)
        responses = [_gh_resp([old_issue]), _gh_resp([])]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        days_past=30,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_include_comments(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        issue_with_comments = dict(_GH_ISSUE, comments=1)
        comment = {
            "body": "Great issue!",
            "user": {"login": "carol"},
        }
        responses = [
            _gh_resp([issue_with_comments]),
            _MockResp([comment]),  # comments endpoint (page 1) - fetched inline
            _MockResp([]),  # comments pagination end
            _gh_resp([]),  # pagination terminator for next issues page
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_comments=True,
                        include_prs=False,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert "Great issue!" in docs[0].text

    def test_rate_limit_triggers_sleep(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        low_limit_headers = {
            "x-ratelimit-remaining": "5",
            "x-ratelimit-reset": "0",
        }
        responses = [_gh_resp([_GH_ISSUE], low_limit_headers), _gh_resp([])]
        sleep_calls = []
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch(
                "memorymesh.connectors.github_connector.time.sleep",
                side_effect=lambda s: sleep_calls.append(s),
            ),
        ):
            list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_prs=False,
                    )
                ).fetch_documents()
            )
        assert any(s > 0 for s in sleep_calls)

    def test_metadata_and_path(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.github_connector import (
            GitHubConfig,
            GitHubConnector,
        )

        responses = [_gh_resp([_GH_ISSUE]), _gh_resp([])]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                GitHubConnector(
                    GitHubConfig(
                        token=SecretStr("ghp_test"),
                        repos=["owner/repo"],
                        include_prs=False,
                    )
                ).fetch_documents()
            )
        doc = docs[0]
        assert doc.file_type == ".github"
        assert "owner/repo" in doc.metadata["repo"]
        assert "42" in str(doc.path)


_SAVE_CP = "memorymesh.connectors.readwise_connector.ReadwiseConnector._save_checkpoint"

_RW_HIGHLIGHT_1 = {
    "id": 1,
    "text": "First highlight",
    "book_id": 100,
    "updated": "2024-01-15T10:00:00.000000Z",
}
_RW_HIGHLIGHT_2 = {
    "id": 2,
    "text": "Second highlight",
    "book_id": 100,
    "updated": "2024-01-16T10:00:00.000000Z",
}
_RW_BOOK = {
    "id": 100,
    "title": "Deep Work",
    "author": "Cal Newport",
    "category": "books",
    "source_url": "https://example.com/deep-work",
}


class TestReadwiseConnector:
    def test_fetches_highlights_and_yields_doc(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        hl_resp = {"count": 1, "next": None, "results": [_RW_HIGHLIGHT_1]}
        responses = [_MockResp(hl_resp), _MockResp(_RW_BOOK)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch(_SAVE_CP),
        ):
            docs = list(
                ReadwiseConnector(ReadwiseConfig(api_key=SecretStr("rw_test"))).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["title"] == "Deep Work"
        assert "First highlight" in docs[0].text

    def test_groups_highlights_by_book(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        hl_resp = {
            "count": 2,
            "next": None,
            "results": [_RW_HIGHLIGHT_1, _RW_HIGHLIGHT_2],
        }
        responses = [_MockResp(hl_resp), _MockResp(_RW_BOOK)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch(_SAVE_CP),
        ):
            docs = list(
                ReadwiseConnector(ReadwiseConfig(api_key=SecretStr("rw_test"))).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["highlight_count"] == 2
        assert "First highlight" in docs[0].text
        assert "Second highlight" in docs[0].text

    def test_source_type_filter(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        article_book = dict(_RW_BOOK, category="articles", id=200)
        hl = dict(_RW_HIGHLIGHT_1, book_id=200)
        hl_resp = {"count": 1, "next": None, "results": [hl]}
        responses = [_MockResp(hl_resp), _MockResp(article_book)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch(_SAVE_CP),
        ):
            docs = list(
                ReadwiseConnector(
                    ReadwiseConfig(
                        api_key=SecretStr("rw_test"),
                        source_types=["books"],  # filter out articles
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_checkpoint_saved(self, tmp_path: Path) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        checkpoint = tmp_path / "checkpoint.json"
        hl_resp = {"count": 1, "next": None, "results": [_RW_HIGHLIGHT_1]}
        responses = [_MockResp(hl_resp), _MockResp(_RW_BOOK)]
        with mock.patch(
            "memorymesh.connectors._http.urllib.request.urlopen",
            side_effect=iter(responses),
        ):
            list(
                ReadwiseConnector(
                    ReadwiseConfig(
                        api_key=SecretStr("rw_test"),
                        checkpoint_path=checkpoint,
                    )
                ).fetch_documents()
            )
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text())
        assert "updated_after" in data

    def test_checkpoint_loaded_on_next_run(self, tmp_path: Path) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        checkpoint = tmp_path / "checkpoint.json"
        checkpoint.write_text(json.dumps({"updated_after": "2024-06-01T00:00:00Z"}))
        # Connector should pass updated__gt in the URL
        captured_urls: list[str] = []

        def mock_urlopen(req, *args, **kwargs):
            captured_urls.append(req.full_url)
            empty = {"count": 0, "next": None, "results": []}
            return _MockResp(empty)

        with mock.patch(
            "memorymesh.connectors._http.urllib.request.urlopen",
            side_effect=mock_urlopen,
        ):
            list(
                ReadwiseConnector(
                    ReadwiseConfig(
                        api_key=SecretStr("rw_test"),
                        checkpoint_path=checkpoint,
                    )
                ).fetch_documents()
            )
        assert any("updated__gt" in url for url in captured_urls)

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.readwise_connector import (
            ReadwiseConfig,
            ReadwiseConnector,
        )

        hl_resp = {"count": 1, "next": None, "results": [_RW_HIGHLIGHT_1]}
        responses = [_MockResp(hl_resp), _MockResp(_RW_BOOK)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch(_SAVE_CP),
        ):
            docs = list(
                ReadwiseConnector(ReadwiseConfig(api_key=SecretStr("rw_test"))).fetch_documents()
            )
        doc = docs[0]
        assert doc.file_type == ".readwise"
        assert "book_id" in doc.metadata
        assert "author" in doc.metadata
        assert "category" in doc.metadata
        assert "100" in str(doc.path)


_SLACK_CHANNELS_RESP = {
    "ok": True,
    "channels": [{"id": "C001", "name": "general"}],
    "response_metadata": {"next_cursor": ""},
}

_SLACK_HISTORY_RESP = {
    "ok": True,
    "messages": [
        {"type": "message", "user": "U001", "text": "Hello everyone", "ts": "1704067200.000001"},
        {"type": "message", "user": "U002", "text": "Hey there", "ts": "1704067260.000001"},
        {"type": "message", "user": "U001", "text": "Good morning", "ts": "1704153600.000001"},
    ],
    "has_more": False,
    "response_metadata": {"next_cursor": ""},
}

_SLACK_USER_U001 = {
    "ok": True,
    "user": {
        "id": "U001",
        "real_name": "Alice Smith",
        "profile": {"display_name": "alice"},
    },
}

_SLACK_USER_U002 = {
    "ok": True,
    "user": {
        "id": "U002",
        "real_name": "Bob Jones",
        "profile": {"display_name": "bob"},
    },
}


class TestSlackConnector:
    def test_fetches_channels_and_history(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        responses = [
            _MockResp(_SLACK_CHANNELS_RESP),
            _MockResp(_SLACK_HISTORY_RESP),
            _MockResp(_SLACK_USER_U001),
            _MockResp(_SLACK_USER_U002),
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                SlackConnector(
                    SlackConfig(token=SecretStr("xoxb-test"), days_past=0)
                ).fetch_documents()
            )
        assert len(docs) >= 1
        all_text = " ".join(d.text for d in docs)
        assert "Hello everyone" in all_text

    def test_groups_messages_by_day(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        # ts 1704067200 = 2024-01-01, ts 1704153600 = 2024-01-02
        responses = [
            _MockResp(_SLACK_CHANNELS_RESP),
            _MockResp(_SLACK_HISTORY_RESP),
            _MockResp(_SLACK_USER_U001),
            _MockResp(_SLACK_USER_U002),
            _MockResp(_SLACK_USER_U001),
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                SlackConnector(
                    SlackConfig(token=SecretStr("xoxb-test"), days_past=0)
                ).fetch_documents()
            )
        dates = {d.metadata["date"] for d in docs}
        # Ts 1704067200 = 2024-01-01, 1704153600 = 2024-01-02
        assert len(dates) == 2

    def test_explicit_channel_ids_skip_listing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        responses = [
            _MockResp(_SLACK_HISTORY_RESP),
            _MockResp(_SLACK_USER_U001),
            _MockResp(_SLACK_USER_U002),
            _MockResp(_SLACK_USER_U001),
        ]
        captured_urls: list[str] = []

        def mock_urlopen(req, *args, **kwargs):
            captured_urls.append(req.full_url)
            return responses.pop(0)

        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=mock_urlopen,
            ),
            mock.patch("time.sleep"),
        ):
            list(
                SlackConnector(
                    SlackConfig(
                        token=SecretStr("xoxb-test"),
                        channel_ids=["C001"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert not any("conversations.list" in url for url in captured_urls)

    def test_user_name_resolved(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        responses = [
            _MockResp(_SLACK_CHANNELS_RESP),
            _MockResp(_SLACK_HISTORY_RESP),
            _MockResp(_SLACK_USER_U001),
            _MockResp(_SLACK_USER_U002),
            _MockResp(_SLACK_USER_U001),
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                SlackConnector(
                    SlackConfig(token=SecretStr("xoxb-test"), days_past=0)
                ).fetch_documents()
            )
        all_text = " ".join(d.text for d in docs)
        # Display names should appear in output, not raw user IDs
        assert "alice" in all_text or "Alice" in all_text

    def test_empty_channel_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        empty_history = {
            "ok": True,
            "messages": [],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        responses = [_MockResp(_SLACK_CHANNELS_RESP), _MockResp(empty_history)]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                SlackConnector(
                    SlackConfig(token=SecretStr("xoxb-test"), days_past=0)
                ).fetch_documents()
            )
        assert docs == []

    def test_metadata_and_path_format(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.slack_connector import (
            SlackConfig,
            SlackConnector,
        )

        responses = [
            _MockResp(_SLACK_CHANNELS_RESP),
            _MockResp(_SLACK_HISTORY_RESP),
            _MockResp(_SLACK_USER_U001),
            _MockResp(_SLACK_USER_U002),
            _MockResp(_SLACK_USER_U001),
        ]
        with (
            mock.patch(
                "memorymesh.connectors._http.urllib.request.urlopen",
                side_effect=iter(responses),
            ),
            mock.patch("time.sleep"),
        ):
            docs = list(
                SlackConnector(
                    SlackConfig(token=SecretStr("xoxb-test"), days_past=0)
                ).fetch_documents()
            )
        doc = docs[0]
        assert doc.file_type == ".slack"
        assert "channel_id" in doc.metadata
        assert "date" in doc.metadata
        assert "message_count" in doc.metadata
        assert str(doc.path).endswith(".slack")


_AH_XML_TMPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
 <ExportDate value="2024-01-16 10:00:00 +0000"/>
 {body}
</HealthData>
"""


def _ah_record(rec_type: str, value: str, unit: str, start: str) -> str:
    return (
        f'<Record type="{rec_type}" unit="{unit}" value="{value}"'
        f' startDate="{start}" endDate="{start}"/>'
    )


def _ah_workout(act_type: str, duration: str, energy: str, start: str) -> str:
    return (
        f'<Workout workoutActivityType="{act_type}" duration="{duration}"'
        f' durationUnit="min" totalEnergyBurned="{energy}"'
        f' totalEnergyBurnedUnit="kcal"'
        f' startDate="{start}" endDate="{start}"/>'
    )


def _ah_xml(*parts: str) -> str:
    return _AH_XML_TMPL.format(body="\n ".join(parts))


class TestAppleHealthConnector:
    def test_parses_records_from_xml(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "5000",
                "count",
                "2024-01-10 09:00:00 +0000",
            )
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=xml_file, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].file_type == ".health"
        assert "StepCount" in docs[0].metadata["record_type"]
        assert docs[0].metadata["count"] == 1

    def test_filters_by_record_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "5000",
                "count",
                "2024-01-10 09:00:00 +0000",
            ),
            _ah_record(
                "HKQuantityTypeIdentifierHeartRate",
                "70",
                "count/min",
                "2024-01-10 09:00:00 +0000",
            ),
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(
                    export_path=xml_file,
                    record_types=["HKQuantityTypeIdentifierStepCount"],
                    days_past=0,
                )
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert "StepCount" in docs[0].metadata["record_type"]

    def test_filters_by_days_past(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "1000",
                "count",
                "2020-01-01 09:00:00 +0000",
            )
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=xml_file, days_past=30)
            ).fetch_documents()
        )
        assert docs == []

    def test_groups_by_month(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "4000",
                "count",
                "2024-01-10 09:00:00 +0000",
            ),
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "6000",
                "count",
                "2024-02-10 09:00:00 +0000",
            ),
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=xml_file, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 2
        months = {d.metadata["year_month"] for d in docs}
        assert "2024-01" in months
        assert "2024-02" in months

    def test_parses_workout(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_workout(
                "HKWorkoutActivityTypeRunning",
                "30",
                "350",
                "2024-01-10 09:00:00 +0000",
            )
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=xml_file, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].file_type == ".workout"
        assert "Running" in docs[0].text

    def test_zip_extraction(self, tmp_path: Path) -> None:
        import zipfile as _zf

        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "8000",
                "count",
                "2024-01-10 09:00:00 +0000",
            )
        )
        zip_path = tmp_path / "export.zip"
        with _zf.ZipFile(zip_path, "w") as zf:
            zf.writestr("apple_health_export/export.xml", xml)
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=zip_path, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].file_type == ".health"

    def test_stats_in_text_and_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.apple_health_connector import (
            AppleHealthConfig,
            AppleHealthConnector,
        )

        xml = _ah_xml(
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "4000",
                "count",
                "2024-01-10 09:00:00 +0000",
            ),
            _ah_record(
                "HKQuantityTypeIdentifierStepCount",
                "8000",
                "count",
                "2024-01-11 09:00:00 +0000",
            ),
        )
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(xml, encoding="utf-8")
        docs = list(
            AppleHealthConnector(
                AppleHealthConfig(export_path=xml_file, days_past=0)
            ).fetch_documents()
        )
        text = docs[0].text
        assert "Min:" in text
        assert "Max:" in text
        assert "Avg:" in text
        assert docs[0].metadata["count"] == 2


_GL_RECORDS = {
    "locations": [
        {
            "timestampMs": "1704873600000",
            "latitudeE7": 485234000,
            "longitudeE7": 23456000,
            "accuracy": 20,
        },
        {
            "timestampMs": "1704960000000",
            "latitudeE7": 485235000,
            "longitudeE7": 23457000,
            "accuracy": 15,
        },
    ]
}

_GL_SEMANTIC = {
    "timelineObjects": [
        {
            "placeVisit": {
                "location": {
                    "name": "Anthropic HQ",
                    "address": "548 Market St, San Francisco",
                },
                "duration": {
                    "startTimestampMs": "1704873600000",
                    "endTimestampMs": "1704877200000",
                },
            }
        }
    ]
}


class TestGoogleLocationConnector:
    def test_parses_records_json(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        records_file = tmp_path / "Records.json"
        records_file.write_text(json.dumps(_GL_RECORDS), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=records_file, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 2
        assert docs[0].file_type == ".location"
        assert docs[0].metadata["point_count"] == 1

    def test_parses_semantic_json(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        sem_dir = tmp_path / "Semantic Location History"
        sem_dir.mkdir()
        (sem_dir / "2024_JANUARY.json").write_text(json.dumps(_GL_SEMANTIC), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=sem_dir, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 1
        assert "Anthropic HQ" in docs[0].text
        assert docs[0].metadata["place_name"] == "Anthropic HQ"

    def test_no_raw_coordinates_in_text(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        records_file = tmp_path / "Records.json"
        records_file.write_text(json.dumps(_GL_RECORDS), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=records_file, days_past=0)
            ).fetch_documents()
        )
        for doc in docs:
            assert "4852" not in doc.text
            assert "latitudeE7" not in doc.text

    def test_filters_by_days_past(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        old_data = {
            "locations": [
                {
                    "timestampMs": "1000000000000",
                    "latitudeE7": 100,
                    "longitudeE7": 100,
                    "accuracy": 10,
                }
            ]
        }
        records_file = tmp_path / "Records.json"
        records_file.write_text(json.dumps(old_data), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=records_file, days_past=30)
            ).fetch_documents()
        )
        assert docs == []

    def test_auto_detects_records_in_dir(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        (tmp_path / "Records.json").write_text(json.dumps(_GL_RECORDS), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=tmp_path, days_past=0)
            ).fetch_documents()
        )
        assert len(docs) == 2

    def test_semantic_duration_calculated(self, tmp_path: Path) -> None:
        from memorymesh.connectors.google_location_connector import (
            GoogleLocationConfig,
            GoogleLocationConnector,
        )

        sem_dir = tmp_path / "Semantic Location History"
        sem_dir.mkdir()
        (sem_dir / "2024_JANUARY.json").write_text(json.dumps(_GL_SEMANTIC), encoding="utf-8")
        docs = list(
            GoogleLocationConnector(
                GoogleLocationConfig(export_path=sem_dir, days_past=0)
            ).fetch_documents()
        )
        # 1704877200000 - 1704873600000 = 3600000 ms = 60 min
        assert docs[0].metadata["duration_min"] == 60


_STRAVA_TOKEN = {"access_token": "test_access_token", "token_type": "Bearer"}
_STRAVA_ACT = {
    "id": 12345,
    "name": "Morning Run",
    "sport_type": "Run",
    "start_date_local": "2024-01-10T09:00:00Z",
    "distance": 5000.0,
    "moving_time": 1800,
    "total_elevation_gain": 50.0,
    "average_heartrate": 145.0,
    "max_heartrate": 170.0,
}

_STRAVA_URL = "memorymesh.connectors.strava_connector.urllib.request.urlopen"


class TestStravaConnector:
    def test_fetches_activities(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.strava_connector import (
            StravaConfig,
            StravaConnector,
        )

        responses = [
            _MockResp(_STRAVA_TOKEN),
            _MockResp([_STRAVA_ACT]),
            _MockResp([]),
        ]
        with mock.patch(_STRAVA_URL, side_effect=iter(responses)):
            docs = list(
                StravaConnector(
                    StravaConfig(
                        client_id="123",
                        client_secret=SecretStr("secret"),
                        refresh_token=SecretStr("rtoken"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert "Morning Run" in docs[0].text
        assert docs[0].file_type == ".strava"

    def test_token_refresh_failure_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.strava_connector import (
            StravaConfig,
            StravaConnector,
        )

        with mock.patch(
            _STRAVA_URL,
            side_effect=Exception("network error"),
        ):
            docs = list(
                StravaConnector(
                    StravaConfig(
                        client_id="123",
                        client_secret=SecretStr("secret"),
                        refresh_token=SecretStr("rtoken"),
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_pagination(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.strava_connector import (
            StravaConfig,
            StravaConnector,
        )

        act2 = dict(_STRAVA_ACT, id=99999, name="Evening Ride")
        responses = [
            _MockResp(_STRAVA_TOKEN),
            _MockResp([_STRAVA_ACT]),
            _MockResp([act2]),
            _MockResp([]),
        ]
        with mock.patch(_STRAVA_URL, side_effect=iter(responses)):
            docs = list(
                StravaConnector(
                    StravaConfig(
                        client_id="123",
                        client_secret=SecretStr("secret"),
                        refresh_token=SecretStr("rtoken"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 2

    def test_max_activities_limit(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.strava_connector import (
            StravaConfig,
            StravaConnector,
        )

        act2 = dict(_STRAVA_ACT, id=99999)
        responses = [
            _MockResp(_STRAVA_TOKEN),
            _MockResp([_STRAVA_ACT, act2]),
            _MockResp([]),
        ]
        with mock.patch(_STRAVA_URL, side_effect=iter(responses)):
            docs = list(
                StravaConnector(
                    StravaConfig(
                        client_id="123",
                        client_secret=SecretStr("secret"),
                        refresh_token=SecretStr("rtoken"),
                        days_past=0,
                        max_activities=1,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1

    def test_activity_text_and_metadata(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.strava_connector import (
            StravaConfig,
            StravaConnector,
        )

        responses = [
            _MockResp(_STRAVA_TOKEN),
            _MockResp([_STRAVA_ACT]),
            _MockResp([]),
        ]
        with mock.patch(_STRAVA_URL, side_effect=iter(responses)):
            docs = list(
                StravaConnector(
                    StravaConfig(
                        client_id="123",
                        client_secret=SecretStr("secret"),
                        refresh_token=SecretStr("rtoken"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        doc = docs[0]
        assert "5.0 km" in doc.text
        assert "30 min" in doc.text
        assert "145 bpm" in doc.text
        assert doc.metadata["activity_id"] == 12345
        assert doc.metadata["distance_km"] == 5.0
        assert doc.metadata["duration_min"] == 30.0


class TestBankCSVConnector:
    def test_parses_csv_basic(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "Date,Description,Amount\n2024-01-10,Coffee,-5.50\n2024-01-15,Salary,3000.00\n",
            encoding="utf-8",
        )
        docs = list(BankCSVConnector(BankCSVConfig(csv_path=csv_file)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].metadata["year_month"] == "2024-01"
        assert docs[0].metadata["transaction_count"] == 2

    def test_groups_by_month(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "Date,Description,Amount\n2024-01-10,Coffee,-5.50\n2024-02-05,Gym,-80.00\n",
            encoding="utf-8",
        )
        docs = list(BankCSVConnector(BankCSVConfig(csv_path=csv_file)).fetch_documents())
        assert len(docs) == 2
        months = {d.metadata["year_month"] for d in docs}
        assert months == {"2024-01", "2024-02"}

    def test_auto_detects_pt_columns(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "Data,Histórico,Valor\n10/01/2024,Supermercado,-250.00\n",
            encoding="utf-8",
        )
        docs = list(BankCSVConnector(BankCSVConfig(csv_path=csv_file)).fetch_documents())
        assert len(docs) == 1
        assert "Supermercado" in docs[0].text

    def test_multiple_csv_files_in_dir(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        (tmp_path / "jan.csv").write_text(
            "Date,Description,Amount\n2024-01-05,ATM,-100.00\n",
            encoding="utf-8",
        )
        (tmp_path / "feb.csv").write_text(
            "Date,Description,Amount\n2024-02-10,Rent,-1200.00\n",
            encoding="utf-8",
        )
        docs = list(BankCSVConnector(BankCSVConfig(csv_path=tmp_path)).fetch_documents())
        assert len(docs) == 2

    def test_flexible_date_dd_mm_yyyy(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "Date,Description,Amount\n15/03/2024,Shop,-42.00\n",
            encoding="utf-8",
        )
        docs = list(BankCSVConnector(BankCSVConfig(csv_path=csv_file)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].metadata["year_month"] == "2024-03"

    def test_metadata_totals_and_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.bank_csv_connector import (
            BankCSVConfig,
            BankCSVConnector,
        )

        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "Date,Description,Amount\n2024-01-10,Coffee,-5.50\n2024-01-15,Salary,3000.00\n",
            encoding="utf-8",
        )
        docs = list(
            BankCSVConnector(BankCSVConfig(csv_path=csv_file, currency="USD")).fetch_documents()
        )
        doc = docs[0]
        assert doc.metadata["total_debit"] == -5.50
        assert doc.metadata["total_credit"] == 3000.00
        assert doc.metadata["currency"] == "USD"
        assert doc.file_type == ".bank"


class TestDiscordConnector:
    def _make_export(self, export_dir: Path) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        channel: dict[str, Any] = {
            "guild": {"name": "Test Server"},
            "channel": {"name": "general"},
            "messages": [
                {
                    "id": "1",
                    "timestamp": "2024-01-15T10:00:00.000+00:00",
                    "content": "Hello world",
                    "author": {"name": "Alice#0001", "isBot": False},
                },
                {
                    "id": "2",
                    "timestamp": "2024-01-15T11:00:00.000+00:00",
                    "content": "Another message",
                    "author": {"name": "Bob#0002", "isBot": False},
                },
                {
                    "id": "3",
                    "timestamp": "2024-01-15T12:00:00.000+00:00",
                    "content": "",
                    "author": {"name": "Carol#0003", "isBot": False},
                },
                {
                    "id": "4",
                    "timestamp": "2024-01-15T13:00:00.000+00:00",
                    "content": "Bot message",
                    "author": {"name": "BotUser#0000", "isBot": True},
                },
            ],
        }
        (export_dir / "channel1.json").write_text(json.dumps(channel), encoding="utf-8")

    def test_yields_one_doc_per_channel_day(self, tmp_path: Path) -> None:
        from memorymesh.connectors.discord_connector import (
            DiscordConfig,
            DiscordConnector,
        )

        self._make_export(tmp_path)
        docs = list(DiscordConnector(DiscordConfig(export_path=tmp_path)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].file_type == ".discord"

    def test_metadata_guild_and_channel(self, tmp_path: Path) -> None:
        from memorymesh.connectors.discord_connector import (
            DiscordConfig,
            DiscordConnector,
        )

        self._make_export(tmp_path)
        docs = list(DiscordConnector(DiscordConfig(export_path=tmp_path)).fetch_documents())
        assert docs[0].metadata["channel"] == "general"
        assert docs[0].metadata["guild"] == "Test Server"
        assert docs[0].metadata["message_count"] == 2

    def test_bot_messages_excluded(self, tmp_path: Path) -> None:
        from memorymesh.connectors.discord_connector import (
            DiscordConfig,
            DiscordConnector,
        )

        self._make_export(tmp_path)
        docs = list(DiscordConnector(DiscordConfig(export_path=tmp_path)).fetch_documents())
        assert "Bot message" not in docs[0].text

    def test_missing_export_path_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.discord_connector import (
            DiscordConfig,
            DiscordConnector,
        )

        docs = list(
            DiscordConnector(DiscordConfig(export_path=tmp_path / "nonexistent")).fetch_documents()
        )
        assert docs == []

    def test_groups_by_date(self, tmp_path: Path) -> None:
        from memorymesh.connectors.discord_connector import (
            DiscordConfig,
            DiscordConnector,
        )

        channel: dict[str, Any] = {
            "guild": {"name": "Server"},
            "channel": {"name": "chat"},
            "messages": [
                {
                    "id": "1",
                    "timestamp": "2024-01-15T10:00:00.000+00:00",
                    "content": "Day 1",
                    "author": {"name": "Alice#0001", "isBot": False},
                },
                {
                    "id": "2",
                    "timestamp": "2024-01-16T10:00:00.000+00:00",
                    "content": "Day 2",
                    "author": {"name": "Alice#0001", "isBot": False},
                },
            ],
        }
        (tmp_path / "chan.json").write_text(json.dumps(channel), encoding="utf-8")
        docs = list(DiscordConnector(DiscordConfig(export_path=tmp_path)).fetch_documents())
        assert len(docs) == 2


class TestLinkedInConnector:
    def test_yields_connections_doc(self, tmp_path: Path) -> None:
        from memorymesh.connectors.linkedin_connector import (
            LinkedInConfig,
            LinkedInConnector,
        )

        (tmp_path / "Connections.csv").write_text(
            "Notes:\nThis is a note\n"
            "First Name,Last Name,Position,Company,Connected On\n"
            "Alice,Smith,Engineer,Acme Corp,01 Jan 2024\n"
            "Bob,Jones,Designer,Beta Inc,15 Mar 2024\n",
            encoding="utf-8",
        )
        cfg = LinkedInConfig(
            export_path=tmp_path,
            include_messages=False,
            include_posts=False,
        )
        docs = list(LinkedInConnector(cfg).fetch_documents())
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "connections"

    def test_connections_content(self, tmp_path: Path) -> None:
        from memorymesh.connectors.linkedin_connector import (
            LinkedInConfig,
            LinkedInConnector,
        )

        (tmp_path / "Connections.csv").write_text(
            "First Name,Last Name,Position,Company,Connected On\n"
            "Alice,Smith,Engineer,Acme Corp,01 Jan 2024\n",
            encoding="utf-8",
        )
        cfg = LinkedInConfig(
            export_path=tmp_path,
            include_messages=False,
            include_posts=False,
        )
        docs = list(LinkedInConnector(cfg).fetch_documents())
        assert "Alice Smith" in docs[0].text

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.linkedin_connector import (
            LinkedInConfig,
            LinkedInConnector,
        )

        (tmp_path / "Connections.csv").write_text(
            "First Name,Last Name,Position,Company,Connected On\nAlice,Smith,,Corp,01 Jan 2024\n",
            encoding="utf-8",
        )
        cfg = LinkedInConfig(
            export_path=tmp_path,
            include_messages=False,
            include_posts=False,
        )
        docs = list(LinkedInConnector(cfg).fetch_documents())
        assert all(d.file_type == ".linkedin" for d in docs)

    def test_yields_messages_per_conversation(self, tmp_path: Path) -> None:
        from memorymesh.connectors.linkedin_connector import (
            LinkedInConfig,
            LinkedInConnector,
        )

        (tmp_path / "messages.csv").write_text(
            "ConversationID,From,SentAt,MessageBody\n"
            "conv1,Alice,2024-01-01,Hello there\n"
            "conv1,Bob,2024-01-02,How are you\n"
            "conv2,Charlie,2024-01-03,Another thread\n",
            encoding="utf-8",
        )
        cfg = LinkedInConfig(
            export_path=tmp_path,
            include_connections=False,
            include_posts=False,
        )
        docs = list(LinkedInConnector(cfg).fetch_documents())
        assert len(docs) == 2
        assert all(d.metadata["type"] == "message" for d in docs)

    def test_no_files_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.linkedin_connector import (
            LinkedInConfig,
            LinkedInConnector,
        )

        cfg = LinkedInConfig(
            export_path=tmp_path,
            include_connections=False,
            include_messages=False,
            include_posts=False,
        )
        docs = list(LinkedInConnector(cfg).fetch_documents())
        assert docs == []


class TestYouTubeTakeoutConnector:
    def _make_history(self, base: Path) -> None:
        history = [
            {
                "header": "YouTube",
                "title": "Watched Python Tutorial",
                "titleUrl": "https://www.youtube.com/watch?v=abc",
                "subtitles": [
                    {
                        "name": "TechChannel",
                        "url": "https://www.youtube.com/channel/1",
                    }
                ],
                "time": "2024-01-15T10:00:00.000Z",
            },
            {
                "header": "YouTube",
                "title": "Watched Data Science Course",
                "titleUrl": "https://www.youtube.com/watch?v=def",
                "subtitles": [
                    {
                        "name": "DataChannel",
                        "url": "https://www.youtube.com/channel/2",
                    }
                ],
                "time": "2024-01-20T12:00:00.000Z",
            },
            {
                "header": "YouTube",
                "title": "Watched a video that has been removed",
                "subtitles": [],
                "time": "2024-01-21T12:00:00.000Z",
            },
        ]
        base.mkdir(parents=True, exist_ok=True)
        (base / "watch-history.json").write_text(json.dumps(history), encoding="utf-8")

    def test_yields_monthly_docs(self, tmp_path: Path) -> None:
        from memorymesh.connectors.youtube_connector import (
            YouTubeConfig,
            YouTubeTakeoutConnector,
        )

        self._make_history(tmp_path)
        cfg = YouTubeConfig(export_path=tmp_path, days_past=0)
        docs = list(YouTubeTakeoutConnector(cfg).fetch_documents())
        assert len(docs) == 1
        assert docs[0].file_type == ".youtube"

    def test_removed_videos_skipped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.youtube_connector import (
            YouTubeConfig,
            YouTubeTakeoutConnector,
        )

        self._make_history(tmp_path)
        cfg = YouTubeConfig(export_path=tmp_path, days_past=0)
        docs = list(YouTubeTakeoutConnector(cfg).fetch_documents())
        assert "removed" not in docs[0].text.lower()

    def test_metadata_contains_channels(self, tmp_path: Path) -> None:
        from memorymesh.connectors.youtube_connector import (
            YouTubeConfig,
            YouTubeTakeoutConnector,
        )

        self._make_history(tmp_path)
        cfg = YouTubeConfig(export_path=tmp_path, days_past=0)
        docs = list(YouTubeTakeoutConnector(cfg).fetch_documents())
        assert "channels" in docs[0].metadata
        assert len(docs[0].metadata["channels"]) > 0

    def test_year_month_in_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.youtube_connector import (
            YouTubeConfig,
            YouTubeTakeoutConnector,
        )

        self._make_history(tmp_path)
        cfg = YouTubeConfig(export_path=tmp_path, days_past=0)
        docs = list(YouTubeTakeoutConnector(cfg).fetch_documents())
        assert docs[0].metadata["year_month"] == "2024-01"

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.youtube_connector import (
            YouTubeConfig,
            YouTubeTakeoutConnector,
        )

        cfg = YouTubeConfig(export_path=tmp_path / "nonexistent", days_past=0)
        docs = list(YouTubeTakeoutConnector(cfg).fetch_documents())
        assert docs == []


class TestFacebookArchiveConnector:
    def _make_archive(self, base: Path) -> None:
        posts_dir = base / "your_posts"
        posts_dir.mkdir(parents=True)
        posts = [
            {
                "timestamp": 1704067200,
                "title": "",
                "data": [{"post": "Hello Facebook world!"}],
            },
            {
                "timestamp": 1704153600,
                "title": "",
                "data": [{"post": "Another day, another post."}],
            },
        ]
        (posts_dir / "your_posts_1.json").write_text(json.dumps(posts), encoding="utf-8")
        msg_dir = base / "messages" / "inbox" / "friend_12345"
        msg_dir.mkdir(parents=True)
        thread = {
            "title": "Alice and Bob",
            "participants": [{"name": "Alice"}, {"name": "Bob"}],
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1704067200000,
                    "content": "Hey Bob!",
                },
                {
                    "sender_name": "Bob",
                    "timestamp_ms": 1704067260000,
                    "content": "Hey Alice!",
                },
            ],
        }
        (msg_dir / "message_1.json").write_text(json.dumps(thread), encoding="utf-8")

    def test_yields_posts_and_messages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.facebook_connector import (
            FacebookArchiveConnector,
            FacebookConfig,
        )

        self._make_archive(tmp_path)
        docs = list(
            FacebookArchiveConnector(FacebookConfig(export_path=tmp_path)).fetch_documents()
        )
        types = {d.metadata.get("type") for d in docs}
        assert "post" in types
        assert "message" in types

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.facebook_connector import (
            FacebookArchiveConnector,
            FacebookConfig,
        )

        self._make_archive(tmp_path)
        docs = list(
            FacebookArchiveConnector(FacebookConfig(export_path=tmp_path)).fetch_documents()
        )
        assert all(d.file_type == ".facebook" for d in docs)

    def test_posts_only(self, tmp_path: Path) -> None:
        from memorymesh.connectors.facebook_connector import (
            FacebookArchiveConnector,
            FacebookConfig,
        )

        self._make_archive(tmp_path)
        docs = list(
            FacebookArchiveConnector(
                FacebookConfig(export_path=tmp_path, include_messages=False)
            ).fetch_documents()
        )
        assert all(d.metadata.get("type") == "post" for d in docs)

    def test_post_count(self, tmp_path: Path) -> None:
        from memorymesh.connectors.facebook_connector import (
            FacebookArchiveConnector,
            FacebookConfig,
        )

        self._make_archive(tmp_path)
        docs = list(
            FacebookArchiveConnector(
                FacebookConfig(export_path=tmp_path, include_messages=False)
            ).fetch_documents()
        )
        assert len(docs) == 2

    def test_missing_export_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.facebook_connector import (
            FacebookArchiveConnector,
            FacebookConfig,
        )

        docs = list(
            FacebookArchiveConnector(
                FacebookConfig(export_path=tmp_path / "nonexistent")
            ).fetch_documents()
        )
        assert docs == []


class TestInstagramConnector:
    def _make_archive(self, base: Path) -> None:
        posts_dir = base / "your_instagram_activity" / "content"
        posts_dir.mkdir(parents=True)
        posts = [
            {
                "media": [
                    {
                        "title": "Beautiful sunset",
                        "creation_timestamp": 1704067200,
                    }
                ]
            },
            {
                "media": [
                    {
                        "title": "Morning coffee",
                        "creation_timestamp": 1704153600,
                    }
                ]
            },
        ]
        (posts_dir / "posts_1.json").write_text(json.dumps(posts), encoding="utf-8")
        msg_dir = base / "your_instagram_activity" / "messages" / "inbox" / "friend_12345"
        msg_dir.mkdir(parents=True)
        thread = {
            "title": "Alice and Bob",
            "messages": [
                {
                    "sender_name": "Alice",
                    "timestamp_ms": 1704067200000,
                    "content": "Hey!",
                }
            ],
        }
        (msg_dir / "message_1.json").write_text(json.dumps(thread), encoding="utf-8")

    def test_yields_posts_and_messages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.instagram_connector import (
            InstagramConfig,
            InstagramConnector,
        )

        self._make_archive(tmp_path)
        docs = list(InstagramConnector(InstagramConfig(export_path=tmp_path)).fetch_documents())
        types = {d.metadata.get("type") for d in docs}
        assert "post" in types
        assert "message" in types

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.instagram_connector import (
            InstagramConfig,
            InstagramConnector,
        )

        self._make_archive(tmp_path)
        docs = list(InstagramConnector(InstagramConfig(export_path=tmp_path)).fetch_documents())
        assert all(d.file_type == ".instagram" for d in docs)

    def test_posts_only(self, tmp_path: Path) -> None:
        from memorymesh.connectors.instagram_connector import (
            InstagramConfig,
            InstagramConnector,
        )

        self._make_archive(tmp_path)
        docs = list(
            InstagramConnector(
                InstagramConfig(export_path=tmp_path, include_messages=False)
            ).fetch_documents()
        )
        assert all(d.metadata.get("type") == "post" for d in docs)

    def test_post_count(self, tmp_path: Path) -> None:
        from memorymesh.connectors.instagram_connector import (
            InstagramConfig,
            InstagramConnector,
        )

        self._make_archive(tmp_path)
        docs = list(
            InstagramConnector(
                InstagramConfig(export_path=tmp_path, include_messages=False)
            ).fetch_documents()
        )
        assert len(docs) == 2

    def test_missing_export_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.instagram_connector import (
            InstagramConfig,
            InstagramConnector,
        )

        docs = list(
            InstagramConnector(
                InstagramConfig(export_path=tmp_path / "nonexistent")
            ).fetch_documents()
        )
        assert docs == []


class TestPinterestConnector:
    def _make_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "pinterest-export.csv"
        csv_path.write_text(
            "Pin Link,Board,Note,Title,Description,Image Link,Saved Date,Source Link\n"
            "https://pin.it/1,Travel,,Eiffel Tower,The famous landmark,"
            "https://img.com/1.jpg,2024-01-01,https://paris.com\n"
            "https://pin.it/2,Travel,,Louvre Museum,Famous art museum,"
            "https://img.com/2.jpg,2024-01-02,https://louvre.fr\n"
            "https://pin.it/3,Food,,Pizza Recipe,Classic margherita,"
            "https://img.com/3.jpg,2024-01-03,https://food.com\n",
            encoding="utf-8",
        )
        return csv_path

    def test_yields_one_doc_per_board(self, tmp_path: Path) -> None:
        from memorymesh.connectors.pinterest_connector import (
            PinterestConfig,
            PinterestConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(PinterestConnector(PinterestConfig(export_path=csv_path)).fetch_documents())
        assert len(docs) == 2

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.pinterest_connector import (
            PinterestConfig,
            PinterestConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(PinterestConnector(PinterestConfig(export_path=csv_path)).fetch_documents())
        assert all(d.file_type == ".pinterest" for d in docs)

    def test_board_names_in_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.pinterest_connector import (
            PinterestConfig,
            PinterestConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(PinterestConnector(PinterestConfig(export_path=csv_path)).fetch_documents())
        boards = {d.metadata["board"] for d in docs}
        assert "Travel" in boards
        assert "Food" in boards

    def test_pins_grouped_in_text(self, tmp_path: Path) -> None:
        from memorymesh.connectors.pinterest_connector import (
            PinterestConfig,
            PinterestConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(PinterestConnector(PinterestConfig(export_path=csv_path)).fetch_documents())
        travel_doc = next(d for d in docs if d.metadata["board"] == "Travel")
        assert "Eiffel Tower" in travel_doc.text
        assert "Louvre Museum" in travel_doc.text

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.pinterest_connector import (
            PinterestConfig,
            PinterestConnector,
        )

        docs = list(
            PinterestConnector(PinterestConfig(export_path=tmp_path / "nope.csv")).fetch_documents()
        )
        assert docs == []


class TestGoodreadsConnector:
    def _make_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "goodreads_library_export.csv"
        csv_path.write_text(
            "Book Id,Title,Author,My Rating,Exclusive Shelf,Bookshelves,"
            "Date Read,Number of Pages,Year Published,My Review\n"
            "1,Python Crash Course,Eric Matthes,5,read,read,"
            "2024-01-15,543,2019,Great intro book\n"
            "2,Clean Code,Robert C. Martin,4,read,read,"
            "2024-02-01,464,2008,\n"
            "3,Abandoned Book,Some Author,0,abandoned,abandoned,,300,2020,\n",
            encoding="utf-8",
        )
        return csv_path

    def test_yields_one_doc_per_book(self, tmp_path: Path) -> None:
        from memorymesh.connectors.goodreads_connector import (
            GoodreadsConfig,
            GoodreadsConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(GoodreadsConnector(GoodreadsConfig(export_path=csv_path)).fetch_documents())
        assert len(docs) == 3

    def test_shelf_filter(self, tmp_path: Path) -> None:
        from memorymesh.connectors.goodreads_connector import (
            GoodreadsConfig,
            GoodreadsConnector,
        )

        csv_path = self._make_csv(tmp_path)
        cfg = GoodreadsConfig(export_path=csv_path, shelves=["read"])
        docs = list(GoodreadsConnector(cfg).fetch_documents())
        assert len(docs) == 2
        assert all(d.metadata["shelf"] == "read" for d in docs)

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.goodreads_connector import (
            GoodreadsConfig,
            GoodreadsConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(GoodreadsConnector(GoodreadsConfig(export_path=csv_path)).fetch_documents())
        assert all(d.file_type == ".goodreads" for d in docs)

    def test_review_included_in_text(self, tmp_path: Path) -> None:
        from memorymesh.connectors.goodreads_connector import (
            GoodreadsConfig,
            GoodreadsConnector,
        )

        csv_path = self._make_csv(tmp_path)
        docs = list(GoodreadsConnector(GoodreadsConfig(export_path=csv_path)).fetch_documents())
        python_doc = next(d for d in docs if d.metadata["title"] == "Python Crash Course")
        assert "Great intro book" in python_doc.text

    def test_missing_file_yields_nothing(self) -> None:
        from memorymesh.connectors.goodreads_connector import (
            GoodreadsConfig,
            GoodreadsConnector,
        )

        docs = list(
            GoodreadsConnector(
                GoodreadsConfig(export_path=Path("/no/such/file.csv"))
            ).fetch_documents()
        )
        assert docs == []


class TestAnkiConnector:
    def _make_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "collection.anki2"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE col (id INTEGER, decks TEXT)")
        decks_json = json.dumps({"1": {"name": "Default"}, "2": {"name": "Science"}})
        conn.execute("INSERT INTO col VALUES (1, ?)", (decks_json,))
        conn.execute(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, flds TEXT, tags TEXT, mod INTEGER)"
        )
        conn.execute(
            "INSERT INTO notes VALUES (1, '<b>Capital of France</b>\x1fParis', 'geo', 1704067200)"
        )
        conn.execute("INSERT INTO notes VALUES (2, 'H2O\x1fWater', 'chemistry', 1704067200)")
        conn.execute("CREATE TABLE cards (id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER)")
        conn.execute("INSERT INTO cards VALUES (1, 1, 1)")
        conn.execute("INSERT INTO cards VALUES (2, 2, 2)")
        conn.commit()
        conn.close()
        return db_path

    def test_yields_one_doc_per_note(self, tmp_path: Path) -> None:
        from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector

        db_path = self._make_db(tmp_path)
        docs = list(AnkiConnector(AnkiConfig(db_path=db_path)).fetch_documents())
        assert len(docs) == 2

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector

        db_path = self._make_db(tmp_path)
        docs = list(AnkiConnector(AnkiConfig(db_path=db_path)).fetch_documents())
        assert all(d.file_type == ".anki" for d in docs)

    def test_qa_format_in_text(self, tmp_path: Path) -> None:
        from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector

        db_path = self._make_db(tmp_path)
        docs = list(AnkiConnector(AnkiConfig(db_path=db_path)).fetch_documents())
        texts = " ".join(d.text for d in docs)
        assert "Q:" in texts
        assert "A:" in texts

    def test_html_stripped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector

        db_path = self._make_db(tmp_path)
        docs = list(AnkiConnector(AnkiConfig(db_path=db_path)).fetch_documents())
        assert all("<b>" not in d.text for d in docs)

    def test_missing_db_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.anki_connector import AnkiConfig, AnkiConnector

        docs = list(AnkiConnector(AnkiConfig(db_path=tmp_path / "nope.anki2")).fetch_documents())
        assert docs == []


class TestLogseqConnector:
    def _make_vault(self, tmp_path: Path) -> Path:
        vault = tmp_path / "graph"
        (vault / "pages").mkdir(parents=True)
        (vault / "journals").mkdir(parents=True)
        (vault / "pages" / "Python.md").write_text(
            "title:: Python Notes\n"
            "- Python is a great language\n"
            "- [[Programming]] is fun\n"
            "- ((12345678-1234-1234-1234-123456789012)) block ref\n",
            encoding="utf-8",
        )
        (vault / "journals" / "2024_01_15.md").write_text(
            "- Had a great day coding\n- Learned about [[Logseq]]\n",
            encoding="utf-8",
        )
        return vault

    def test_yields_pages_and_journals(self, tmp_path: Path) -> None:
        from memorymesh.connectors.logseq_connector import (
            LogseqConfig,
            LogseqConnector,
        )

        vault = self._make_vault(tmp_path)
        docs = list(LogseqConnector(LogseqConfig(vault_path=vault)).fetch_documents())
        assert len(docs) == 2

    def test_journal_flag(self, tmp_path: Path) -> None:
        from memorymesh.connectors.logseq_connector import (
            LogseqConfig,
            LogseqConnector,
        )

        vault = self._make_vault(tmp_path)
        docs = list(LogseqConnector(LogseqConfig(vault_path=vault)).fetch_documents())
        journals = [d for d in docs if d.metadata["is_journal"]]
        pages = [d for d in docs if not d.metadata["is_journal"]]
        assert len(journals) == 1
        assert len(pages) == 1

    def test_properties_extracted(self, tmp_path: Path) -> None:
        from memorymesh.connectors.logseq_connector import (
            LogseqConfig,
            LogseqConnector,
        )

        vault = self._make_vault(tmp_path)
        docs = list(LogseqConnector(LogseqConfig(vault_path=vault)).fetch_documents())
        page = next(d for d in docs if not d.metadata["is_journal"])
        assert "title" in page.metadata["properties"]

    def test_wikilinks_in_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.logseq_connector import (
            LogseqConfig,
            LogseqConnector,
        )

        vault = self._make_vault(tmp_path)
        docs = list(LogseqConnector(LogseqConfig(vault_path=vault)).fetch_documents())
        page = next(d for d in docs if not d.metadata["is_journal"])
        assert "Programming" in page.metadata["wikilinks"]

    def test_block_ref_replaced(self, tmp_path: Path) -> None:
        from memorymesh.connectors.logseq_connector import (
            LogseqConfig,
            LogseqConnector,
        )

        vault = self._make_vault(tmp_path)
        docs = list(LogseqConnector(LogseqConfig(vault_path=vault)).fetch_documents())
        page = next(d for d in docs if not d.metadata["is_journal"])
        assert "[block]" in page.text
        assert "12345678-1234-1234-1234-123456789012" not in page.text


class TestLetterboxdConnector:
    def _make_export(self, tmp_path: Path) -> None:
        (tmp_path / "diary.csv").write_text(
            "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n"
            "2024-01-15,Inception,2010,https://letterboxd.com/...,4.5,No,,2024-01-15\n"
            "2024-01-20,The Matrix,1999,https://letterboxd.com/...,5,No,,2024-01-20\n",
            encoding="utf-8",
        )
        (tmp_path / "reviews.csv").write_text(
            "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Review,Watched Date\n"
            "2024-01-15,Inception,2010,https://letterboxd.com/...,4.5,No,"
            "Mind-blowing film,2024-01-15\n",
            encoding="utf-8",
        )
        (tmp_path / "watchlist.csv").write_text(
            "Date,Name,Year,Letterboxd URI\n"
            "2024-01-01,Parasite,2019,https://letterboxd.com/...\n"
            "2024-01-02,Spirited Away,2001,https://letterboxd.com/...\n",
            encoding="utf-8",
        )

    def test_yields_diary_and_watchlist(self, tmp_path: Path) -> None:
        from memorymesh.connectors.letterboxd_connector import (
            LetterboxdConfig,
            LetterboxdConnector,
        )

        self._make_export(tmp_path)
        docs = list(LetterboxdConnector(LetterboxdConfig(export_path=tmp_path)).fetch_documents())
        assert len(docs) == 3

    def test_review_merged_into_diary(self, tmp_path: Path) -> None:
        from memorymesh.connectors.letterboxd_connector import (
            LetterboxdConfig,
            LetterboxdConnector,
        )

        self._make_export(tmp_path)
        docs = list(LetterboxdConnector(LetterboxdConfig(export_path=tmp_path)).fetch_documents())
        inception = next(d for d in docs if d.metadata.get("name") == "Inception")
        assert inception.metadata["has_review"] is True
        assert "Mind-blowing" in inception.text

    def test_watchlist_doc_metadata(self, tmp_path: Path) -> None:
        from memorymesh.connectors.letterboxd_connector import (
            LetterboxdConfig,
            LetterboxdConnector,
        )

        self._make_export(tmp_path)
        docs = list(LetterboxdConnector(LetterboxdConfig(export_path=tmp_path)).fetch_documents())
        watchlist = next(d for d in docs if d.metadata.get("type") == "watchlist")
        assert watchlist.metadata["film_count"] == 2

    def test_file_type(self, tmp_path: Path) -> None:
        from memorymesh.connectors.letterboxd_connector import (
            LetterboxdConfig,
            LetterboxdConnector,
        )

        self._make_export(tmp_path)
        docs = list(LetterboxdConnector(LetterboxdConfig(export_path=tmp_path)).fetch_documents())
        assert all(d.file_type == ".letterboxd" for d in docs)

    def test_empty_export_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.letterboxd_connector import (
            LetterboxdConfig,
            LetterboxdConnector,
        )

        docs = list(LetterboxdConnector(LetterboxdConfig(export_path=tmp_path)).fetch_documents())
        assert docs == []


_REDDIT_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_REDDIT_POST = {
    "id": "abc123",
    "title": "Why Python is awesome",
    "selftext": "Because it is very readable.",
    "subreddit": "Python",
    "score": 42,
    "created_utc": 1704067200.0,
    "url": "https://reddit.com/r/Python/abc123",
}

_REDDIT_COMMENT = {
    "id": "xyz456",
    "link_title": "Some discussion thread",
    "body": "I totally agree with this.",
    "subreddit": "Python",
    "score": 10,
    "created_utc": 1704067200.0,
}

_REDDIT_LISTING_POST = {
    "data": {
        "children": [{"data": _REDDIT_POST}],
        "after": None,
    }
}

_REDDIT_LISTING_COMMENT = {
    "data": {
        "children": [{"data": _REDDIT_COMMENT}],
        "after": None,
    }
}

_REDDIT_LISTING_EMPTY = {"data": {"children": [], "after": None}}


class TestRedditConnector:
    def test_yields_post_document(self) -> None:
        from memorymesh.connectors.reddit_connector import (
            RedditConfig,
            RedditConnector,
        )

        responses = [_MockResp(_REDDIT_LISTING_POST)]
        with mock.patch(_REDDIT_URLOPEN, side_effect=iter(responses)), mock.patch("time.sleep"):
            docs = list(
                RedditConnector(
                    RedditConfig(
                        username="testuser",
                        include_comments=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".reddit"
        assert "Why Python is awesome" in docs[0].text
        assert docs[0].metadata["post_id"] == "abc123"

    def test_yields_comment_document(self) -> None:
        from memorymesh.connectors.reddit_connector import (
            RedditConfig,
            RedditConnector,
        )

        responses = [_MockResp(_REDDIT_LISTING_COMMENT)]
        with mock.patch(_REDDIT_URLOPEN, side_effect=iter(responses)), mock.patch("time.sleep"):
            docs = list(
                RedditConnector(
                    RedditConfig(
                        username="testuser",
                        include_posts=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["comment_id"] == "xyz456"
        assert "I totally agree" in docs[0].text

    def test_subreddit_filter(self) -> None:
        from memorymesh.connectors.reddit_connector import (
            RedditConfig,
            RedditConnector,
        )

        responses = [_MockResp(_REDDIT_LISTING_POST)]
        with mock.patch(_REDDIT_URLOPEN, side_effect=iter(responses)), mock.patch("time.sleep"):
            docs = list(
                RedditConnector(
                    RedditConfig(
                        username="testuser",
                        include_comments=False,
                        subreddits=["learnpython"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_empty_listing_yields_nothing(self) -> None:
        from memorymesh.connectors.reddit_connector import (
            RedditConfig,
            RedditConnector,
        )

        responses = [_MockResp(_REDDIT_LISTING_EMPTY), _MockResp(_REDDIT_LISTING_EMPTY)]
        with mock.patch(_REDDIT_URLOPEN, side_effect=iter(responses)), mock.patch("time.sleep"):
            docs = list(
                RedditConnector(RedditConfig(username="testuser", days_past=0)).fetch_documents()
            )
        assert docs == []

    def test_pagination(self) -> None:
        from memorymesh.connectors.reddit_connector import (
            RedditConfig,
            RedditConnector,
        )

        post2 = dict(_REDDIT_POST, id="post2", title="Second post")
        page1 = {
            "data": {
                "children": [{"data": _REDDIT_POST}],
                "after": "t3_abc123",
            }
        }
        page2 = {
            "data": {
                "children": [{"data": post2}],
                "after": None,
            }
        }
        responses = [_MockResp(page1), _MockResp(page2)]
        with mock.patch(_REDDIT_URLOPEN, side_effect=iter(responses)), mock.patch("time.sleep"):
            docs = list(
                RedditConnector(
                    RedditConfig(
                        username="testuser",
                        include_comments=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 2


_LASTFM_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_LASTFM_RESP = {
    "recenttracks": {
        "track": [
            {
                "name": "Bohemian Rhapsody",
                "artist": {"#text": "Queen"},
                "album": {"#text": "A Night at the Opera"},
                "date": {"uts": "1704067200"},
            },
            {
                "name": "Hotel California",
                "artist": {"#text": "Eagles"},
                "album": {"#text": "Hotel California"},
                "date": {"uts": "1704070800"},
                "@attr": {},
            },
            {
                "name": "Currently Playing",
                "artist": {"#text": "Someone"},
                "album": {"#text": "Album"},
                "@attr": {"nowplaying": "true"},
            },
        ],
        "@attr": {"totalPages": "1", "page": "1", "total": "2"},
    }
}


class TestLastFmConnector:
    def test_yields_monthly_doc(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.lastfm_connector import (
            LastFmConfig,
            LastFmConnector,
        )

        responses = [_MockResp(_LASTFM_RESP)]
        with mock.patch(_LASTFM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LastFmConnector(
                    LastFmConfig(
                        api_key=SecretStr("test_key"),
                        username="testuser",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".lastfm"

    def test_nowplaying_track_skipped(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.lastfm_connector import (
            LastFmConfig,
            LastFmConnector,
        )

        responses = [_MockResp(_LASTFM_RESP)]
        with mock.patch(_LASTFM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LastFmConnector(
                    LastFmConfig(
                        api_key=SecretStr("test_key"),
                        username="testuser",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "Currently Playing" not in docs[0].text

    def test_metadata_top_artists(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.lastfm_connector import (
            LastFmConfig,
            LastFmConnector,
        )

        responses = [_MockResp(_LASTFM_RESP)]
        with mock.patch(_LASTFM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LastFmConnector(
                    LastFmConfig(
                        api_key=SecretStr("test_key"),
                        username="testuser",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "top_artists" in docs[0].metadata
        assert "Queen" in docs[0].metadata["top_artists"]

    def test_track_count_in_metadata(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.lastfm_connector import (
            LastFmConfig,
            LastFmConnector,
        )

        responses = [_MockResp(_LASTFM_RESP)]
        with mock.patch(_LASTFM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LastFmConnector(
                    LastFmConfig(
                        api_key=SecretStr("test_key"),
                        username="testuser",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs[0].metadata["track_count"] == 2

    def test_api_error_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.lastfm_connector import (
            LastFmConfig,
            LastFmConnector,
        )

        with mock.patch(_LASTFM_URLOPEN, side_effect=Exception("network error")):
            docs = list(
                LastFmConnector(
                    LastFmConfig(
                        api_key=SecretStr("test_key"),
                        username="testuser",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_STEAM_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_STEAM_LIBRARY = {
    "response": {
        "games": [
            {
                "appid": 570,
                "name": "Dota 2",
                "playtime_forever": 6000,
                "rtime_last_played": 1704067200,
            },
            {
                "appid": 999,
                "name": "Unplayed Game",
                "playtime_forever": 0,
                "rtime_last_played": 0,
            },
        ]
    }
}

_STEAM_RECENT = {"response": {"games": [{"appid": 570, "name": "Dota 2", "playtime_2weeks": 120}]}}


class TestSteamConnector:
    def test_yields_game_and_recent_docs(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector

        responses = [_MockResp(_STEAM_LIBRARY), _MockResp(_STEAM_RECENT)]
        with mock.patch(_STEAM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                SteamConnector(
                    SteamConfig(
                        api_key=SecretStr("test_key"),
                        steam_id="76561198000000001",
                        min_playtime_hours=0.1,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 2

    def test_file_type(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector

        responses = [_MockResp(_STEAM_LIBRARY), _MockResp(_STEAM_RECENT)]
        with mock.patch(_STEAM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                SteamConnector(
                    SteamConfig(
                        api_key=SecretStr("test_key"),
                        steam_id="76561198000000001",
                    )
                ).fetch_documents()
            )
        assert all(d.file_type == ".steam" for d in docs)

    def test_playtime_filter(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector

        responses = [_MockResp(_STEAM_LIBRARY), _MockResp(_STEAM_RECENT)]
        with mock.patch(_STEAM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                SteamConnector(
                    SteamConfig(
                        api_key=SecretStr("test_key"),
                        steam_id="76561198000000001",
                        min_playtime_hours=1000.0,
                    )
                ).fetch_documents()
            )
        game_docs = [d for d in docs if d.metadata.get("type") != "recent_games"]
        assert len(game_docs) == 0

    def test_metadata_playtime_hours(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector

        responses = [_MockResp(_STEAM_LIBRARY), _MockResp(_STEAM_RECENT)]
        with mock.patch(_STEAM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                SteamConnector(
                    SteamConfig(
                        api_key=SecretStr("test_key"),
                        steam_id="76561198000000001",
                        min_playtime_hours=0.1,
                    )
                ).fetch_documents()
            )
        dota = next(d for d in docs if d.metadata.get("name") == "Dota 2")
        assert dota.metadata["playtime_hours"] == 100.0

    def test_recent_summary_doc(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.steam_connector import SteamConfig, SteamConnector

        responses = [_MockResp(_STEAM_LIBRARY), _MockResp(_STEAM_RECENT)]
        with mock.patch(_STEAM_URLOPEN, side_effect=iter(responses)):
            docs = list(
                SteamConnector(
                    SteamConfig(
                        api_key=SecretStr("test_key"),
                        steam_id="76561198000000001",
                        min_playtime_hours=0.1,
                    )
                ).fetch_documents()
            )
        recent = next(d for d in docs if d.metadata.get("type") == "recent_games")
        assert "Dota 2" in recent.text
        assert recent.path.name == "recent.steam"


_HN_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_HN_STORY_HIT = {
    "objectID": "12345",
    "title": "Show HN: My cool project",
    "url": "https://github.com/me/project",
    "points": 150,
    "num_comments": 30,
    "story_text": None,
    "created_at": "2024-01-15T10:00:00.000Z",
    "created_at_i": 1704067200,
}

_HN_COMMENT_HIT = {
    "objectID": "67890",
    "story_title": "Ask HN: Best tools for RAG?",
    "story_url": "https://news.ycombinator.com/item?id=99",
    "comment_text": "<p>I recommend using <b>ChromaDB</b>.</p>",
    "points": 5,
    "created_at": "2024-01-15T11:00:00.000Z",
    "created_at_i": 1704071200,
}

_HN_STORIES_RESP = {"hits": [_HN_STORY_HIT], "nbPages": 1, "page": 0}
_HN_COMMENTS_RESP = {"hits": [_HN_COMMENT_HIT], "nbPages": 1, "page": 0}
_HN_EMPTY_RESP = {"hits": [], "nbPages": 0, "page": 0}


class TestHackerNewsConnector:
    def test_yields_story_document(self) -> None:
        from memorymesh.connectors.hackernews_connector import (
            HackerNewsConfig,
            HackerNewsConnector,
        )

        responses = [_MockResp(_HN_STORIES_RESP)]
        with mock.patch(_HN_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HackerNewsConnector(
                    HackerNewsConfig(
                        username="testuser",
                        include_comments=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".hn"
        assert "Show HN: My cool project" in docs[0].text
        assert docs[0].metadata["object_id"] == "12345"

    def test_yields_comment_document(self) -> None:
        from memorymesh.connectors.hackernews_connector import (
            HackerNewsConfig,
            HackerNewsConnector,
        )

        responses = [_MockResp(_HN_COMMENTS_RESP)]
        with mock.patch(_HN_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HackerNewsConnector(
                    HackerNewsConfig(
                        username="testuser",
                        include_stories=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["object_id"] == "67890"

    def test_html_stripped_from_comment(self) -> None:
        from memorymesh.connectors.hackernews_connector import (
            HackerNewsConfig,
            HackerNewsConnector,
        )

        responses = [_MockResp(_HN_COMMENTS_RESP)]
        with mock.patch(_HN_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HackerNewsConnector(
                    HackerNewsConfig(
                        username="testuser",
                        include_stories=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "<p>" not in docs[0].text
        assert "ChromaDB" in docs[0].text

    def test_empty_response_yields_nothing(self) -> None:
        from memorymesh.connectors.hackernews_connector import (
            HackerNewsConfig,
            HackerNewsConnector,
        )

        responses = [_MockResp(_HN_EMPTY_RESP), _MockResp(_HN_EMPTY_RESP)]
        with mock.patch(_HN_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HackerNewsConnector(
                    HackerNewsConfig(username="testuser", days_past=0)
                ).fetch_documents()
            )
        assert docs == []

    def test_max_items_respected(self) -> None:
        from memorymesh.connectors.hackernews_connector import (
            HackerNewsConfig,
            HackerNewsConnector,
        )

        responses = [_MockResp(_HN_STORIES_RESP)]
        with mock.patch(_HN_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HackerNewsConnector(
                    HackerNewsConfig(
                        username="testuser",
                        include_comments=False,
                        days_past=0,
                        max_items=1,
                    )
                ).fetch_documents()
            )
        assert len(docs) <= 1


_POCKET_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_POCKET_ITEM = {
    "item_id": "987654321",
    "resolved_title": "How to Build a RAG System",
    "resolved_url": "https://example.com/rag-tutorial",
    "excerpt": "A comprehensive guide to retrieval-augmented generation.",
    "time_added": "1704067200",
    "time_read": "1704153600",
    "word_count": "2500",
    "tags": {"python": {"tag": "python"}, "ai": {"tag": "ai"}},
}

_POCKET_RESP = {"list": {"987654321": _POCKET_ITEM}, "status": 1}
_POCKET_EMPTY = {"list": {}, "status": 1}


class TestPocketConnector:
    def test_yields_article_document(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.pocket_connector import (
            PocketConfig,
            PocketConnector,
        )

        responses = [_MockResp(_POCKET_RESP), _MockResp(_POCKET_EMPTY)]
        with mock.patch(_POCKET_URLOPEN, side_effect=iter(responses)):
            docs = list(
                PocketConnector(
                    PocketConfig(
                        consumer_key=SecretStr("ck123"),
                        access_token=SecretStr("at456"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".pocket"
        assert "How to Build a RAG System" in docs[0].text

    def test_metadata_tags(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.pocket_connector import (
            PocketConfig,
            PocketConnector,
        )

        responses = [_MockResp(_POCKET_RESP), _MockResp(_POCKET_EMPTY)]
        with mock.patch(_POCKET_URLOPEN, side_effect=iter(responses)):
            docs = list(
                PocketConnector(
                    PocketConfig(
                        consumer_key=SecretStr("ck123"),
                        access_token=SecretStr("at456"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "python" in docs[0].metadata["tags"]
        assert docs[0].metadata["is_read"] is True

    def test_item_id_in_path(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.pocket_connector import (
            PocketConfig,
            PocketConnector,
        )

        responses = [_MockResp(_POCKET_RESP), _MockResp(_POCKET_EMPTY)]
        with mock.patch(_POCKET_URLOPEN, side_effect=iter(responses)):
            docs = list(
                PocketConnector(
                    PocketConfig(
                        consumer_key=SecretStr("ck123"),
                        access_token=SecretStr("at456"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "987654321" in str(docs[0].path)

    def test_empty_list_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.pocket_connector import (
            PocketConfig,
            PocketConnector,
        )

        responses = [_MockResp(_POCKET_EMPTY)]
        with mock.patch(_POCKET_URLOPEN, side_effect=iter(responses)):
            docs = list(
                PocketConnector(
                    PocketConfig(
                        consumer_key=SecretStr("ck123"),
                        access_token=SecretStr("at456"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_api_failure_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.pocket_connector import (
            PocketConfig,
            PocketConnector,
        )

        with mock.patch(_POCKET_URLOPEN, side_effect=Exception("network error")):
            docs = list(
                PocketConnector(
                    PocketConfig(
                        consumer_key=SecretStr("ck123"),
                        access_token=SecretStr("at456"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_LINEAR_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_LINEAR_ISSUE = {
    "id": "issue-uuid-1",
    "identifier": "ENG-42",
    "title": "Fix the memory leak in indexer",
    "description": "The indexer leaks memory when processing large PDFs.",
    "state": {"name": "In Progress"},
    "priority": 2,
    "createdAt": "2024-01-10T08:00:00.000Z",
    "updatedAt": "2024-01-15T10:00:00.000Z",
    "team": {"id": "team-uuid-1", "name": "Engineering"},
    "comments": {
        "nodes": [
            {
                "body": "Looking into this now.",
                "createdAt": "2024-01-12T09:00:00.000Z",
                "user": {"name": "Alice"},
            }
        ]
    },
}

_LINEAR_RESP = {
    "data": {
        "issues": {
            "nodes": [_LINEAR_ISSUE],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }
    }
}

_LINEAR_EMPTY = {
    "data": {
        "issues": {
            "nodes": [],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }
    }
}


class TestLinearConnector:
    def test_yields_issue_document(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.linear_connector import (
            LinearConfig,
            LinearConnector,
        )

        responses = [_MockResp(_LINEAR_RESP)]
        with mock.patch(_LINEAR_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LinearConnector(
                    LinearConfig(api_key=SecretStr("lin_api_test"), days_past=0)
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".linear"
        assert "ENG-42" in docs[0].text

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.linear_connector import (
            LinearConfig,
            LinearConnector,
        )

        responses = [_MockResp(_LINEAR_RESP)]
        with mock.patch(_LINEAR_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LinearConnector(
                    LinearConfig(api_key=SecretStr("lin_api_test"), days_past=0)
                ).fetch_documents()
            )
        assert docs[0].metadata["identifier"] == "ENG-42"
        assert docs[0].metadata["state"] == "In Progress"
        assert docs[0].metadata["team"] == "Engineering"

    def test_comment_in_text(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.linear_connector import (
            LinearConfig,
            LinearConnector,
        )

        responses = [_MockResp(_LINEAR_RESP)]
        with mock.patch(_LINEAR_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LinearConnector(
                    LinearConfig(api_key=SecretStr("lin_api_test"), days_past=0)
                ).fetch_documents()
            )
        assert "Looking into this now" in docs[0].text

    def test_state_filter(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.linear_connector import (
            LinearConfig,
            LinearConnector,
        )

        responses = [_MockResp(_LINEAR_RESP)]
        with mock.patch(_LINEAR_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LinearConnector(
                    LinearConfig(
                        api_key=SecretStr("lin_api_test"),
                        states=["Done"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_empty_response_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.linear_connector import (
            LinearConfig,
            LinearConnector,
        )

        responses = [_MockResp(_LINEAR_EMPTY)]
        with mock.patch(_LINEAR_URLOPEN, side_effect=iter(responses)):
            docs = list(
                LinearConnector(
                    LinearConfig(api_key=SecretStr("lin_api_test"), days_past=0)
                ).fetch_documents()
            )
        assert docs == []


_JIRA_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_JIRA_SEARCH_RESP: dict[str, Any] = {
    "issues": [
        {
            "key": "ENG-1",
            "fields": {
                "summary": "Fix login bug",
                "status": {"name": "In Progress"},
                "project": {"name": "Engineering"},
                "priority": {"name": "High"},
                "assignee": {"displayName": "Alice"},
                "created": "2024-01-01T00:00:00.000Z",
                "updated": "2024-01-02T00:00:00.000Z",
                "description": {
                    "type": "doc",
                    "content": [{"type": "text", "text": "Description body"}],
                },
                "comment": {
                    "comments": [
                        {
                            "author": {"displayName": "Bob"},
                            "created": "2024-01-02T10:00:00.000Z",
                            "body": {
                                "type": "doc",
                                "content": [{"type": "text", "text": "A comment"}],
                            },
                        }
                    ]
                },
            },
        }
    ],
    "total": 1,
    "startAt": 0,
    "maxResults": 50,
}

_JIRA_EMPTY: dict[str, Any] = {
    "issues": [],
    "total": 0,
    "startAt": 0,
    "maxResults": 50,
}


class TestJiraConnector:
    def test_fetch_single_issue(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.jira_connector import JiraConfig, JiraConnector

        responses = [_MockResp(_JIRA_SEARCH_RESP), _MockResp(_JIRA_EMPTY)]
        with mock.patch(_JIRA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                JiraConnector(
                    JiraConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].path.name == "ENG-1.jira"
        assert docs[0].file_type == ".jira"
        assert "Fix login bug" in docs[0].text

    def test_adf_text_extraction(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.jira_connector import JiraConfig, JiraConnector

        responses = [_MockResp(_JIRA_SEARCH_RESP), _MockResp(_JIRA_EMPTY)]
        with mock.patch(_JIRA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                JiraConnector(
                    JiraConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "Description body" in docs[0].text
        assert "A comment" in docs[0].text

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.jira_connector import JiraConfig, JiraConnector

        responses = [_MockResp(_JIRA_SEARCH_RESP), _MockResp(_JIRA_EMPTY)]
        with mock.patch(_JIRA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                JiraConnector(
                    JiraConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["key"] == "ENG-1"
        assert meta["status"] == "In Progress"
        assert meta["assignee"] == "Alice"

    def test_empty_response(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.jira_connector import JiraConfig, JiraConnector

        responses = [_MockResp(_JIRA_EMPTY)]
        with mock.patch(_JIRA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                JiraConnector(
                    JiraConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_CONFLUENCE_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_CONFLUENCE_PAGES_RESP: dict[str, Any] = {
    "results": [
        {
            "id": "12345",
            "title": "Architecture Overview",
            "space": {"key": "ENG", "name": "Engineering"},
            "history": {
                "createdBy": {"displayName": "Alice"},
                "createdDate": "2024-01-01T00:00:00.000Z",
            },
            "version": {"when": "2024-06-01T00:00:00.000Z"},
            "body": {"storage": {"value": "<p>This page describes the architecture.</p>"}},
        }
    ],
    "size": 1,
}

_CONFLUENCE_EMPTY: dict[str, Any] = {"results": [], "size": 0}


class TestConfluenceConnector:
    def test_fetch_page(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.confluence_connector import (
            ConfluenceConfig,
            ConfluenceConnector,
        )

        responses = [
            _MockResp(_CONFLUENCE_PAGES_RESP),
            _MockResp(_CONFLUENCE_EMPTY),
        ]
        with mock.patch(_CONFLUENCE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ConfluenceConnector(
                    ConfluenceConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        space_keys=["ENG"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".confluence"
        assert "Architecture Overview" in docs[0].text

    def test_html_stripping(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.confluence_connector import (
            ConfluenceConfig,
            ConfluenceConnector,
        )

        responses = [
            _MockResp(_CONFLUENCE_PAGES_RESP),
            _MockResp(_CONFLUENCE_EMPTY),
        ]
        with mock.patch(_CONFLUENCE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ConfluenceConnector(
                    ConfluenceConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        space_keys=["ENG"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "<p>" not in docs[0].text
        assert "architecture" in docs[0].text.lower()

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.confluence_connector import (
            ConfluenceConfig,
            ConfluenceConnector,
        )

        responses = [
            _MockResp(_CONFLUENCE_PAGES_RESP),
            _MockResp(_CONFLUENCE_EMPTY),
        ]
        with mock.patch(_CONFLUENCE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ConfluenceConnector(
                    ConfluenceConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        space_keys=["ENG"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["id"] == "12345"
        assert meta["space_key"] == "ENG"

    def test_empty_space_returns_no_docs(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.confluence_connector import (
            ConfluenceConfig,
            ConfluenceConnector,
        )

        responses = [_MockResp(_CONFLUENCE_EMPTY)]
        with mock.patch(_CONFLUENCE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ConfluenceConnector(
                    ConfluenceConfig(
                        base_url="https://test.atlassian.net",
                        email="me@example.com",
                        api_token=SecretStr("tok"),
                        space_keys=["ENG"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_GITLAB_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_GITLAB_ISSUES: list[dict[str, Any]] = [
    {
        "iid": 1,
        "title": "Fix null pointer",
        "description": "NullPointerException in auth module",
        "state": "opened",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-06-01T00:00:00Z",
    }
]

_GITLAB_MRS: list[dict[str, Any]] = [
    {
        "iid": 42,
        "title": "Add OAuth2 support",
        "description": "Adds OAuth2 login flow",
        "state": "opened",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-06-01T00:00:00Z",
    }
]


class TestGitLabConnector:
    def test_fetch_issues(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.gitlab_connector import (
            GitLabConfig,
            GitLabConnector,
        )

        responses = [
            _MockResp(_GITLAB_ISSUES),
            _MockResp([]),  # second page empty
            _MockResp([]),  # MR endpoint (fetch_mrs=False so not called, but safe)
        ]
        with mock.patch(_GITLAB_URLOPEN, side_effect=iter(responses)):
            docs = list(
                GitLabConnector(
                    GitLabConfig(
                        api_token=SecretStr("glpat-test"),
                        projects=["mygroup/myrepo"],
                        fetch_issues=True,
                        fetch_mrs=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".gitlab"
        assert "Fix null pointer" in docs[0].text

    def test_fetch_mrs(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.gitlab_connector import (
            GitLabConfig,
            GitLabConnector,
        )

        responses = [_MockResp(_GITLAB_MRS), _MockResp([])]
        with mock.patch(_GITLAB_URLOPEN, side_effect=iter(responses)):
            docs = list(
                GitLabConnector(
                    GitLabConfig(
                        api_token=SecretStr("glpat-test"),
                        projects=["mygroup/myrepo"],
                        fetch_issues=False,
                        fetch_mrs=True,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "mr"

    def test_metadata_contains_iid(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.gitlab_connector import (
            GitLabConfig,
            GitLabConnector,
        )

        responses = [_MockResp(_GITLAB_ISSUES), _MockResp([])]
        with mock.patch(_GITLAB_URLOPEN, side_effect=iter(responses)):
            docs = list(
                GitLabConnector(
                    GitLabConfig(
                        api_token=SecretStr("glpat-test"),
                        projects=["mygroup/myrepo"],
                        fetch_issues=True,
                        fetch_mrs=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs[0].metadata["iid"] == 1

    def test_no_projects_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.gitlab_connector import (
            GitLabConfig,
            GitLabConnector,
        )

        docs = list(
            GitLabConnector(
                GitLabConfig(
                    api_token=SecretStr("glpat-test"),
                    projects=[],
                    days_past=0,
                )
            ).fetch_documents()
        )
        assert docs == []


_TRELLO_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_TRELLO_BOARDS: list[dict[str, Any]] = [{"id": "board1", "name": "MyBoard", "closed": False}]

_TRELLO_BOARD_META: dict[str, Any] = {
    "id": "board1",
    "name": "MyBoard",
    "closed": False,
}

_TRELLO_LISTS: list[dict[str, Any]] = [{"id": "list1", "name": "To Do"}]

_TRELLO_CARDS: list[dict[str, Any]] = [
    {
        "id": "card1",
        "name": "Write tests",
        "desc": "Add unit tests for all connectors",
        "idList": "list1",
        "labels": [{"name": "priority"}],
        "due": None,
        "dateLastActivity": "2024-06-01T00:00:00.000Z",
        "closed": False,
    }
]


class TestTrelloConnector:
    def test_fetch_cards(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.trello_connector import TrelloConfig, TrelloConnector

        responses = [
            _MockResp(_TRELLO_BOARD_META),  # fetch_board
            _MockResp(_TRELLO_LISTS),  # fetch_lists
            _MockResp(_TRELLO_CARDS),  # fetch_cards
        ]
        with mock.patch(_TRELLO_URLOPEN, side_effect=iter(responses)):
            docs = list(
                TrelloConnector(
                    TrelloConfig(
                        api_key=SecretStr("key"),
                        token=SecretStr("tok"),
                        board_ids=["board1"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".trello"
        assert "Write tests" in docs[0].text

    def test_list_name_included(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.trello_connector import TrelloConfig, TrelloConnector

        responses = [
            _MockResp(_TRELLO_BOARD_META),
            _MockResp(_TRELLO_LISTS),
            _MockResp(_TRELLO_CARDS),
        ]
        with mock.patch(_TRELLO_URLOPEN, side_effect=iter(responses)):
            docs = list(
                TrelloConnector(
                    TrelloConfig(
                        api_key=SecretStr("key"),
                        token=SecretStr("tok"),
                        board_ids=["board1"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "To Do" in docs[0].text

    def test_label_in_metadata(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.trello_connector import TrelloConfig, TrelloConnector

        responses = [
            _MockResp(_TRELLO_BOARD_META),
            _MockResp(_TRELLO_LISTS),
            _MockResp(_TRELLO_CARDS),
        ]
        with mock.patch(_TRELLO_URLOPEN, side_effect=iter(responses)):
            docs = list(
                TrelloConnector(
                    TrelloConfig(
                        api_key=SecretStr("key"),
                        token=SecretStr("tok"),
                        board_ids=["board1"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "priority" in docs[0].metadata["labels"]

    def test_closed_card_filtered(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.trello_connector import TrelloConfig, TrelloConnector

        closed_cards = [dict(_TRELLO_CARDS[0], closed=True)]
        responses = [
            _MockResp(_TRELLO_BOARD_META),
            _MockResp(_TRELLO_LISTS),
            _MockResp(closed_cards),
        ]
        with mock.patch(_TRELLO_URLOPEN, side_effect=iter(responses)):
            docs = list(
                TrelloConnector(
                    TrelloConfig(
                        api_key=SecretStr("key"),
                        token=SecretStr("tok"),
                        board_ids=["board1"],
                        include_closed=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_ASANA_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_ASANA_WORKSPACES: dict[str, Any] = {"data": [{"gid": "ws1", "name": "My Workspace"}]}

_ASANA_TASKS: dict[str, Any] = {
    "data": [
        {
            "gid": "task1",
            "name": "Write documentation",
            "notes": "Add API docs",
            "completed": False,
            "due_on": "2024-07-01",
            "modified_at": "2024-06-01T00:00:00.000Z",
            "created_at": "2024-05-01T00:00:00.000Z",
            "assignee": {"name": "Alice"},
        }
    ],
    "next_page": None,
}

_ASANA_EMPTY: dict[str, Any] = {"data": [], "next_page": None}


class TestAsanaConnector:
    def test_fetch_task_from_workspace(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.asana_connector import AsanaConfig, AsanaConnector

        responses = [_MockResp(_ASANA_WORKSPACES), _MockResp(_ASANA_TASKS)]
        with mock.patch(_ASANA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AsanaConnector(
                    AsanaConfig(
                        access_token=SecretStr("1/test:abc"),
                        assigned_to_me=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".asana"
        assert "Write documentation" in docs[0].text

    def test_task_from_project(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.asana_connector import AsanaConfig, AsanaConnector

        responses = [_MockResp(_ASANA_TASKS)]
        with mock.patch(_ASANA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AsanaConnector(
                    AsanaConfig(
                        access_token=SecretStr("1/test:abc"),
                        project_gids=["proj1"],
                        assigned_to_me=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].metadata["gid"] == "task1"

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.asana_connector import AsanaConfig, AsanaConnector

        responses = [_MockResp(_ASANA_TASKS)]
        with mock.patch(_ASANA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AsanaConnector(
                    AsanaConfig(
                        access_token=SecretStr("1/test:abc"),
                        project_gids=["proj1"],
                        assigned_to_me=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["due_on"] == "2024-07-01"
        assert meta["assignee"] == "Alice"

    def test_completed_task_filtered_by_default(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.asana_connector import AsanaConfig, AsanaConnector

        completed_tasks: dict[str, Any] = {
            "data": [dict(_ASANA_TASKS["data"][0], completed=True)],
            "next_page": None,
        }
        responses = [_MockResp(completed_tasks)]
        with mock.patch(_ASANA_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AsanaConnector(
                    AsanaConfig(
                        access_token=SecretStr("1/test:abc"),
                        project_gids=["proj1"],
                        assigned_to_me=False,
                        include_completed=False,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


class TestRoamConnector:
    def _write_export(self, tmp_path: Path, data: Any) -> Path:
        p = tmp_path / "roam.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_fetch_page(self, tmp_path: Path) -> None:
        from memorymesh.connectors.roam_connector import RoamConfig, RoamConnector

        export = [
            {
                "title": "My Page",
                "create-time": 1700000000000,
                "edit-time": 1700000000000,
                "children": [{"string": "Hello [[World]]"}],
            }
        ]
        path = self._write_export(tmp_path, export)
        docs = list(RoamConnector(RoamConfig(export_path=path, days_past=0)).fetch_documents())
        assert len(docs) == 1
        assert docs[0].file_type == ".roam"

    def test_roam_syntax_stripped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.roam_connector import RoamConfig, RoamConnector

        export = [
            {
                "title": "Syntax",
                "edit-time": 1700000000000,
                "children": [{"string": "[[PageRef]] and #tag and {{[[template]]}}"}],
            }
        ]
        path = self._write_export(tmp_path, export)
        docs = list(RoamConnector(RoamConfig(export_path=path, days_past=0)).fetch_documents())
        assert "[[" not in docs[0].text
        assert "{{" not in docs[0].text

    def test_empty_blocks_skipped(self, tmp_path: Path) -> None:
        from memorymesh.connectors.roam_connector import RoamConfig, RoamConnector

        export = [{"title": "Empty", "edit-time": 1700000000000, "children": []}]
        path = self._write_export(tmp_path, export)
        docs = list(RoamConnector(RoamConfig(export_path=path, days_past=0)).fetch_documents())
        assert docs == []

    def test_nested_blocks_flattened(self, tmp_path: Path) -> None:
        from memorymesh.connectors.roam_connector import RoamConfig, RoamConnector

        export = [
            {
                "title": "Nested",
                "edit-time": 1700000000000,
                "children": [
                    {
                        "string": "Parent",
                        "children": [{"string": "Child block"}],
                    }
                ],
            }
        ]
        path = self._write_export(tmp_path, export)
        docs = list(RoamConnector(RoamConfig(export_path=path, days_past=0)).fetch_documents())
        assert "Parent" in docs[0].text
        assert "Child block" in docs[0].text


class TestAppleNotesConnector:
    def test_skipped_on_non_macos(self, tmp_path: Path) -> None:
        import platform

        from memorymesh.connectors.apple_notes_connector import (
            AppleNotesConfig,
            AppleNotesConnector,
        )

        if platform.system() == "Darwin":
            pytest.skip("Not testing skip logic on macOS")

        db = tmp_path / "NoteStore.sqlite"
        db.touch()
        docs = list(
            AppleNotesConnector(AppleNotesConfig(db_path=db, days_past=0)).fetch_documents()
        )
        assert docs == []

    def test_missing_db_yields_nothing(self) -> None:
        from memorymesh.connectors.apple_notes_connector import (
            AppleNotesConfig,
            AppleNotesConnector,
        )

        docs = list(
            AppleNotesConnector(
                AppleNotesConfig(db_path=Path("/nonexistent/NoteStore.sqlite"), days_past=0)
            ).fetch_documents()
        )
        assert docs == []

    def test_fetch_note_on_macos(self, tmp_path: Path) -> None:
        import platform

        if platform.system() != "Darwin":
            pytest.skip("Apple Notes only available on macOS")

        from memorymesh.connectors.apple_notes_connector import (
            AppleNotesConfig,
            AppleNotesConnector,
        )

        db = tmp_path / "NoteStore.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(
            """CREATE TABLE ZICCLOUDSYNCINGOBJECT (
                Z_PK INTEGER PRIMARY KEY,
                ZTITLE1 TEXT,
                ZSNIPPET TEXT,
                ZCREATIONDATE REAL,
                ZMODIFICATIONDATE REAL,
                ZMARKEDFORDELETION INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO ZICCLOUDSYNCINGOBJECT VALUES (1, 'My Note', 'Note body', 0.0, 0.0, 0)"
        )
        conn.commit()
        conn.close()

        docs = list(
            AppleNotesConnector(AppleNotesConfig(db_path=db, days_past=0)).fetch_documents()
        )
        assert len(docs) == 1
        assert docs[0].file_type == ".applenotes"
        assert "My Note" in docs[0].text

    def test_note_without_snippet_skipped(self, tmp_path: Path) -> None:
        import platform

        if platform.system() != "Darwin":
            pytest.skip("Apple Notes only available on macOS")

        from memorymesh.connectors.apple_notes_connector import (
            AppleNotesConfig,
            AppleNotesConnector,
        )

        db = tmp_path / "NoteStore.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(
            """CREATE TABLE ZICCLOUDSYNCINGOBJECT (
                Z_PK INTEGER PRIMARY KEY,
                ZTITLE1 TEXT,
                ZSNIPPET TEXT,
                ZCREATIONDATE REAL,
                ZMODIFICATIONDATE REAL,
                ZMARKEDFORDELETION INTEGER
            )"""
        )
        conn.execute("INSERT INTO ZICCLOUDSYNCINGOBJECT VALUES (1, 'Empty', NULL, 0.0, 0.0, 0)")
        conn.commit()
        conn.close()

        docs = list(
            AppleNotesConnector(AppleNotesConfig(db_path=db, days_past=0)).fetch_documents()
        )
        assert docs == []


_MASTODON_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_MASTODON_ACCOUNT: dict[str, Any] = {"id": "123456"}

_MASTODON_STATUSES: list[dict[str, Any]] = [
    {
        "id": "111",
        "created_at": "2024-06-01T12:00:00.000Z",
        "content": "<p>Hello <b>Mastodon</b>!</p>",
        "language": "en",
        "visibility": "public",
        "reblog": None,
        "in_reply_to_id": None,
        "tags": [{"name": "intro"}],
    }
]


class TestMastodonConnector:
    def test_fetch_toot(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.mastodon_connector import (
            MastodonConfig,
            MastodonConnector,
        )

        responses = [_MockResp(_MASTODON_ACCOUNT), _MockResp(_MASTODON_STATUSES)]
        with mock.patch(
            _MASTODON_URLOPEN,
            side_effect=iter(responses),
        ):
            docs = list(
                MastodonConnector(
                    MastodonConfig(
                        instance_url="https://mastodon.social",
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".mastodon"

    def test_html_stripped(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.mastodon_connector import (
            MastodonConfig,
            MastodonConnector,
        )

        responses = [_MockResp(_MASTODON_ACCOUNT), _MockResp(_MASTODON_STATUSES)]
        with mock.patch(_MASTODON_URLOPEN, side_effect=iter(responses)):
            docs = list(
                MastodonConnector(
                    MastodonConfig(
                        instance_url="https://mastodon.social",
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "<p>" not in docs[0].text
        assert "Hello" in docs[0].text

    def test_boost_skipped(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.mastodon_connector import (
            MastodonConfig,
            MastodonConnector,
        )

        boosted = [dict(_MASTODON_STATUSES[0], reblog={"id": "999"})]
        responses = [_MockResp(_MASTODON_ACCOUNT), _MockResp(boosted)]
        with mock.patch(_MASTODON_URLOPEN, side_effect=iter(responses)):
            docs = list(
                MastodonConnector(
                    MastodonConfig(
                        instance_url="https://mastodon.social",
                        access_token=SecretStr("tok"),
                        skip_boosts=True,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_tag_in_metadata(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.mastodon_connector import (
            MastodonConfig,
            MastodonConnector,
        )

        responses = [_MockResp(_MASTODON_ACCOUNT), _MockResp(_MASTODON_STATUSES)]
        with mock.patch(_MASTODON_URLOPEN, side_effect=iter(responses)):
            docs = list(
                MastodonConnector(
                    MastodonConfig(
                        instance_url="https://mastodon.social",
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "intro" in docs[0].metadata["tags"]


_BLUESKY_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_BLUESKY_SESSION: dict[str, Any] = {
    "accessJwt": "eyJtest",
    "did": "did:plc:abc123",
}

_BLUESKY_FEED: dict[str, Any] = {
    "feed": [
        {
            "post": {
                "uri": "at://did:plc:abc123/app.bsky.feed.post/rkey1",
                "cid": "cid1",
                "record": {
                    "text": "Hello Bluesky!",
                    "createdAt": "2024-06-01T12:00:00.000Z",
                },
                "replyCount": 2,
                "repostCount": 5,
                "likeCount": 10,
            },
            "reason": None,
        }
    ],
    "cursor": None,
}


class TestBlueSkyConnector:
    def test_fetch_post(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.bluesky_connector import (
            BlueSkyConfig,
            BlueSkyConnector,
        )

        responses = [_MockResp(_BLUESKY_SESSION), _MockResp(_BLUESKY_FEED)]
        with mock.patch(_BLUESKY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                BlueSkyConnector(
                    BlueSkyConfig(
                        handle="user.bsky.social",
                        password=SecretStr("app-pass"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".bluesky"
        assert docs[0].text == "Hello Bluesky!"

    def test_repost_skipped(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.bluesky_connector import (
            BlueSkyConfig,
            BlueSkyConnector,
        )

        feed_with_repost = {
            "feed": [
                {
                    "post": _BLUESKY_FEED["feed"][0]["post"],
                    "reason": {"$type": "app.bsky.feed.defs#reasonRepost"},
                }
            ],
            "cursor": None,
        }
        responses = [_MockResp(_BLUESKY_SESSION), _MockResp(feed_with_repost)]
        with mock.patch(_BLUESKY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                BlueSkyConnector(
                    BlueSkyConfig(
                        handle="user.bsky.social",
                        password=SecretStr("app-pass"),
                        skip_reposts=True,
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.bluesky_connector import (
            BlueSkyConfig,
            BlueSkyConnector,
        )

        responses = [_MockResp(_BLUESKY_SESSION), _MockResp(_BLUESKY_FEED)]
        with mock.patch(_BLUESKY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                BlueSkyConnector(
                    BlueSkyConfig(
                        handle="user.bsky.social",
                        password=SecretStr("app-pass"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["like_count"] == 10
        assert meta["repost_count"] == 5

    def test_auth_failure_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.bluesky_connector import (
            BlueSkyConfig,
            BlueSkyConnector,
        )

        with mock.patch(_BLUESKY_URLOPEN, side_effect=Exception("Auth failed")):
            docs = list(
                BlueSkyConnector(
                    BlueSkyConfig(
                        handle="user.bsky.social",
                        password=SecretStr("wrong"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_CHESS_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_CHESS_ARCHIVES: dict[str, Any] = {
    "archives": ["https://api.chess.com/pub/player/testuser/games/2024/06"]
}

_CHESS_GAMES: dict[str, Any] = {
    "games": [
        {
            "pgn": (
                '[White "testuser"]\n[Black "opponent"]\n'
                '[Result "1-0"]\n[Opening "Ruy Lopez"]\n'
                '[TimeControl "600"]\n\n1. e4 e5 *'
            )
        }
    ]
}


class TestChessComConnector:
    def test_fetch_monthly_archive(self) -> None:
        from memorymesh.connectors.chesscom_connector import (
            ChessComConfig,
            ChessComConnector,
        )

        responses = [_MockResp(_CHESS_ARCHIVES), _MockResp(_CHESS_GAMES)]
        with mock.patch(_CHESS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ChessComConnector(
                    ChessComConfig(username="testuser", days_past=0)
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".chess"

    def test_pgn_headers_parsed(self) -> None:
        from memorymesh.connectors.chesscom_connector import (
            ChessComConfig,
            ChessComConnector,
        )

        responses = [_MockResp(_CHESS_ARCHIVES), _MockResp(_CHESS_GAMES)]
        with mock.patch(_CHESS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ChessComConnector(
                    ChessComConfig(username="testuser", days_past=0)
                ).fetch_documents()
            )
        assert "testuser" in docs[0].text
        assert "1-0" in docs[0].text

    def test_metadata_contains_year_month(self) -> None:
        from memorymesh.connectors.chesscom_connector import (
            ChessComConfig,
            ChessComConnector,
        )

        responses = [_MockResp(_CHESS_ARCHIVES), _MockResp(_CHESS_GAMES)]
        with mock.patch(_CHESS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ChessComConnector(
                    ChessComConfig(username="testuser", days_past=0)
                ).fetch_documents()
            )
        assert docs[0].metadata["year"] == 2024
        assert docs[0].metadata["month"] == 6

    def test_empty_archive_yields_nothing(self) -> None:
        from memorymesh.connectors.chesscom_connector import (
            ChessComConfig,
            ChessComConnector,
        )

        empty_games: dict[str, Any] = {"games": []}
        responses = [_MockResp(_CHESS_ARCHIVES), _MockResp(empty_games)]
        with mock.patch(_CHESS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                ChessComConnector(
                    ChessComConfig(username="testuser", days_past=0)
                ).fetch_documents()
            )
        assert docs == []


_DUOLINGO_DATA: dict[str, Any] = {
    "username": "testuser",
    "site_streak": 42,
    "total_xp": 5000,
    "languages": [
        {
            "language_string": "Spanish",
            "xp": 3000,
            "level": 10,
            "crowns": 50,
            "skills": [
                {"name": "Greetings"},
                {"name": "Food"},
            ],
        },
        {
            "language_string": "French",
            "xp": 2000,
            "level": 7,
            "crowns": 30,
            "skills": [],
        },
    ],
}


class TestDuolingoConnector:
    def _write_export(self, tmp_path: Path, data: Any) -> Path:
        p = tmp_path / "duolingo.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_fetch_languages(self, tmp_path: Path) -> None:
        from memorymesh.connectors.duolingo_connector import (
            DuolingoConfig,
            DuolingoConnector,
        )

        path = self._write_export(tmp_path, _DUOLINGO_DATA)
        docs = list(DuolingoConnector(DuolingoConfig(export_path=path)).fetch_documents())
        # 2 languages + 1 summary
        assert len(docs) == 3

    def test_language_doc_content(self, tmp_path: Path) -> None:
        from memorymesh.connectors.duolingo_connector import (
            DuolingoConfig,
            DuolingoConnector,
        )

        path = self._write_export(tmp_path, _DUOLINGO_DATA)
        docs = list(DuolingoConnector(DuolingoConfig(export_path=path)).fetch_documents())
        lang_docs = [d for d in docs if d.path.name != "summary.duolingo"]
        names = {d.metadata["language"] for d in lang_docs}
        assert "Spanish" in names
        assert "French" in names

    def test_summary_doc(self, tmp_path: Path) -> None:
        from memorymesh.connectors.duolingo_connector import (
            DuolingoConfig,
            DuolingoConnector,
        )

        path = self._write_export(tmp_path, _DUOLINGO_DATA)
        docs = list(DuolingoConnector(DuolingoConfig(export_path=path)).fetch_documents())
        summary = next(d for d in docs if d.path.name == "summary.duolingo")
        assert "42" in summary.text
        assert summary.metadata["streak"] == 42

    def test_file_not_found_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.duolingo_connector import (
            DuolingoConfig,
            DuolingoConnector,
        )

        docs = list(
            DuolingoConnector(
                DuolingoConfig(export_path=tmp_path / "missing.json")
            ).fetch_documents()
        )
        assert docs == []


_FEEDLY_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_FEEDLY_PROFILE: dict[str, Any] = {"id": "user/abc123/profile"}

_FEEDLY_ENTRIES: dict[str, Any] = {
    "items": [
        {
            "id": "entry/abc123",
            "title": "Understanding Transformers",
            "alternate": [{"href": "https://example.com/transformers"}],
            "origin": {"title": "ML Blog"},
            "published": 1700000000000,
            "content": {"content": "<p>A deep dive into transformer models.</p>"},
            "tags": [{"label": "machine-learning"}],
        }
    ],
    "continuation": None,
}

_FEEDLY_EMPTY: dict[str, Any] = {"items": [], "continuation": None}


class TestFeedlyConnector:
    def test_fetch_article(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.feedly_connector import (
            FeedlyConfig,
            FeedlyConnector,
        )

        responses = [
            _MockResp(_FEEDLY_PROFILE),
            _MockResp(_FEEDLY_ENTRIES),
        ]
        with mock.patch(_FEEDLY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                FeedlyConnector(
                    FeedlyConfig(
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".feedly"
        assert "Transformers" in docs[0].text

    def test_html_stripped(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.feedly_connector import (
            FeedlyConfig,
            FeedlyConnector,
        )

        responses = [_MockResp(_FEEDLY_PROFILE), _MockResp(_FEEDLY_ENTRIES)]
        with mock.patch(_FEEDLY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                FeedlyConnector(
                    FeedlyConfig(
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "<p>" not in docs[0].text
        assert "transformer models" in docs[0].text.lower()

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.feedly_connector import (
            FeedlyConfig,
            FeedlyConnector,
        )

        responses = [_MockResp(_FEEDLY_PROFILE), _MockResp(_FEEDLY_ENTRIES)]
        with mock.patch(_FEEDLY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                FeedlyConnector(
                    FeedlyConfig(
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["feed_title"] == "ML Blog"
        assert "machine-learning" in meta["tags"]

    def test_empty_stream_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.feedly_connector import (
            FeedlyConfig,
            FeedlyConnector,
        )

        responses = [_MockResp(_FEEDLY_PROFILE), _MockResp(_FEEDLY_EMPTY)]
        with mock.patch(_FEEDLY_URLOPEN, side_effect=iter(responses)):
            docs = list(
                FeedlyConnector(
                    FeedlyConfig(
                        access_token=SecretStr("tok"),
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert docs == []


_HYPOTHESIS_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_HYPOTHESIS_ROWS: dict[str, Any] = {
    "rows": [
        {
            "id": "ann-abc123",
            "uri": "https://example.com/paper.pdf",
            "created": "2024-01-01T00:00:00.000Z",
            "updated": "2024-06-01T00:00:00.000Z",
            "tags": ["research", "NLP"],
            "target": [
                {
                    "selector": [
                        {
                            "type": "TextQuoteSelector",
                            "exact": "Attention is all you need",
                        }
                    ]
                }
            ],
            "body": [
                {
                    "type": "TextualBody",
                    "value": "Key paper for transformer architecture",
                }
            ],
        }
    ],
    "total": 1,
}

_HYPOTHESIS_EMPTY: dict[str, Any] = {"rows": [], "total": 0}


class TestHypothesisConnector:
    def test_fetch_annotation(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.hypothesis_connector import (
            HypothesisConfig,
            HypothesisConnector,
        )

        responses = [_MockResp(_HYPOTHESIS_ROWS), _MockResp(_HYPOTHESIS_EMPTY)]
        with mock.patch(_HYPOTHESIS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HypothesisConnector(
                    HypothesisConfig(api_key=SecretStr("tok"), days_past=0)
                ).fetch_documents()
            )
        assert len(docs) == 1
        assert docs[0].file_type == ".hypothesis"

    def test_quote_and_note_in_text(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.hypothesis_connector import (
            HypothesisConfig,
            HypothesisConnector,
        )

        responses = [_MockResp(_HYPOTHESIS_ROWS), _MockResp(_HYPOTHESIS_EMPTY)]
        with mock.patch(_HYPOTHESIS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HypothesisConnector(
                    HypothesisConfig(api_key=SecretStr("tok"), days_past=0)
                ).fetch_documents()
            )
        assert "Attention is all you need" in docs[0].text
        assert "transformer architecture" in docs[0].text

    def test_metadata_tags(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.hypothesis_connector import (
            HypothesisConfig,
            HypothesisConnector,
        )

        responses = [_MockResp(_HYPOTHESIS_ROWS), _MockResp(_HYPOTHESIS_EMPTY)]
        with mock.patch(_HYPOTHESIS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HypothesisConnector(
                    HypothesisConfig(api_key=SecretStr("tok"), days_past=0)
                ).fetch_documents()
            )
        assert "research" in docs[0].metadata["tags"]
        assert "NLP" in docs[0].metadata["tags"]

    def test_empty_response_yields_nothing(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.hypothesis_connector import (
            HypothesisConfig,
            HypothesisConnector,
        )

        responses = [_MockResp(_HYPOTHESIS_EMPTY)]
        with mock.patch(_HYPOTHESIS_URLOPEN, side_effect=iter(responses)):
            docs = list(
                HypothesisConnector(
                    HypothesisConfig(api_key=SecretStr("tok"), days_past=0)
                ).fetch_documents()
            )
        assert docs == []


_GARMIN_ACTIVITIES: list[dict[str, Any]] = [
    {
        "activityId": 12345,
        "activityType": "running",
        "startTimeLocal": "2024-06-01 07:00:00",
        "duration": 3600.0,
        "distance": 10000.0,
        "averageHR": 145,
    },
    {
        "activityId": 12346,
        "activityType": "cycling",
        "startTimeLocal": "2024-06-02 08:00:00",
        "duration": 5400.0,
        "distance": 40000.0,
        "averageHR": 130,
    },
]


class TestGarminConnector:
    def _write_export_dir(self, tmp_path: Path) -> Path:
        export_dir = tmp_path / "garmin_export"
        export_dir.mkdir()
        (export_dir / "summarizedActivities.json").write_text(
            json.dumps(_GARMIN_ACTIVITIES), encoding="utf-8"
        )
        return export_dir

    def _write_export_zip(self, tmp_path: Path) -> Path:
        import zipfile

        zip_path = tmp_path / "garmin.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "DI_CONNECT/summarizedActivities.json",
                json.dumps(_GARMIN_ACTIVITIES),
            )
        return zip_path

    def test_fetch_from_dir(self, tmp_path: Path) -> None:
        from memorymesh.connectors.garmin_connector import (
            GarminConfig,
            GarminConnector,
        )

        export_dir = self._write_export_dir(tmp_path)
        docs = list(
            GarminConnector(GarminConfig(export_path=export_dir, days_past=0)).fetch_documents()
        )
        assert len(docs) == 2
        assert docs[0].file_type == ".garmin"

    def test_fetch_from_zip(self, tmp_path: Path) -> None:
        from memorymesh.connectors.garmin_connector import (
            GarminConfig,
            GarminConnector,
        )

        zip_path = self._write_export_zip(tmp_path)
        docs = list(
            GarminConnector(GarminConfig(export_path=zip_path, days_past=0)).fetch_documents()
        )
        assert len(docs) == 2

    def test_metadata_fields(self, tmp_path: Path) -> None:
        from memorymesh.connectors.garmin_connector import (
            GarminConfig,
            GarminConnector,
        )

        export_dir = self._write_export_dir(tmp_path)
        docs = list(
            GarminConnector(GarminConfig(export_path=export_dir, days_past=0)).fetch_documents()
        )
        meta = docs[0].metadata
        assert meta["activity_id"] == "12345"
        assert meta["activity_type"] == "running"
        assert meta["avg_hr"] == 145

    def test_missing_path_yields_nothing(self, tmp_path: Path) -> None:
        from memorymesh.connectors.garmin_connector import (
            GarminConfig,
            GarminConnector,
        )

        docs = list(
            GarminConnector(
                GarminConfig(
                    export_path=tmp_path / "nonexistent",
                    days_past=0,
                )
            ).fetch_documents()
        )
        assert docs == []


_AIRTABLE_URLOPEN = "memorymesh.connectors._http.urllib.request.urlopen"

_AIRTABLE_TABLES: dict[str, Any] = {
    "tables": [
        {"id": "tbl1", "name": "Tasks"},
        {"id": "tbl2", "name": "Projects"},
    ]
}

_AIRTABLE_RECORDS: dict[str, Any] = {
    "records": [
        {
            "id": "rec1",
            "createdTime": "2024-06-01T00:00:00.000Z",
            "fields": {
                "Name": "Build MVP",
                "Status": "In Progress",
                "Priority": "High",
            },
        }
    ],
    "offset": None,
}

_AIRTABLE_EMPTY: dict[str, Any] = {"records": [], "offset": None}


class TestAirtableConnector:
    def test_fetch_record(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.airtable_connector import (
            AirtableConfig,
            AirtableConnector,
        )

        responses = [
            _MockResp(_AIRTABLE_TABLES),
            _MockResp(_AIRTABLE_RECORDS),
            _MockResp(_AIRTABLE_EMPTY),
            _MockResp(_AIRTABLE_RECORDS),
            _MockResp(_AIRTABLE_EMPTY),
        ]
        with mock.patch(_AIRTABLE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AirtableConnector(
                    AirtableConfig(
                        access_token=SecretStr("patXXX"),
                        base_id="appXXX",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) >= 1
        assert docs[0].file_type == ".airtable"

    def test_table_filter(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.airtable_connector import (
            AirtableConfig,
            AirtableConnector,
        )

        responses = [
            _MockResp(_AIRTABLE_TABLES),
            _MockResp(_AIRTABLE_RECORDS),
            _MockResp(_AIRTABLE_EMPTY),
        ]
        with mock.patch(_AIRTABLE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AirtableConnector(
                    AirtableConfig(
                        access_token=SecretStr("patXXX"),
                        base_id="appXXX",
                        table_names=["Tasks"],
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert len(docs) == 1

    def test_metadata_fields(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.airtable_connector import (
            AirtableConfig,
            AirtableConnector,
        )

        responses = [
            _MockResp(_AIRTABLE_TABLES),
            _MockResp(_AIRTABLE_RECORDS),
            _MockResp(_AIRTABLE_EMPTY),
            _MockResp(_AIRTABLE_EMPTY),
        ]
        with mock.patch(_AIRTABLE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AirtableConnector(
                    AirtableConfig(
                        access_token=SecretStr("patXXX"),
                        base_id="appXXX",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        meta = docs[0].metadata
        assert meta["record_id"] == "rec1"
        assert meta["table"] == "Tasks"

    def test_fields_in_text(self) -> None:
        from pydantic import SecretStr

        from memorymesh.connectors.airtable_connector import (
            AirtableConfig,
            AirtableConnector,
        )

        responses = [
            _MockResp(_AIRTABLE_TABLES),
            _MockResp(_AIRTABLE_RECORDS),
            _MockResp(_AIRTABLE_EMPTY),
            _MockResp(_AIRTABLE_EMPTY),
        ]
        with mock.patch(_AIRTABLE_URLOPEN, side_effect=iter(responses)):
            docs = list(
                AirtableConnector(
                    AirtableConfig(
                        access_token=SecretStr("patXXX"),
                        base_id="appXXX",
                        days_past=0,
                    )
                ).fetch_documents()
            )
        assert "Build MVP" in docs[0].text
        assert "In Progress" in docs[0].text


# Keep all tests synchronous - no asyncio needed for these connectors.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
