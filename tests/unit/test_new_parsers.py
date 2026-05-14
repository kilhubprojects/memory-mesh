"""Unit tests for Phase B parsers: Obsidian, Notion, Conversations, Email,
Calendar, and Browser History.
"""

from __future__ import annotations

import json
import mailbox
import sqlite3
import unittest.mock as mock
from pathlib import Path

import pytest

# B1 - ObsidianParser


@pytest.fixture()
def obsidian_note(tmp_path: Path) -> Path:
    """An Obsidian Markdown file with frontmatter, backlinks, and an image embed."""
    content = (
        "---\n"
        "tags:\n"
        "  - python\n"
        "  - memorymesh\n"
        "aliases: [MemoryMesh Note, MM Note]\n"
        "created: 2024-01-01\n"
        "modified: 2024-06-15\n"
        "---\n"
        "\n"
        "# My Note\n"
        "\n"
        "This note links to [[ProjectA]] and [[ProjectB|Alias B]].\n"
        "It also references [[Architecture|arch doc]] for details.\n"
        "Here is an embedded image: ![[diagram.png]]\n"
    )
    p = tmp_path / "my_note.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def obsidian_note_no_frontmatter(tmp_path: Path) -> Path:
    """An Obsidian note without any YAML frontmatter."""
    content = "# Plain Note\n\nJust a plain note with [[SomeLink]].\n"
    p = tmp_path / "plain.md"
    p.write_text(content, encoding="utf-8")
    return p


class TestObsidianParser:
    def test_frontmatter_tags_parsed(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        assert "python" in doc.metadata["tags"]
        assert "memorymesh" in doc.metadata["tags"]

    def test_frontmatter_aliases_parsed(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        aliases = doc.metadata["aliases"]
        assert isinstance(aliases, list)
        assert len(aliases) >= 1

    def test_frontmatter_dates_parsed(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        assert doc.metadata["created"] == "2024-01-01"
        assert doc.metadata["modified"] == "2024-06-15"

    def test_backlinks_extracted(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        backlinks = doc.metadata["backlinks"]
        assert "ProjectA" in backlinks
        assert "ProjectB" in backlinks
        assert "Architecture" in backlinks

    def test_image_embed_ignored(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        backlinks = doc.metadata["backlinks"]
        assert "diagram.png" not in backlinks

    def test_no_frontmatter_does_not_crash(self, obsidian_note_no_frontmatter: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note_no_frontmatter)
        assert doc.text != "" or doc.metadata.get("error") is None
        assert isinstance(doc.metadata.get("backlinks"), list)
        assert "SomeLink" in doc.metadata["backlinks"]

    def test_file_type_is_md(self, obsidian_note: Path) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        doc = ObsidianParser().parse(obsidian_note)
        assert doc.file_type == ".md"

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser

        exts = ObsidianParser().supported_extensions
        assert ".md" in exts


# B2 - NotionParser


@pytest.fixture()
def notion_html_file(tmp_path: Path) -> Path:
    """A minimal Notion HTML export with title and two database properties."""
    html = (
        "<html><head><title>My Page a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4</title></head>"
        "<body>"
        "<h1>My Page</h1>"
        "<p>This is the page content about MemoryMesh.</p>"
        '<div data-type="select"><span>Tag1</span></div>'
        '<div data-type="date"><span>2024-01-01</span></div>'
        "</body></html>"
    )
    # Notion export filename format: "Page Title <uuid>.html"
    p = tmp_path / "My Page a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4.html"
    p.write_text(html, encoding="utf-8")
    return p


@pytest.fixture()
def notion_html_malformed(tmp_path: Path) -> Path:
    """A malformed HTML file that should not crash the parser."""
    p = tmp_path / "bad.html"
    p.write_text("<html><body><h1>Broken", encoding="utf-8")
    return p


class TestNotionParser:
    def test_title_extracted(self, notion_html_file: Path) -> None:
        from memorymesh.parsing.notion_parser import NotionParser

        doc = NotionParser().parse(notion_html_file)
        assert doc.metadata["page_title"] == "My Page"

    def test_notion_id_extracted(self, notion_html_file: Path) -> None:
        from memorymesh.parsing.notion_parser import NotionParser

        doc = NotionParser().parse(notion_html_file)
        assert doc.metadata["notion_id"] is not None
        assert len(str(doc.metadata["notion_id"])) > 0

    def test_database_name_is_parent_dir(self, notion_html_file: Path) -> None:
        from memorymesh.parsing.notion_parser import NotionParser

        doc = NotionParser().parse(notion_html_file)
        assert doc.metadata["database_name"] == notion_html_file.parent.name

    def test_malformed_html_does_not_crash(self, notion_html_malformed: Path) -> None:
        from memorymesh.parsing.notion_parser import NotionParser

        doc = NotionParser().parse(notion_html_malformed)
        # Should not raise; may return empty text
        assert isinstance(doc.text, str)

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.notion_parser import NotionParser

        exts = NotionParser().supported_extensions
        assert ".html" in exts


# B3 - ConversationParser


@pytest.fixture()
def claude_export(tmp_path: Path) -> Path:
    """A minimal Claude.ai JSON conversation export."""
    data = [
        {
            "uuid": "sess-001",
            "name": "Test Conversation",
            "chat_messages": [
                {"role": "human", "content": "Hello Claude!", "created_at": "2024-01-01T10:00:00"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                    "created_at": "2024-01-01T10:00:05",
                },
                {
                    "role": "human",
                    "content": "Tell me about MemoryMesh.",
                    "created_at": "2024-01-01T10:00:10",
                },
            ],
        }
    ]
    p = tmp_path / "claude_export.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def chatgpt_export(tmp_path: Path) -> Path:
    """A minimal ChatGPT JSON conversation export."""
    data = {
        "title": "ChatGPT Chat",
        "id": "chat-abc123",
        "mapping": {
            "node1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["What is RAG?"]},
                    "create_time": 1704067200.0,
                }
            },
            "node2": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["RAG stands for Retrieval-Augmented Generation."]},
                    "create_time": 1704067205.0,
                }
            },
        },
    }
    p = tmp_path / "chatgpt_export.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def invalid_json_file(tmp_path: Path) -> Path:
    """A file with invalid JSON content."""
    p = tmp_path / "invalid.json"
    p.write_text("this is not json { broken", encoding="utf-8")
    return p


class TestConversationParser:
    def test_claude_turns_present_in_text(self, claude_export: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(claude_export)
        assert "Hello Claude" in doc.text
        assert "How can I help" in doc.text

    def test_claude_metadata_source_app(self, claude_export: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(claude_export)
        assert doc.metadata["source_app"] == "claude"

    def test_claude_session_id(self, claude_export: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(claude_export)
        assert doc.metadata["session_id"] == "sess-001"

    def test_chatgpt_turns_present(self, chatgpt_export: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(chatgpt_export)
        assert "RAG" in doc.text

    def test_chatgpt_metadata_source_app(self, chatgpt_export: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(chatgpt_export)
        assert doc.metadata["source_app"] == "chatgpt"

    def test_invalid_json_returns_error_doc(self, invalid_json_file: Path) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        doc = ConversationParser().parse(invalid_json_file)
        assert doc.text == ""
        assert "error" in doc.metadata

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.conversation_parser import ConversationParser

        exts = ConversationParser().supported_extensions
        assert ".json" in exts


# B4 - EmailParser


@pytest.fixture()
def mbox_file(tmp_path: Path) -> Path:
    """An mbox with 3 messages: plain text, HTML, and multipart with attachment."""
    p = tmp_path / "test.mbox"
    mbox = mailbox.mbox(str(p))

    # Message 1: plain text
    msg1 = mailbox.mboxMessage()
    msg1["From"] = "alice@example.com"
    msg1["To"] = "bob@example.com"
    msg1["Subject"] = "Hello there"
    msg1["Date"] = "Mon, 01 Jan 2024 10:00:00 +0000"
    msg1["Message-ID"] = "<1@example.com>"
    msg1.set_payload("This is a plain text email about MemoryMesh.")
    msg1.set_type("text/plain")
    mbox.add(msg1)

    # Message 2: HTML only
    msg2 = mailbox.mboxMessage()
    msg2["From"] = "carol@example.com"
    msg2["To"] = "alice@example.com"
    msg2["Subject"] = "HTML Email"
    msg2["Date"] = "Tue, 02 Jan 2024 10:00:00 +0000"
    msg2["Message-ID"] = "<2@example.com>"
    msg2.set_payload("<html><body><p>HTML content here.</p></body></html>")
    msg2.set_type("text/html")
    mbox.add(msg2)

    # Message 3: multipart with a binary attachment
    import email.mime.base
    import email.mime.multipart
    import email.mime.text

    msg3 = email.mime.multipart.MIMEMultipart()
    msg3["From"] = "dave@example.com"
    msg3["To"] = "alice@example.com"
    msg3["Subject"] = "With Attachment"
    msg3["Date"] = "Wed, 03 Jan 2024 10:00:00 +0000"
    msg3["Message-ID"] = "<3@example.com>"

    text_part = email.mime.text.MIMEText("See the attached file.", "plain")
    msg3.attach(text_part)

    binary_part = email.mime.base.MIMEBase("application", "octet-stream")
    binary_part.set_payload(b"\x00\x01\x02")
    binary_part["Content-Disposition"] = 'attachment; filename="file.bin"'
    msg3.attach(binary_part)

    mbox.add(mailbox.mboxMessage(msg3))
    mbox.close()
    return p


class TestEmailParser:
    def test_plain_text_extracted(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        docs = EmailParser().parse_all(mbox_file)
        texts = [d.text for d in docs]
        assert any("plain text email" in t for t in texts)

    def test_html_stripped_to_text(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        docs = EmailParser().parse_all(mbox_file)
        texts = [d.text for d in docs]
        assert any("HTML content here" in t for t in texts)
        # No HTML tags should survive
        assert all("<html>" not in t for t in texts)

    def test_attachment_skipped(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        docs = EmailParser().parse_all(mbox_file)
        # The multipart message should still produce a doc (from the text/plain part)
        # and the attachment is skipped (had_attachment flag).
        attach_docs = [d for d in docs if d.metadata.get("had_attachment")]
        assert len(attach_docs) == 1

    def test_max_messages_respected(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        docs = EmailParser(max_messages=1).parse_all(mbox_file)
        assert len(docs) <= 1

    def test_metadata_fields_present(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        docs = EmailParser().parse_all(mbox_file)
        assert len(docs) >= 1
        first = docs[0].metadata
        assert "from_addr" in first
        assert "subject" in first
        assert "date" in first
        assert "message_id" in first

    def test_parse_returns_combined_doc(self, mbox_file: Path) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        doc = EmailParser().parse(mbox_file)
        assert isinstance(doc.text, str)
        assert doc.metadata.get("message_count", 0) > 0

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.email_parser import EmailParser

        exts = EmailParser().supported_extensions
        assert ".mbox" in exts


# B5 - CalendarParser


@pytest.fixture()
def ics_file(tmp_path: Path) -> Path:
    """An iCal file with 2 VEVENTs and 1 VTODO."""
    content = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "X-WR-CALNAME:Work Calendar\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:event-001@example.com\r\n"
        "SUMMARY:Team Standup\r\n"
        "DESCRIPTION:Daily sync meeting\r\n"
        "LOCATION:Room 101\r\n"
        "DTSTART:20240101T090000Z\r\n"
        "DTEND:20240101T093000Z\r\n"
        "ORGANIZER:mailto:boss@example.com\r\n"
        "END:VEVENT\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:event-002@example.com\r\n"
        "SUMMARY:Sprint Planning\r\n"
        "DESCRIPTION:Quarterly planning session\r\n"
        "DTSTART:20240102T100000Z\r\n"
        "DTEND:20240102T120000Z\r\n"
        "END:VEVENT\r\n"
        "BEGIN:VTODO\r\n"
        "UID:todo-001@example.com\r\n"
        "SUMMARY:Write tests\r\n"
        "END:VTODO\r\n"
        "END:VCALENDAR\r\n"
    )
    p = tmp_path / "calendar.ics"
    p.write_bytes(content.encode("utf-8"))
    return p


class TestCalendarParser:
    def test_vevents_parsed(self, ics_file: Path) -> None:
        pytest.importorskip("icalendar")
        from memorymesh.parsing.calendar_parser import CalendarParser

        docs = CalendarParser().parse_all(ics_file)
        assert len(docs) == 2

    def test_vtodo_ignored(self, ics_file: Path) -> None:
        pytest.importorskip("icalendar")
        from memorymesh.parsing.calendar_parser import CalendarParser

        docs = CalendarParser().parse_all(ics_file)
        summaries = [str(d.metadata.get("summary", "")) for d in docs]
        assert "Write tests" not in summaries

    def test_metadata_fields_present(self, ics_file: Path) -> None:
        pytest.importorskip("icalendar")
        from memorymesh.parsing.calendar_parser import CalendarParser

        docs = CalendarParser().parse_all(ics_file)
        assert len(docs) >= 1
        first = docs[0].metadata
        assert "dtstart" in first
        assert "dtend" in first
        assert "uid" in first

    def test_calendar_name_extracted(self, ics_file: Path) -> None:
        pytest.importorskip("icalendar")
        from memorymesh.parsing.calendar_parser import CalendarParser

        docs = CalendarParser().parse_all(ics_file)
        assert all(d.metadata.get("calendar_name") == "Work Calendar" for d in docs)

    def test_parse_returns_combined_doc(self, ics_file: Path) -> None:
        pytest.importorskip("icalendar")
        from memorymesh.parsing.calendar_parser import CalendarParser

        doc = CalendarParser().parse(ics_file)
        assert "Team Standup" in doc.text or doc.metadata.get("event_count", 0) > 0

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.calendar_parser import CalendarParser

        exts = CalendarParser().supported_extensions
        assert ".ics" in exts


# B6 - BrowserHistoryParser


@pytest.fixture()
def chrome_history_db(tmp_path: Path) -> Path:
    """A minimal Chrome-compatible history SQLite database."""
    p = tmp_path / "History"
    con = sqlite3.connect(str(p))
    con.execute(
        "CREATE TABLE urls ("
        "id INTEGER PRIMARY KEY, url TEXT, title TEXT, "
        "visit_count INTEGER, last_visit_time INTEGER"
        ")"
    )
    # Chrome epoch: microseconds since 1601-01-01
    # Use a round number that converts cleanly
    con.execute(
        "INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES "
        "('https://example.com', 'Example', 5, 13280000000000000)"
    )
    con.execute(
        "INSERT INTO urls (url, title, visit_count, last_visit_time) VALUES "
        "('https://python.org', 'Python', 1, 13280000000000001)"
    )
    con.commit()
    con.close()
    return p


@pytest.fixture()
def firefox_history_db(tmp_path: Path) -> Path:
    """A minimal Firefox-compatible places.sqlite database."""
    p = tmp_path / "places.sqlite"
    con = sqlite3.connect(str(p))
    con.execute(
        "CREATE TABLE moz_places ("
        "id INTEGER PRIMARY KEY, url TEXT, title TEXT, "
        "visit_count INTEGER, last_visit_date INTEGER"
        ")"
    )
    con.execute(
        "INSERT INTO moz_places (url, title, visit_count, last_visit_date) VALUES "
        "('https://mozilla.org', 'Mozilla', 3, 1704067200000000)"
    )
    con.execute(
        "INSERT INTO moz_places (url, title, visit_count, last_visit_date) VALUES "
        "('https://rarely.visited', 'Rare', 1, 1704067200000001)"
    )
    con.commit()
    con.close()
    return p


class TestBrowserHistoryParser:
    def test_chrome_urls_extracted(self, chrome_history_db: Path) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        # Patch shutil.copy2 to copy the fixture in-place (avoid live-DB safety path)
        with mock.patch(
            "shutil.copy2",
            side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
        ):
            docs = BrowserHistoryParser().parse_all(chrome_history_db)

        urls = [d.metadata["url"] for d in docs]
        assert "https://example.com" in urls

    def test_chrome_min_visit_count_filter(self, chrome_history_db: Path) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        with mock.patch(
            "shutil.copy2",
            side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
        ):
            docs = BrowserHistoryParser().parse_all(chrome_history_db)

        urls = [d.metadata["url"] for d in docs]
        # python.org has visit_count=1, below threshold of 2
        assert "https://python.org" not in urls

    def test_firefox_urls_extracted(self, firefox_history_db: Path) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        with mock.patch(
            "shutil.copy2",
            side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
        ):
            docs = BrowserHistoryParser().parse_all(firefox_history_db)

        urls = [d.metadata["url"] for d in docs]
        assert "https://mozilla.org" in urls

    def test_firefox_min_visit_count_filter(self, firefox_history_db: Path) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        with mock.patch(
            "shutil.copy2",
            side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
        ):
            docs = BrowserHistoryParser().parse_all(firefox_history_db)

        urls = [d.metadata["url"] for d in docs]
        assert "https://rarely.visited" not in urls

    def test_metadata_fields_present(self, chrome_history_db: Path) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        with mock.patch(
            "shutil.copy2",
            side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
        ):
            docs = BrowserHistoryParser().parse_all(chrome_history_db)

        assert len(docs) >= 1
        meta = docs[0].metadata
        assert "url" in meta
        assert "visit_count" in meta
        assert "last_visit" in meta
        assert "browser" in meta

    def test_tmp_copy_cleaned_up(self, chrome_history_db: Path) -> None:
        """The temp DB copy must be deleted after parsing."""
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        deleted: list[Path] = []

        def mock_unlink(self: Path, *, missing_ok: bool = False) -> None:
            deleted.append(self)

        with (
            mock.patch(
                "shutil.copy2",
                side_effect=lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()),
            ),
            mock.patch.object(Path, "unlink", mock_unlink),
        ):
            BrowserHistoryParser().parse_all(chrome_history_db)

        # At least one unlink call should have been made for the tmp copy
        assert len(deleted) >= 1

    def test_supported_extensions(self) -> None:
        from memorymesh.parsing.browser_history_parser import BrowserHistoryParser

        exts = BrowserHistoryParser().supported_extensions
        assert ".db" in exts


# Registry: new parsers are discoverable


class TestUpdatedRegistry:
    def test_ics_extension_registered(self) -> None:
        from memorymesh.parsing.registry import get_parser

        parser = get_parser(".ics")
        assert parser is not None

    def test_mbox_extension_registered(self) -> None:
        from memorymesh.parsing.registry import get_parser

        parser = get_parser(".mbox")
        assert parser is not None

    def test_obsidian_source_type_overrides_md(self) -> None:
        from memorymesh.parsing.obsidian_parser import ObsidianParser
        from memorymesh.parsing.registry import get_parser

        parser = get_parser(".md", source_type="obsidian")
        assert isinstance(parser, ObsidianParser)

    def test_notion_source_type_overrides_html(self) -> None:
        from memorymesh.parsing.notion_parser import NotionParser
        from memorymesh.parsing.registry import get_parser

        parser = get_parser(".html", source_type="notion")
        assert isinstance(parser, NotionParser)
