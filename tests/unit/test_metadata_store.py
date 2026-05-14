"""Unit tests for MetadataStore (SQLite)."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from memorymesh.core.models import FileRecord
from memorymesh.storage.metadata_store import MetadataStore


@pytest.fixture()
def store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "test.sqlite3")
    s.init_schema()
    return s


def _record(**kwargs: object) -> FileRecord:
    defaults: dict[str, object] = {
        "path": "/tmp/file.txt",
        "source_name": "notes",
        "sha256": "abc123",
        "mtime": time.time(),
        "size_bytes": 1024,
        "file_type": ".txt",
        "n_chunks": 3,
        "status": "indexed",
        "error_message": None,
        "indexed_at": time.time(),
        "embedding_model_id": "all-MiniLM-L6-v2",
    }
    defaults.update(kwargs)
    return FileRecord(**defaults)  # type: ignore[arg-type]


class TestInitSchema:
    def test_creates_tables(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "fresh.sqlite3")
        store.init_schema()
        rows = (
            store._connection()
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        )
        names = {r["name"] for r in rows}
        assert {"files", "sources", "index_state"} <= names

    def test_idempotent(self, store: MetadataStore) -> None:
        store.init_schema()  # second call must not raise


class TestIndexState:
    def test_fresh_install_is_clean(self, store: MetadataStore) -> None:
        assert store.is_clean_state() is True

    def test_mark_startup_sets_dirty(self, store: MetadataStore) -> None:
        store.mark_startup()
        assert store.is_clean_state() is False

    def test_mark_clean_shutdown_after_startup(self, store: MetadataStore) -> None:
        store.mark_startup()
        store.mark_clean_shutdown()
        assert store.is_clean_state() is True

    def test_epoch_increments(self, store: MetadataStore) -> None:
        store.mark_startup()
        epoch1 = store.get_state("epoch")
        store.mark_startup()
        epoch2 = store.get_state("epoch")
        assert int(epoch2) == int(epoch1) + 1  # type: ignore[arg-type]

    def test_get_set_state(self, store: MetadataStore) -> None:
        assert store.get_state("x") is None
        store.set_state("x", "hello")
        assert store.get_state("x") == "hello"
        store.set_state("x", "world")
        assert store.get_state("x") == "world"


class TestUpsertAndGet:
    def test_upsert_and_get(self, store: MetadataStore) -> None:
        r = _record(path="/tmp/a.py")
        store.upsert_file(r)
        fetched = store.get_file("/tmp/a.py")
        assert fetched is not None
        assert fetched.sha256 == r.sha256
        assert fetched.status == "indexed"

    def test_get_missing_returns_none(self, store: MetadataStore) -> None:
        assert store.get_file("/nonexistent") is None

    def test_upsert_updates_existing(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/b.py", sha256="old"))
        store.upsert_file(_record(path="/tmp/b.py", sha256="new"))
        assert store.get_file("/tmp/b.py").sha256 == "new"  # type: ignore[union-attr]


class TestDeleteAndMarkDeleted:
    def test_delete_file_removes_row(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/c.py"))
        store.delete_file("/tmp/c.py")
        assert store.get_file("/tmp/c.py") is None

    def test_mark_deleted_keeps_row(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/d.py"))
        store.mark_deleted("/tmp/d.py")
        rec = store.get_file("/tmp/d.py")
        assert rec is not None
        assert rec.status == "deleted"

    def test_mark_pending_reindex(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/e.py"))
        store.mark_pending_reindex("/tmp/e.py")
        assert store.get_file("/tmp/e.py").status == "pending_reindex"  # type: ignore[union-attr]


class TestListFiles:
    def test_list_all(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/1.txt"))
        store.upsert_file(_record(path="/tmp/2.txt"))
        assert len(store.list_files()) == 2

    def test_filter_by_source(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/x.py", source_name="code"))
        store.upsert_file(_record(path="/tmp/y.txt", source_name="docs"))
        assert len(store.list_files(source_name="code")) == 1

    def test_filter_by_status(self, store: MetadataStore) -> None:
        store.upsert_file(_record(path="/tmp/ok.py", status="indexed"))
        store.upsert_file(_record(path="/tmp/bad.py", status="parse_error"))
        result = store.list_files(status="parse_error")
        assert len(result) == 1
        assert result[0].path == "/tmp/bad.py"


class TestGetStats:
    def test_stats_counts_by_status(self, store: MetadataStore) -> None:
        for i in range(3):
            store.upsert_file(_record(path=f"/tmp/{i}.py", status="indexed"))
        store.upsert_file(_record(path="/tmp/bad.py", status="parse_error"))
        stats = store.get_stats()
        assert stats["indexed"] == 3
        assert stats["parse_error"] == 1


class TestSources:
    def test_upsert_and_list_sources(self, store: MetadataStore) -> None:
        store.upsert_source("docs", "/home/user/docs", recursive=True)
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["name"] == "docs"

    def test_update_scan_time(self, store: MetadataStore) -> None:
        store.upsert_source("code", "/home/user/code", recursive=False)
        before = time.time()
        store.update_source_scan_time("code")
        src = store.list_sources()[0]
        assert float(src["last_full_scan_at"]) >= before


@pytest.mark.skipif(sys.platform == "win32", reason="Unix permissions only")
def test_state_dir_permissions(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "db" / "meta.sqlite3")
    store.init_schema()
    mode = oct(stat.S_IMODE(os.stat(tmp_path / "db").st_mode))
    assert mode == oct(0o700)


class TestSchemaVersion:
    """Verify that init_schema() correctly stamps PRAGMA user_version."""

    def test_fresh_db_gets_current_schema_version(self, tmp_path: Path) -> None:
        import sqlite3

        from memorymesh.storage.metadata_store import SCHEMA_VERSION

        store = MetadataStore(tmp_path / "fresh.sqlite3")
        store.init_schema()

        conn = sqlite3.connect(str(tmp_path / "fresh.sqlite3"))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()

        assert version == SCHEMA_VERSION

    def test_idempotent_init_schema(self, tmp_path: Path) -> None:
        """Calling init_schema() twice does not change user_version."""
        import sqlite3

        from memorymesh.storage.metadata_store import SCHEMA_VERSION

        store = MetadataStore(tmp_path / "idem.sqlite3")
        store.init_schema()
        store.init_schema()

        conn = sqlite3.connect(str(tmp_path / "idem.sqlite3"))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()

        assert version == SCHEMA_VERSION

    def test_old_db_migrated_to_current_version(self, tmp_path: Path) -> None:
        """A DB at version 1 is migrated to SCHEMA_VERSION."""
        import sqlite3

        from memorymesh.storage.metadata_store import SCHEMA_VERSION

        db_path = tmp_path / "old.sqlite3"

        # Simulate a version-1 database (initial schema, no Wave 3 tables)
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                size_bytes INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                n_chunks INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error_message TEXT,
                indexed_at REAL NOT NULL,
                embedding_model_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS sources (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                recursive INTEGER NOT NULL DEFAULT 1,
                last_full_scan_at REAL
            );
            CREATE TABLE IF NOT EXISTS index_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        store = MetadataStore(db_path)
        store.init_schema()

        conn = sqlite3.connect(str(db_path))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        # forgotten_chunks should now exist (added in v3)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()

        assert version == SCHEMA_VERSION
        assert "forgotten_chunks" in tables
        assert "chunk_tiers" in tables

    def test_schema_version_constant_is_int(self) -> None:
        from memorymesh.storage.metadata_store import SCHEMA_VERSION

        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION >= 3
