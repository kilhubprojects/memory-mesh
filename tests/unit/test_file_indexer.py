"""Unit tests for FileIndexer with mocked Chroma and EmbeddingProvider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memorymesh.core.models import (
    MemoryMeshConfig,
    StorageConfig,
)
from memorymesh.indexer.file_indexer import FileIndexer, _should_ignore, compute_hash
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.metadata_store import MetadataStore


def _make_indexer(tmp_path: Path) -> FileIndexer:
    """Build a FileIndexer with a real SQLite + BM25 but mocked vector store and provider."""
    config = MemoryMeshConfig(
        storage=StorageConfig(base_dir=tmp_path / "state"),
    )
    metadata_store = MetadataStore(tmp_path / "metadata.sqlite3")
    bm25 = BM25Index(tmp_path / "bm25.pkl")

    vector_store = MagicMock()
    vector_store.delete_by_path.return_value = None
    vector_store.upsert.return_value = None

    provider = MagicMock()
    provider.model_id = "mock-model"
    provider.embed_documents.side_effect = lambda texts: [[0.1] * 8 for _ in texts]

    return FileIndexer(
        config=config,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
    )


class TestComputeHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello", encoding="utf-8")
        assert compute_hash(f) == compute_hash(f)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("aaa", encoding="utf-8")
        b.write_text("bbb", encoding="utf-8")
        assert compute_hash(a) != compute_hash(b)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            compute_hash(tmp_path / "ghost.txt")


class TestShouldIgnore:
    def test_git_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / ".git" / "config"
        assert _should_ignore(path, tmp_path, ["**/.git/**"])

    def test_normal_file_not_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "file.py"
        assert not _should_ignore(path, tmp_path, ["**/.git/**"])

    def test_node_modules_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "node_modules" / "pkg" / "index.js"
        assert _should_ignore(path, tmp_path, ["**/node_modules/**"])


class TestIndexFile:
    def test_indexes_text_file(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        result = indexer.index_file(sample_text_file, source_name="test")
        assert result.status == "indexed"
        assert result.n_chunks >= 1

    def test_skips_unchanged_file(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        indexer.index_file(sample_text_file, source_name="test")
        result2 = indexer.index_file(sample_text_file, source_name="test")
        assert result2.status == "skipped"

    def test_force_reindexes(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        indexer.index_file(sample_text_file, source_name="test")
        result2 = indexer.index_file(sample_text_file, source_name="test", force=True)
        assert result2.status == "indexed"

    def test_unsupported_extension_returns_unsupported(self, tmp_path: Path) -> None:
        indexer = _make_indexer(tmp_path)
        exe = tmp_path / "binary.exe"
        exe.write_bytes(b"\x00\x01\x02")
        result = indexer.index_file(exe)
        assert result.status == "unsupported"

    def test_missing_file_returns_parse_error(self, tmp_path: Path) -> None:
        indexer = _make_indexer(tmp_path)
        result = indexer.index_file(tmp_path / "ghost.txt")
        assert result.status == "parse_error"

    def test_embed_failure_returns_parse_error(
        self, tmp_path: Path, sample_text_file: Path
    ) -> None:
        from memorymesh.core.exceptions import EmbeddingError

        indexer = _make_indexer(tmp_path)
        indexer._provider.embed_documents.side_effect = EmbeddingError("model broken")
        result = indexer.index_file(sample_text_file, source_name="test")
        assert result.status == "parse_error"
        assert "model broken" in (result.error or "")

    def test_error_record_written_to_sqlite(self, tmp_path: Path, sample_text_file: Path) -> None:
        from memorymesh.core.exceptions import EmbeddingError

        indexer = _make_indexer(tmp_path)
        indexer._provider.embed_documents.side_effect = EmbeddingError("boom")
        indexer.index_file(sample_text_file, source_name="test")
        record = indexer._metadata_store.get_file(str(sample_text_file))
        assert record is not None
        assert record.status == "parse_error"

    def test_duration_ms_positive(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        result = indexer.index_file(sample_text_file)
        assert result.duration_ms >= 0.0

    def test_chroma_failure_writes_failed_status_to_sqlite(
        self, tmp_path: Path, sample_text_file: Path
    ) -> None:
        """When Chroma upsert raises, SQLite should record status='failed'."""
        indexer = _make_indexer(tmp_path)
        indexer._vector_store.upsert.side_effect = RuntimeError("Chroma down")
        indexer.index_file(sample_text_file, source_name="test")
        record = indexer._metadata_store.get_file(str(sample_text_file))
        assert record is not None
        assert record.status == "failed"

    def test_bm25_failure_writes_failed_status_to_sqlite(
        self, tmp_path: Path, sample_text_file: Path
    ) -> None:
        """When BM25 persistence raises, SQLite records status='failed'."""
        indexer = _make_indexer(tmp_path)
        bm25_mock = MagicMock()
        bm25_mock.add_chunks.side_effect = RuntimeError("BM25 down")
        indexer._bm25 = bm25_mock
        indexer.index_file(sample_text_file, source_name="test")
        record = indexer._metadata_store.get_file(str(sample_text_file))
        assert record is not None
        assert record.status == "failed"

    def test_sqlite_intent_written_before_chroma(
        self, tmp_path: Path, sample_text_file: Path
    ) -> None:
        """SQLite 'indexing' status is written before the Chroma upsert call."""
        status_at_chroma_call: list[str] = []
        indexer = _make_indexer(tmp_path)

        def _capture_status_then_upsert(chunks: list) -> None:
            rec = indexer._metadata_store.get_file(str(sample_text_file))
            if rec is not None:
                status_at_chroma_call.append(rec.status)

        indexer._vector_store.upsert.side_effect = _capture_status_then_upsert
        indexer.index_file(sample_text_file, source_name="test")
        # At the time Chroma.upsert was called, SQLite should already have "indexing"
        assert status_at_chroma_call and status_at_chroma_call[0] == "indexing"


class TestIndexDirectory:
    def test_indexes_all_files_in_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("content a", encoding="utf-8")
        (src / "b.txt").write_text("content b", encoding="utf-8")

        indexer = _make_indexer(tmp_path / "state")
        results = indexer.index_directory(src, source_name="test")
        indexed = [r for r in results if r.status == "indexed"]
        assert len(indexed) == 2

    def test_respects_ignore_patterns(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "keep.txt").write_text("keep", encoding="utf-8")
        skip_dir = src / "skip_me"
        skip_dir.mkdir()
        (skip_dir / "hidden.txt").write_text("hidden", encoding="utf-8")

        indexer = _make_indexer(tmp_path / "state")
        results = indexer.index_directory(src, ignore_patterns=["skip_me/**"])
        paths = [r.path for r in results]
        assert not any("hidden" in str(p) for p in paths)

    def test_non_recursive_skips_subdirs(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "top.txt").write_text("top", encoding="utf-8")
        sub = src / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep", encoding="utf-8")

        indexer = _make_indexer(tmp_path / "state")
        results = indexer.index_directory(src, recursive=False)
        paths = [str(r.path) for r in results]
        assert not any("deep" in p for p in paths)


class TestDeleteFile:
    def test_delete_marks_deleted_in_sqlite(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        indexer.index_file(sample_text_file, source_name="test")
        indexer.delete_file(sample_text_file)
        record = indexer._metadata_store.get_file(str(sample_text_file))
        assert record is not None
        assert record.status == "deleted"

    def test_delete_calls_vector_store(self, tmp_path: Path, sample_text_file: Path) -> None:
        indexer = _make_indexer(tmp_path)
        indexer.delete_file(sample_text_file)
        indexer._vector_store.delete_by_path.assert_called_once_with(str(sample_text_file))
