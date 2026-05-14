"""Unit tests for BM25Index."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorymesh.storage.bm25_index import BM25Index, _tokenize


class TestTokenize:
    def test_lowercases_text(self) -> None:
        tokens = _tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self) -> None:
        tokens = _tokenize("the quick brown fox")
        assert "the" not in tokens

    def test_removes_single_char_tokens(self) -> None:
        tokens = _tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens

    def test_returns_list_of_strings(self) -> None:
        tokens = _tokenize("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenize("") == []


class TestBM25IndexRebuild:
    def test_empty_index_count_zero(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        assert idx.count() == 0

    def test_rebuild_sets_count(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild(
            [
                ("path/a.txt:0", "path/a.txt", "hello world python"),
                ("path/b.txt:0", "path/b.txt", "memory search engine"),
            ]
        )
        assert idx.count() == 2

    def test_rebuild_overwrites_previous(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([("id1", "p1", "first doc")])
        idx.rebuild([("id2", "p2", "second doc"), ("id3", "p3", "third doc")])
        assert idx.count() == 2

    def test_rebuild_empty_list(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([])
        assert idx.count() == 0


class TestBM25IndexSearch:
    @pytest.fixture()
    def populated(self, tmp_path: Path) -> BM25Index:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild(
            [
                ("docs/python.txt:0", "docs/python.txt", "Python programming language tutorial"),
                ("docs/java.txt:0", "docs/java.txt", "Java programming object oriented"),
                ("docs/ml.txt:0", "docs/ml.txt", "machine learning neural networks deep"),
            ]
        )
        return idx

    def test_search_returns_hits(self, populated: BM25Index) -> None:
        hits = populated.search("python programming", top_k=3)
        assert len(hits) >= 1

    def test_exact_keyword_top_hit(self, populated: BM25Index) -> None:
        hits = populated.search("machine learning", top_k=3)
        assert hits[0].path == "docs/ml.txt"

    def test_scores_between_0_and_1(self, populated: BM25Index) -> None:
        hits = populated.search("programming", top_k=3)
        for h in hits:
            assert 0.0 <= h.score <= 1.0 + 1e-9

    def test_top_k_respected(self, populated: BM25Index) -> None:
        hits = populated.search("programming", top_k=1)
        assert len(hits) <= 1

    def test_empty_index_returns_empty(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        assert idx.search("anything", top_k=5) == []

    def test_query_no_tokens_returns_empty(self, populated: BM25Index) -> None:
        # stopwords-only query → no useful tokens
        assert populated.search("a the and", top_k=5) == []


class TestBM25IndexDeleteByPath:
    def test_delete_removes_entries(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild(
            [
                ("a.txt:0", "a.txt", "python tutorial"),
                ("b.txt:0", "b.txt", "java tutorial"),
            ]
        )
        removed = idx.delete_by_path("a.txt")
        assert removed == 1
        assert idx.count() == 1

    def test_delete_nonexistent_returns_zero(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([("a.txt:0", "a.txt", "content")])
        assert idx.delete_by_path("nope.txt") == 0


class TestBM25IndexPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        pkl = tmp_path / "bm25.pkl"
        idx = BM25Index(pkl)
        idx.rebuild([("id1", "f.txt", "save and load test")])

        idx2 = BM25Index(pkl)
        assert idx2.load() is True
        assert idx2.count() == 1

    def test_load_nonexistent_returns_false(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "nope.pkl")
        assert idx.load() is False

    def test_load_corrupt_file_returns_false(self, tmp_path: Path) -> None:
        pkl = tmp_path / "corrupt.pkl"
        pkl.write_bytes(b"not a pickle")
        idx = BM25Index(pkl)
        assert idx.load() is False
        assert idx.count() == 0

    def test_integrity_check_fails_on_tampered_pickle(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([("id1", "/f.txt", "hello world test")])
        # Tamper with the pickle after saving (sidecar SHA will no longer match).
        (tmp_path / "bm25.pkl").write_bytes(b"corrupted")
        idx2 = BM25Index(tmp_path / "bm25.pkl")
        result = idx2.load()
        assert result is False

    def test_sha256_sidecar_written_on_save(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([("id1", "/f.txt", "hello world")])
        assert (tmp_path / "bm25.pkl.sha256").exists()

    def test_valid_pickle_loads_after_save(self, tmp_path: Path) -> None:
        idx = BM25Index(tmp_path / "bm25.pkl")
        idx.rebuild([("id1", "/f.txt", "hello world test")])
        idx2 = BM25Index(tmp_path / "bm25.pkl")
        assert idx2.load() is True
        assert idx2.count() == 1
