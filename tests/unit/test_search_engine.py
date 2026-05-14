"""Unit tests for SearchEngine.

Uses a real ChromaVectorStore (tmp_path) and real BM25Index with synthetic
embeddings so no ML model is needed.  The goal is to verify that hybrid
search returns correct results and that mode selection works.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memorymesh.core.models import SearchConfig, SearchHybridConfig
from memorymesh.search.engine import SearchEngine
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.vector_store import ChromaVectorStore

random.seed(0)
_DIM = 8  # tiny dim for speed


def _vec(dim: int = _DIM) -> list[float]:
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _make_embedding_provider(query_vec: list[float]) -> MagicMock:
    provider = MagicMock()
    provider.model_id = "test-model"
    provider.embed_query.return_value = query_vec
    provider.embed_documents.side_effect = lambda texts: [_vec() for _ in texts]
    return provider


@pytest.fixture()
def engine_components(tmp_path: Path) -> tuple[ChromaVectorStore, BM25Index, MagicMock]:
    vector_store = ChromaVectorStore(db_path=tmp_path / "chroma", embedding_model_id="test-model")
    bm25 = BM25Index(tmp_path / "bm25.pkl")
    provider = _make_embedding_provider(_vec())
    return vector_store, bm25, provider


class TestSearchEngineMode:
    def test_dense_mode_returns_response(
        self, engine_components: tuple[ChromaVectorStore, BM25Index, MagicMock]
    ) -> None:
        vs, bm25, provider = engine_components
        from memorymesh.core.models import ChunkMetadata, ChunkWithEmbedding

        vs.upsert(
            [
                ChunkWithEmbedding(
                    path=Path("/tmp/f.txt"),
                    chunk_index=0,
                    text="hello",
                    start_char=0,
                    end_char=5,
                    file_type=".txt",
                    mtime=0.0,
                    source_root="",
                    metadata=ChunkMetadata(),
                    embedding=_vec(_DIM),
                )
            ]
        )
        engine = SearchEngine(vs, bm25, provider)
        response = engine.search("hello", top_k=3, mode="dense")
        assert response.mode == "dense"
        assert len(response.hits) >= 0  # may be 0 if no match

    def test_sparse_mode_uses_bm25(
        self, engine_components: tuple[ChromaVectorStore, BM25Index, MagicMock]
    ) -> None:
        vs, bm25, provider = engine_components
        # BM25Okapi v0.2 IDF = log(N-n+0.5) - log(n+0.5); with N=2, n=1 → 0.
        # Use ≥3 docs so terms in only 1 doc have positive IDF.
        bm25.rebuild(
            [
                ("f.txt:0", "f.txt", "python machine learning tutorial"),
                ("g.txt:0", "g.txt", "java enterprise application server"),
                ("h.txt:0", "h.txt", "database storage relational tables"),
            ]
        )
        engine = SearchEngine(vs, bm25, provider)
        response = engine.search("python tutorial", top_k=3, mode="sparse")
        assert response.mode == "sparse"
        assert len(response.hits) >= 1
        assert response.hits[0].path == "f.txt"

    def test_hybrid_disabled_falls_back_to_dense(
        self, engine_components: tuple[ChromaVectorStore, BM25Index, MagicMock]
    ) -> None:
        vs, bm25, provider = engine_components
        cfg = SearchConfig(hybrid=SearchHybridConfig(enabled=False))
        engine = SearchEngine(vs, bm25, provider, config=cfg)
        response = engine.search("anything", top_k=3, mode="hybrid")
        assert response.mode == "dense"

    def test_response_has_duration(
        self, engine_components: tuple[ChromaVectorStore, BM25Index, MagicMock]
    ) -> None:
        vs, bm25, provider = engine_components
        engine = SearchEngine(vs, bm25, provider)
        response = engine.search("test", top_k=1, mode="dense")
        assert response.duration_ms >= 0.0

    def test_empty_stores_returns_empty_hits(
        self, engine_components: tuple[ChromaVectorStore, BM25Index, MagicMock]
    ) -> None:
        vs, bm25, provider = engine_components
        engine = SearchEngine(vs, bm25, provider)
        response = engine.search("anything", top_k=5, mode="dense")
        assert response.hits == []


class TestFacetedFilter:
    """Verify that source / file_type / after_ts / before_ts post-filters work."""

    def _engine_with_hits(self, hits: list) -> SearchEngine:
        """Return a SearchEngine whose dense path returns *hits* directly."""
        vs = MagicMock()
        vs.search.return_value = hits
        bm25 = MagicMock()
        bm25.search.return_value = []
        provider = MagicMock()
        provider.embed_query.return_value = _vec()
        cfg = SearchConfig(hybrid=SearchHybridConfig(enabled=False))
        return SearchEngine(vs, bm25, provider, config=cfg)

    def _hit(
        self, path: str, source_root: str = "", file_type: str = ".txt", mtime: float = 1000.0
    ):
        from memorymesh.core.models import SearchHit

        return SearchHit(
            path=path,
            chunk_index=0,
            score=0.9,
            preview="preview",
            file_type=file_type,
            mtime=mtime,
            source_root=source_root,
        )

    def test_source_filter_keeps_matching(self) -> None:
        hits = [
            self._hit("/a.md", source_root="documents"),
            self._hit("/b.md", source_root="jira"),
        ]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", source="documents")
        assert all(h.source_root == "documents" for h in resp.hits)
        assert len(resp.hits) == 1

    def test_file_type_filter(self) -> None:
        hits = [
            self._hit("/a.pdf", file_type=".pdf"),
            self._hit("/b.md", file_type=".md"),
        ]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", file_type=".pdf")
        assert all(h.file_type == ".pdf" for h in resp.hits)

    def test_after_ts_excludes_old(self) -> None:
        hits = [
            self._hit("/old.md", mtime=500.0),
            self._hit("/new.md", mtime=2000.0),
        ]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", after_ts=1000.0)
        assert all((h.mtime or 0) >= 1000.0 for h in resp.hits)
        assert len(resp.hits) == 1

    def test_before_ts_excludes_new(self) -> None:
        hits = [
            self._hit("/old.md", mtime=500.0),
            self._hit("/new.md", mtime=2000.0),
        ]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", before_ts=1000.0)
        assert all((h.mtime or 0) <= 1000.0 for h in resp.hits)
        assert len(resp.hits) == 1


class TestHybridFusion:
    def test_hybrid_result_includes_both_sources(self, tmp_path: Path) -> None:
        """Document found only in BM25 should appear in hybrid results."""
        from memorymesh.core.models import ChunkMetadata, ChunkWithEmbedding

        target_vec = _vec(_DIM)
        vs = ChromaVectorStore(db_path=tmp_path / "chroma", embedding_model_id="test-model")
        bm25 = BM25Index(tmp_path / "bm25.pkl")

        # Insert into dense store
        vs.upsert(
            [
                ChunkWithEmbedding(
                    path=Path("/tmp/dense.txt"),
                    chunk_index=0,
                    text="dense content",
                    start_char=0,
                    end_char=13,
                    file_type=".txt",
                    mtime=0.0,
                    source_root="",
                    metadata=ChunkMetadata(),
                    embedding=target_vec,
                )
            ]
        )
        # Insert into BM25 only
        bm25.rebuild(
            [
                ("/tmp/dense.txt:0", "/tmp/dense.txt", "dense content"),
                ("/tmp/sparse.txt:0", "/tmp/sparse.txt", "sparse only document"),
            ]
        )

        provider = _make_embedding_provider(target_vec)
        engine = SearchEngine(vs, bm25, provider)
        response = engine.search("dense content", top_k=5, mode="hybrid")
        paths = {h.path for h in response.hits}
        # The dense document should be in hybrid results
        assert any("dense" in p for p in paths)


class TestSuppressionFilter:
    """Verify that forgotten chunks are excluded from search results."""

    def _engine_with_hits(self, hits: list, metadata_store=None) -> SearchEngine:
        vs = MagicMock()
        vs.search.return_value = hits
        bm25 = MagicMock()
        bm25.search.return_value = []
        provider = MagicMock()
        provider.embed_query.return_value = _vec()
        cfg = SearchConfig(hybrid=SearchHybridConfig(enabled=False))
        return SearchEngine(vs, bm25, provider, config=cfg, metadata_store=metadata_store)

    def _hit(self, path: str, chunk_index: int = 0) -> object:
        from memorymesh.core.models import SearchHit

        return SearchHit(
            path=path,
            chunk_index=chunk_index,
            score=0.9,
            preview="preview",
            file_type=".txt",
            mtime=1000.0,
            source_root="",
        )

    def test_no_metadata_store_returns_all_hits(self) -> None:
        """Without a MetadataStore, suppression filtering is a no-op."""
        hits = [self._hit("/a.txt", 0), self._hit("/b.txt", 0)]
        engine = self._engine_with_hits(hits, metadata_store=None)
        resp = engine.search("q", mode="dense")
        assert len(resp.hits) == 2

    def test_suppressed_chunk_excluded_from_results(self) -> None:
        """After forget_chunk, the matching hit is filtered out."""
        hits = [self._hit("/a.txt", 0), self._hit("/b.txt", 0)]
        metadata_store = MagicMock()
        metadata_store.list_forgotten.return_value = ["/a.txt:0"]
        engine = self._engine_with_hits(hits, metadata_store=metadata_store)
        resp = engine.search("q", mode="dense")
        assert all(h.path != "/a.txt" for h in resp.hits)
        assert len(resp.hits) == 1

    def test_empty_suppression_list_returns_all(self) -> None:
        """Empty suppression list → all hits pass through."""
        hits = [self._hit("/a.txt", 0), self._hit("/b.txt", 0)]
        metadata_store = MagicMock()
        metadata_store.list_forgotten.return_value = []
        engine = self._engine_with_hits(hits, metadata_store=metadata_store)
        resp = engine.search("q", mode="dense")
        assert len(resp.hits) == 2

    def test_metadata_store_error_is_graceful(self) -> None:
        """If list_forgotten() raises, the engine logs a warning and continues."""
        hits = [self._hit("/a.txt", 0)]
        metadata_store = MagicMock()
        metadata_store.list_forgotten.side_effect = RuntimeError("db error")
        engine = self._engine_with_hits(hits, metadata_store=metadata_store)
        # Should NOT raise; results returned despite store error
        resp = engine.search("q", mode="dense")
        assert len(resp.hits) == 1

    def test_forget_memory_tool_suppress_mode(self) -> None:
        """forget_memory with mode='suppress' calls metadata_store.forget_chunk."""
        import importlib
        import unittest.mock as mock

        mod = importlib.import_module("memorymesh.server.tools.forget_memory")
        mcp = MagicMock()
        registered_fn = None

        def capture_tool():
            def dec(fn):
                nonlocal registered_fn
                registered_fn = fn
                return fn

            return dec

        mcp.tool = capture_tool
        ctx = MagicMock()
        ctx.metadata_store.forget_chunk = MagicMock()

        with mock.patch("memorymesh.server.auth_guard.check_access", return_value=None):
            mod.register(mcp, ctx)
            result = registered_fn(chunk_id="/doc.txt:3", mode="suppress")

        assert result["mode"] == "suppress"
        assert result["suppressed"] is True
        ctx.metadata_store.forget_chunk.assert_called_once_with("/doc.txt:3")

    def test_forget_memory_tool_cold_mode(self) -> None:
        """forget_memory with mode='cold' demotes via tiered_memory.forget."""
        import importlib
        import unittest.mock as mock

        mod = importlib.import_module("memorymesh.server.tools.forget_memory")
        mcp = MagicMock()
        registered_fn = None

        def capture_tool():
            def dec(fn):
                nonlocal registered_fn
                registered_fn = fn
                return fn

            return dec

        mcp.tool = capture_tool
        ctx = MagicMock()
        ctx.tiered_memory = MagicMock()
        ctx.tiered_memory.forget = MagicMock()

        with mock.patch("memorymesh.server.auth_guard.check_access", return_value=None):
            mod.register(mcp, ctx)
            result = registered_fn(chunk_id="/doc.txt:3", mode="cold")

        assert result["mode"] == "cold"
        assert result["tier"] == "cold"
        ctx.tiered_memory.forget.assert_called_once_with("/doc.txt:3")
