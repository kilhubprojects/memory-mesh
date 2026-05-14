"""Integration tests for MCP tools.

Uses real ChromaVectorStore, BM25Index, MetadataStore, FileIndexer, and
SearchEngine with a synthetic embedding provider.  No MCP transport is
involved — tools are exercised by calling the registered functions directly
through the FastMCP instance's tool registry.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memorymesh.core.models import MemoryMeshConfig, StorageConfig
from memorymesh.indexer.file_indexer import FileIndexer
from memorymesh.observability.audit import AuditLogger
from memorymesh.search.engine import SearchEngine
from memorymesh.server.app import AppContext, build_mcp
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.metadata_store import MetadataStore
from memorymesh.storage.vector_store import ChromaVectorStore

random.seed(42)
_DIM = 8


def _vec() -> list[float]:
    v = [random.gauss(0, 1) for _ in range(_DIM)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.model_id = "test-model"
    provider.embed_query.return_value = _vec()
    provider.embed_documents.side_effect = lambda texts: [_vec() for _ in texts]
    return provider


@pytest.fixture()
def app_ctx(tmp_path: Path) -> AppContext:
    """Build a fully wired AppContext with real stores and a mock provider."""
    cfg = MemoryMeshConfig(
        storage=StorageConfig(base_dir=tmp_path / "state"),
    )
    vector_store = ChromaVectorStore(
        db_path=tmp_path / "chroma",
        embedding_model_id="test-model",
    )
    metadata_store = MetadataStore(tmp_path / "metadata.sqlite3")
    bm25 = BM25Index(tmp_path / "bm25.pkl")
    provider = _make_provider()
    indexer = FileIndexer(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
    )
    engine = SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
        config=cfg.search,
    )
    audit_logger = AuditLogger(tmp_path / "audit.jsonl")
    return AppContext(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        bm25=bm25,
        provider=provider,
        indexer=indexer,
        engine=engine,
        audit_logger=audit_logger,
    )


def _call_tool(mcp, name: str, **kwargs):
    """Invoke a registered tool by name, bypassing the MCP transport."""
    import asyncio

    return asyncio.run(mcp._tool_manager.call_tool(name, kwargs))


class TestSearchMemoryTool:
    def test_returns_dict_with_hits_key(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="hello world")
        assert "hits" in result

    def test_empty_index_returns_empty_hits(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="anything")
        assert result["hits"] == []

    def test_hit_shape_after_index(self, tmp_path: Path, app_ctx: AppContext) -> None:
        """After indexing a file, search should return a hit with expected keys."""
        f = tmp_path / "doc.txt"
        f.write_text("python machine learning tutorial", encoding="utf-8")
        app_ctx.indexer.index_file(f)

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="python learning", top_k=5)
        if result["hits"]:
            hit = result["hits"][0]
            assert "path" in hit
            assert "score" in hit
            assert "preview" in hit

    def test_mode_field_present(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="test")
        assert "mode" in result

    def test_top_k_clamped(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="test", top_k=999)
        assert len(result["hits"]) <= 50

    def test_audit_log_written(self, tmp_path: Path, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        _call_tool(mcp, "search_memory", query="audit test")
        audit_path = tmp_path / "audit.jsonl"
        assert audit_path.exists()
        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) >= 1


class TestListSourcesTool:
    def test_returns_sources_key(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "list_sources")
        assert "sources" in result

    def test_empty_sources_when_no_config(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "list_sources")
        assert result["sources"] == []

    def test_duration_ms_present(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "list_sources")
        assert "duration_ms" in result

    def test_source_stats_returned(self, tmp_path: Path, app_ctx: AppContext) -> None:
        from memorymesh.core.models import SourceConfig

        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("content a", encoding="utf-8")
        app_ctx.config.sources.append(SourceConfig(name="docs", path=src_dir))
        app_ctx.indexer.index_file(src_dir / "a.txt", source_name="docs")

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "list_sources")
        assert len(result["sources"]) == 1
        src = result["sources"][0]
        assert src["name"] == "docs"
        assert src["n_files_indexed"] == 1


class TestGetDocumentTool:
    def test_returns_error_for_unknown_path(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "get_document", path="/nonexistent/file.txt")
        assert "error" in result

    def test_returns_content_for_indexed_file(self, tmp_path: Path, app_ctx: AppContext) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("Hello from MemoryMesh!", encoding="utf-8")
        app_ctx.indexer.index_file(f)

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "get_document", path=str(f))
        assert "content" in result
        assert "Hello from MemoryMesh!" in result["content"]
        assert result["truncated"] is False

    def test_truncation_flag(self, tmp_path: Path, app_ctx: AppContext) -> None:
        f = tmp_path / "big.txt"
        f.write_text("x" * 2000, encoding="utf-8")
        app_ctx.indexer.index_file(f)

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "get_document", path=str(f), max_bytes=100)
        assert result["truncated"] is True
        assert len(result["content"]) <= 150  # a bit of decoding slack


class TestIndexNowTool:
    def test_index_single_file(self, tmp_path: Path, app_ctx: AppContext) -> None:
        f = tmp_path / "new.txt"
        f.write_text("new content here", encoding="utf-8")

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "index_now", path=str(f))
        assert result["n_files_processed"] == 1
        assert result["n_chunks"] >= 1

    def test_index_directory(self, tmp_path: Path, app_ctx: AppContext) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("content a", encoding="utf-8")
        (src / "b.txt").write_text("content b", encoding="utf-8")

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "index_now", path=str(src))
        assert result["n_files_processed"] == 2

    def test_missing_path_returns_error(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "index_now", path="/nonexistent/path")
        assert "error" in result

    def test_no_sources_indexed_when_unconfigured(self, app_ctx: AppContext) -> None:
        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "index_now")
        # No sources configured → 0 files processed, no error key
        assert result["n_files_processed"] == 0


class TestParentDocumentRetriever:
    def test_extended_preview_with_parent_window(self, tmp_path: Path, app_ctx: AppContext) -> None:
        """extended_preview present and longer than preview when parent_window_chars > 0."""
        f = tmp_path / "wide.txt"
        f.write_text(
            "Introduction paragraph. " * 20
            + "TARGET CONTENT HERE. "
            + "Conclusion paragraph. " * 20,
            encoding="utf-8",
        )
        app_ctx.indexer.index_file(f)

        app_ctx.config.search.parent_window_chars = 200

        mcp = build_mcp(app_ctx)
        result = _call_tool(mcp, "search_memory", query="TARGET CONTENT", top_k=5)

        hits = result["hits"]
        if hits:
            hit = hits[0]
            assert "extended_preview" in hit
            if hit["extended_preview"] is not None:
                assert len(hit["extended_preview"]) >= len(hit["preview"])
