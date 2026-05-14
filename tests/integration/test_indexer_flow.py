"""Integration tests for the full indexer pipeline.

These tests use real files in tmp_path, a real ChromaVectorStore, a real
BM25Index, and a mocked EmbeddingProvider (to avoid downloading a model).
They verify that the end-to-end flow from file → search hit works correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memorymesh.core.models import MemoryMeshConfig, StorageConfig
from memorymesh.indexer.file_indexer import FileIndexer
from memorymesh.search.engine import SearchEngine
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.metadata_store import MetadataStore
from memorymesh.storage.vector_store import ChromaVectorStore

_DIM = 16
_COUNTER = [0]


def _deterministic_vec(text: str) -> list[float]:
    """Produce a reproducible vector from text (for test repeatability)."""
    seed = sum(ord(c) for c in text) % 1000
    import random as _r

    rng = _r.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(_DIM)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v] if norm > 0 else v


def _make_stack(tmp_path: Path) -> tuple[FileIndexer, SearchEngine]:
    config = MemoryMeshConfig(storage=StorageConfig(base_dir=tmp_path / "state"))
    metadata_store = MetadataStore(tmp_path / "metadata.sqlite3")
    vector_store = ChromaVectorStore(db_path=tmp_path / "chroma", embedding_model_id="mock-model")
    bm25 = BM25Index(tmp_path / "bm25.pkl")

    provider = MagicMock()
    provider.model_id = "mock-model"
    provider.embed_documents.side_effect = lambda texts: [_deterministic_vec(t) for t in texts]
    provider.embed_query.side_effect = _deterministic_vec

    indexer = FileIndexer(
        config=config,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
    )
    engine = SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
    )
    return indexer, engine


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Create a small corpus of 5 diverse files."""
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "python_intro.txt").write_text(
        "Python is a high-level programming language. It is easy to learn.",
        encoding="utf-8",
    )
    (src / "ml_notes.md").write_text(
        "# Machine Learning\n\nNeural networks learn from data.\n\n"
        "## Deep Learning\n\nConvolutional networks process images.\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        'def add(a: int, b: int) -> int:\n    """Return a + b."""\n    return a + b\n',
        encoding="utf-8",
    )
    (src / "readme.md").write_text(
        "# Project\n\nThis is the project readme.\n",
        encoding="utf-8",
    )
    (src / "notes.txt").write_text(
        "Meeting notes: discuss architecture and deployment strategy.",
        encoding="utf-8",
    )
    return src


class TestFullIndexFlow:
    def test_indexes_all_files(self, tmp_path: Path, corpus: Path) -> None:
        indexer, _ = _make_stack(tmp_path)
        results = indexer.index_directory(corpus)
        indexed = [r for r in results if r.status == "indexed"]
        assert len(indexed) == 5

    def test_metadata_written_to_sqlite(self, tmp_path: Path, corpus: Path) -> None:
        indexer, _ = _make_stack(tmp_path)
        indexer.index_directory(corpus)
        records = indexer._metadata_store.list_files(status="indexed")
        assert len(records) == 5

    def test_second_scan_skips_unchanged_files(self, tmp_path: Path, corpus: Path) -> None:
        indexer, _ = _make_stack(tmp_path)
        indexer.index_directory(corpus)
        results2 = indexer.index_directory(corpus)
        skipped = [r for r in results2 if r.status == "skipped"]
        assert len(skipped) == 5

    def test_modified_file_reindexed(self, tmp_path: Path, corpus: Path) -> None:
        indexer, _ = _make_stack(tmp_path)
        indexer.index_directory(corpus)

        modified = corpus / "notes.txt"
        modified.write_text("Updated content about indexing strategies.", encoding="utf-8")

        results2 = indexer.index_directory(corpus)
        reindexed = [r for r in results2 if r.status == "indexed"]
        assert len(reindexed) == 1
        assert reindexed[0].path == modified

    def test_delete_file_removes_from_metadata(self, tmp_path: Path, corpus: Path) -> None:
        indexer, _ = _make_stack(tmp_path)
        indexer.index_directory(corpus)

        target = corpus / "notes.txt"
        indexer.delete_file(target)
        record = indexer._metadata_store.get_file(str(target))
        assert record is not None
        assert record.status == "deleted"


class TestSearchAfterIndex:
    def test_dense_search_returns_hits(self, tmp_path: Path, corpus: Path) -> None:
        indexer, engine = _make_stack(tmp_path)
        indexer.index_directory(corpus)
        response = engine.search("programming language", top_k=3, mode="dense")
        assert len(response.hits) >= 1

    def test_sparse_search_finds_keyword(self, tmp_path: Path, corpus: Path) -> None:
        indexer, engine = _make_stack(tmp_path)
        indexer.index_directory(corpus)

        # Rebuild BM25 from indexed files
        records = indexer._metadata_store.list_files(status="indexed")
        from memorymesh.parsing.registry import build_registry

        registry = build_registry()
        docs = []
        for rec in records:
            p = Path(rec.path)
            parser = registry.get(p.suffix.lower())
            if parser:
                try:
                    parsed = parser.parse(p)
                    docs.append((f"{rec.path}:0", rec.path, parsed.text))
                except Exception:
                    pass
        engine._bm25.rebuild(docs)

        response = engine.search("neural networks", top_k=5, mode="sparse")
        assert len(response.hits) >= 1
        assert any("ml_notes" in h.path for h in response.hits)

    def test_hybrid_search_returns_response(self, tmp_path: Path, corpus: Path) -> None:
        indexer, engine = _make_stack(tmp_path)
        indexer.index_directory(corpus)
        response = engine.search("project architecture", top_k=5, mode="hybrid")
        assert response.mode in ("hybrid", "dense")  # dense if BM25 is empty
        assert response.duration_ms >= 0.0
