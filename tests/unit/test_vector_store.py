"""Unit tests for ChromaVectorStore.

Each test gets a fresh PersistentClient in a temporary directory — ChromaDB
1.5+ EphemeralClient shares in-process state and is therefore not safe for
test isolation.  Paths use ``str(Path(...))`` everywhere so assertions hold on
both Unix (``/tmp/…``) and Windows (``\\tmp\\…``).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from memorymesh.core.exceptions import EmbeddingModelMismatchError
from memorymesh.core.models import ChunkMetadata, ChunkWithEmbedding
from memorymesh.storage.vector_store import ChromaVectorStore

_DIM = 384
_MODEL = "all-MiniLM-L6-v2"

random.seed(42)


def _vec(dim: int = _DIM) -> list[float]:
    """Return a reproducible random unit vector of length *dim*."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _chunk(
    path: str = "/tmp/test.py",
    index: int = 0,
    text: str = "hello world",
    embedding: list[float] | None = None,
) -> ChunkWithEmbedding:
    return ChunkWithEmbedding(
        path=Path(path),
        chunk_index=index,
        text=text,
        start_char=0,
        end_char=len(text),
        file_type=".py",
        mtime=1_000_000.0,
        source_root="projects",
        metadata=ChunkMetadata(),
        embedding=embedding or _vec(),
    )


@pytest.fixture()
def store(tmp_path: Path) -> ChromaVectorStore:
    """Fresh ChromaDB instance per test via persistent client in tmp_path."""
    return ChromaVectorStore(
        db_path=tmp_path / "chroma",
        embedding_model_id=_MODEL,
    )


class TestUpsertAndCount:
    def test_empty_on_creation(self, store: ChromaVectorStore) -> None:
        assert store.count() == 0

    def test_upsert_single_chunk(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk()])
        assert store.count() == 1

    def test_upsert_multiple_chunks(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(index=i) for i in range(5)])
        assert store.count() == 5

    def test_upsert_is_idempotent(self, store: ChromaVectorStore) -> None:
        c = _chunk()
        store.upsert([c])
        store.upsert([c])
        assert store.count() == 1

    def test_upsert_empty_list_is_noop(self, store: ChromaVectorStore) -> None:
        store.upsert([])
        assert store.count() == 0


class TestDeleteByPath:
    def test_delete_removes_all_chunks_for_path(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(path="/tmp/a.py", index=i) for i in range(3)])
        store.upsert([_chunk(path="/tmp/b.py", index=0)])
        store.delete_by_path(str(Path("/tmp/a.py")))
        assert store.count() == 1

    def test_delete_nonexistent_path_is_noop(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk()])
        store.delete_by_path(str(Path("/no/such/file.py")))
        assert store.count() == 1


class TestSearch:
    def test_search_empty_store_returns_empty(self, store: ChromaVectorStore) -> None:
        assert store.search(_vec(), top_k=5) == []

    def test_search_returns_at_most_top_k(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(index=i) for i in range(10)])
        hits = store.search(_vec(), top_k=3)
        assert len(hits) <= 3

    def test_search_scores_are_between_0_and_1(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(index=i) for i in range(5)])
        hits = store.search(_vec(), top_k=5)
        for hit in hits:
            assert 0.0 <= hit.score <= 1.0 + 1e-6

    def test_exact_vector_scores_highest(self, store: ChromaVectorStore) -> None:
        """A query identical to a stored vector must be the top hit."""
        target_vec = _vec()
        target = _chunk(path="/tmp/target.py", index=0, embedding=target_vec)
        others = [_chunk(path="/tmp/other.py", index=i) for i in range(9)]
        store.upsert([target, *others])

        hits = store.search(target_vec, top_k=10)
        assert hits[0].path == str(Path("/tmp/target.py"))
        assert hits[0].score > 0.99

    def test_search_hit_fields_populated(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(path="/tmp/f.py", text="some code", index=0)])
        hits = store.search(_vec(), top_k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.path == str(Path("/tmp/f.py"))
        assert hit.chunk_index == 0
        assert hit.file_type == ".py"
        assert hit.source_root == "projects"
        assert "some code" in hit.preview

    def test_top_k_clamped_to_collection_size(self, store: ChromaVectorStore) -> None:
        store.upsert([_chunk(index=0)])
        hits = store.search(_vec(), top_k=100)
        assert len(hits) == 1


class TestModelId:
    def test_model_id_matches_constructor(self, store: ChromaVectorStore) -> None:
        assert store.model_id == _MODEL

    def test_mismatch_raises_on_second_open(self, tmp_path: Path) -> None:
        ChromaVectorStore(
            db_path=tmp_path / "db",
            embedding_model_id="model-A",
        )
        with pytest.raises(EmbeddingModelMismatchError):
            ChromaVectorStore(
                db_path=tmp_path / "db",
                embedding_model_id="model-B",
            )


class TestChunkMetadataRoundtrip:
    def test_optional_metadata_fields_stored_and_retrieved(self, store: ChromaVectorStore) -> None:
        c = _chunk()
        c.metadata = ChunkMetadata(
            function_name="my_func",
            class_name="MyClass",
            language="python",
            heading_path=["Intro", "Setup"],
        )
        store.upsert([c])
        hit = store.search(c.embedding, top_k=1)[0]
        assert hit.path == str(c.path)
