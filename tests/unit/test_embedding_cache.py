"""Unit tests for Wave 6 CachedEmbeddingProvider."""

from __future__ import annotations

from unittest.mock import MagicMock

from memorymesh.embeddings.cache import CachedEmbeddingProvider


def _mock_provider(dim: int = 4) -> MagicMock:
    provider = MagicMock(spec=["model_id", "embedding_dim", "embed_documents", "embed_query"])
    provider.model_id = "mock-4d"
    provider.embedding_dim = dim
    provider.embed_documents.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    provider.embed_query.side_effect = lambda text: [0.2] * dim
    return provider


class TestCachedEmbeddingProvider:
    def test_model_id_delegates(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base)
        assert cached.model_id == "mock-4d"

    def test_embed_documents_calls_base_on_miss(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=100)
        result = cached.embed_documents(["hello", "world"])
        assert len(result) == 2
        base.embed_documents.assert_called_once()

    def test_embed_documents_cache_hit_avoids_base(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=100)
        # First call — miss.
        cached.embed_documents(["hello"])
        # Second call — should be a cache hit.
        cached.embed_documents(["hello"])
        assert base.embed_documents.call_count == 1

    def test_embed_query_cached(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=100)
        cached.embed_query("search term")
        cached.embed_query("search term")
        assert base.embed_query.call_count == 1

    def test_partial_batch_cache_hit(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=100)
        cached.embed_documents(["a", "b"])
        base.embed_documents.reset_mock()
        # "b" is cached; "c" is new → should call base with ["c"] only.
        result = cached.embed_documents(["b", "c"])
        assert len(result) == 2
        call_args = base.embed_documents.call_args
        assert call_args is not None
        assert call_args[0][0] == ["c"]

    def test_lru_eviction(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=2)
        cached.embed_documents(["first"])
        cached.embed_documents(["second"])
        base.embed_documents.reset_mock()
        # "first" should have been evicted when "third" is added.
        cached.embed_documents(["third"])
        base.embed_documents.reset_mock()
        # Re-embedding "first" should miss.
        cached.embed_documents(["first"])
        base.embed_documents.assert_called_once()

    def test_cache_stats(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=10)
        cached.embed_query("q1")
        cached.embed_query("q1")  # hit
        cached.embed_query("q2")
        stats = cached.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert stats["capacity"] == 10

    def test_clear_empties_cache(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=10)
        cached.embed_query("hello")
        cached.clear()
        assert cached.cache_stats["size"] == 0

    def test_model_change_invalidates_entries(self) -> None:
        """Different model_ids produce different keys — no cross-contamination."""
        base1 = _mock_provider()
        base1.model_id = "model-A"
        base2 = _mock_provider()
        base2.model_id = "model-B"

        cached1 = CachedEmbeddingProvider(base1, capacity=10)
        cached2 = CachedEmbeddingProvider(base2, capacity=10)

        cached1.embed_query("test")
        # cached2 has no entry for "test" despite same text.
        cached2.embed_query("test")
        assert base2.embed_query.call_count == 1

    def test_empty_texts_returns_empty(self) -> None:
        base = _mock_provider()
        cached = CachedEmbeddingProvider(base, capacity=10)
        result = cached.embed_documents([])
        assert result == []
