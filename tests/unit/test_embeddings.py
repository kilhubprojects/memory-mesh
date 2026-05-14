"""Unit tests for the embeddings layer.

SentenceTransformers loads a real model (~90 MB on first run).  Tests are
marked with ``@pytest.mark.slow`` so CI can skip them with ``-m "not slow"``.
The model is loaded once per session via a module-scoped fixture.
"""

from __future__ import annotations

import pytest

from memorymesh.core.exceptions import ConfigError
from memorymesh.embeddings.base import EmbeddingProvider
from memorymesh.embeddings.registry import get_provider


@pytest.fixture(scope="module")
def provider() -> EmbeddingProvider:
    """Module-scoped SentenceTransformers provider — loads model once."""
    from memorymesh.core.models import EmbeddingsConfig
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    cfg = EmbeddingsConfig(model="all-MiniLM-L6-v2", device="cpu", batch_size=8)
    return SentenceTransformersProvider(config=cfg)


@pytest.mark.slow
class TestSentenceTransformersProvider:
    def test_model_id_matches_config(self, provider: EmbeddingProvider) -> None:
        assert provider.model_id == "all-MiniLM-L6-v2"

    def test_embed_query_returns_list_of_floats(self, provider: EmbeddingProvider) -> None:
        vec = provider.embed_query("hello world")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_query_correct_dim(self, provider: EmbeddingProvider) -> None:
        vec = provider.embed_query("test")
        assert len(vec) == 384  # all-MiniLM-L6-v2 output dim

    def test_embed_documents_returns_correct_count(self, provider: EmbeddingProvider) -> None:
        texts = ["first doc", "second doc", "third doc"]
        vecs = provider.embed_documents(texts)
        assert len(vecs) == 3

    def test_embed_documents_all_same_dim(self, provider: EmbeddingProvider) -> None:
        vecs = provider.embed_documents(["a", "bb", "ccc"])
        dims = {len(v) for v in vecs}
        assert len(dims) == 1

    def test_embed_documents_empty_list_returns_empty(self, provider: EmbeddingProvider) -> None:
        assert provider.embed_documents([]) == []

    def test_similar_texts_higher_score(self, provider: EmbeddingProvider) -> None:
        """Semantically close texts should have higher cosine similarity than unrelated ones."""
        v_a = provider.embed_query("The cat sat on the mat")
        v_b = provider.embed_query("A cat was sitting on a rug")
        v_c = provider.embed_query("Quantum mechanics and black holes")

        def cosine(x: list[float], y: list[float]) -> float:
            dot = sum(a * b for a, b in zip(x, y, strict=True))
            nx = sum(a**2 for a in x) ** 0.5
            ny = sum(b**2 for b in y) ** 0.5
            return dot / (nx * ny) if nx and ny else 0.0

        sim_ab = cosine(v_a, v_b)
        sim_ac = cosine(v_a, v_c)
        assert sim_ab > sim_ac

    def test_embedding_dim_property(self, provider: EmbeddingProvider) -> None:
        assert provider.embedding_dim == 384


class TestRegistry:
    def test_get_provider_returns_sentence_transformers(self) -> None:
        from memorymesh.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )

        p = get_provider()
        assert isinstance(p, SentenceTransformersProvider)

    def test_get_provider_with_none_uses_defaults(self) -> None:
        p = get_provider(None)
        assert p.model_id == "all-MiniLM-L6-v2"

    def test_get_provider_unknown_raises_config_error(self) -> None:
        from memorymesh.core.models import EmbeddingsConfig

        cfg = EmbeddingsConfig(provider="unknown_provider")
        with pytest.raises(ConfigError, match="Unknown embedding provider"):
            get_provider(cfg)
