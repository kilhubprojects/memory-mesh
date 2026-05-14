"""In-process embedding cache for MemoryMesh (Wave 6).

Wraps any :class:`~memorymesh.embeddings.base.EmbeddingProvider` with an LRU
cache keyed on the input text string.  This eliminates redundant embedding
calls for repeated queries and document re-indexes when the file content has
not changed.

Cache design
------------
* **Keying**: SHA-256 of ``model_id + ":" + text`` so cache entries are
  invalidated automatically when the model changes.
* **Capacity**: Configurable maximum entry count (default 2 048).  Older
  entries are evicted LRU-style when the limit is hit.
* **Thread safety**: A ``threading.Lock`` guards the underlying dict.
* **Persistence**: The cache is purely in-process and is discarded on daemon
  restart.  Persistence is intentionally out of scope — re-warming from cold
  takes < 1 ms per document on a GPU.

Usage
-----
Wrap your provider at startup::

    from memorymesh.embeddings.cache import CachedEmbeddingProvider
    provider = CachedEmbeddingProvider(base_provider, capacity=4096)
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from loguru import logger

from memorymesh.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """LRU-cached wrapper around any :class:`EmbeddingProvider`.

    Args:
        provider: The underlying provider to cache calls for.
        capacity: Maximum number of individual text embeddings to hold in RAM.
            Each embedding is a list of 32-bit floats (e.g. 384 or 768 floats).
            At 384 dims, 4 096 entries ≈ 6 MB.
    """

    def __init__(self, provider: EmbeddingProvider, capacity: int = 2048) -> None:
        self._provider = provider
        self._capacity = max(1, capacity)
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def model_id(self) -> str:
        """Delegate to the underlying provider."""
        return self._provider.model_id

    @property
    def embedding_dim(self) -> int:
        """Delegate to the underlying provider."""
        return self._provider.embedding_dim

    def _key(self, text: str) -> str:
        """Build a stable cache key from model_id + text content."""
        raw = f"{self.model_id}:{text}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _get(self, key: str) -> list[float] | None:
        """Return the cached vector for *key*, or ``None`` on miss."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def _set(self, key: str, vector: list[float]) -> None:
        """Store *vector* under *key*, evicting LRU entries on overflow."""
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts*, returning cached vectors where available.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors in the same order as *texts*.
        """
        keys = [self._key(t) for t in texts]
        result: list[list[float] | None] = [None] * len(texts)

        # Collect cache hits.
        miss_indices: list[int] = []
        for i, key in enumerate(keys):
            vec = self._get(key)
            if vec is not None:
                result[i] = vec
            else:
                miss_indices.append(i)

        # Batch-embed cache misses.
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            miss_vecs = self._provider.embed_documents(miss_texts)
            for _i, (idx, vec) in enumerate(zip(miss_indices, miss_vecs, strict=False)):
                result[idx] = vec
                self._set(keys[idx], vec)

        total = len(texts)
        hits = total - len(miss_indices)
        logger.debug(
            f"EmbeddingCache: {hits}/{total} hits "
            f"(cumulative hits={self._hits} misses={self._misses})"
        )
        return [r for r in result if r is not None]  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string, returning a cached vector if available.

        Args:
            text: Query string to embed.

        Returns:
            L2-normalised embedding vector.
        """
        key = self._key(text)
        vec = self._get(key)
        if vec is not None:
            return vec
        vec = self._provider.embed_query(text)
        self._set(key, vec)
        return vec

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return current hit/miss counters and cache size.

        Returns:
            Dict with ``hits``, ``misses``, ``size``, and ``capacity`` keys.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "capacity": self._capacity,
            }

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.debug("EmbeddingCache: cleared")
