"""Semantic deduplication for indexed chunks.

Chunks that are near-duplicates (cosine similarity ≥ threshold) of already-seen
chunks in the current indexing batch are dropped before storage.  This reduces
index bloat when documents are re-ingested or when connectors return overlapping
content.

The deduplicator maintains an in-memory seen-list per indexing session; it does
not persist state across restarts (the index itself prevents full re-insertion of
identical chunks via the hash guard in FileIndexer).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import ChunkWithEmbedding


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticDeduplicator:
    """Drop near-duplicate chunks within an indexing session.

    Args:
        threshold: Cosine-similarity cutoff.  Chunks with similarity >=
            *threshold* to any previously-seen chunk are considered duplicates
            and discarded.
    """

    def __init__(self, threshold: float = 0.97) -> None:
        self._threshold = threshold
        self._seen: list[list[float]] = []

    def filter(self, chunks: list[ChunkWithEmbedding]) -> list[ChunkWithEmbedding]:
        """Return *chunks* with near-duplicates removed.

        Args:
            chunks: Embedded chunks to de-duplicate.

        Returns:
            Subset of *chunks* with no near-duplicate pairs relative to chunks
            seen in prior ``filter()`` calls during the same session.
        """
        kept: list[ChunkWithEmbedding] = []
        for chunk in chunks:
            vec = chunk.embedding
            is_dup = any(_cosine(vec, seen_vec) >= self._threshold for seen_vec in self._seen)
            if is_dup:
                logger.debug(
                    f"SemanticDeduplicator: dropped duplicate chunk "
                    f"{chunk.path}:{chunk.chunk_index}"
                )
            else:
                self._seen.append(vec)
                kept.append(chunk)
        return kept

    def reset(self) -> None:
        """Clear the seen-list; call between independent documents if desired."""
        self._seen.clear()
