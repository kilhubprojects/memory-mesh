"""Cross-encoder reranker for hybrid search results.

Reranking is a two-stage retrieval pattern: first retrieve a wide candidate
pool via fast approximate methods (RRF over dense + BM25), then rerank the
top candidates using an expensive but accurate cross-encoder that scores
each (query, passage) pair jointly.

The model is loaded lazily on the first :meth:`CrossEncoderReranker.rerank`
call so that ``enabled: false`` has zero overhead at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import SearchHit, SearchRerankerConfig


class CrossEncoderReranker:
    """Reranks a list of :class:`~memorymesh.core.models.SearchHit` objects.

    Uses a sentence-transformers ``CrossEncoder`` model to score each
    ``(query, preview)`` pair and reorder hits by the absolute cross-encoder
    score, replacing the RRF score in the returned hits.

    Args:
        config: Reranker configuration including model name and candidate pool
            size.
    """

    def __init__(self, config: SearchRerankerConfig) -> None:
        self._config = config
        self._model: object | None = None
        self._load_failed: bool = False

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        """Rerank ``hits`` by cross-encoder score and return the top ``top_k``.

        Falls back to the original RRF ordering when the model is unavailable
        or prediction raises an exception - the daemon never crashes due to a
        reranker failure.

        Args:
            query: The original search query string.
            hits: Candidate hits from the RRF fusion step.
            top_k: Number of hits to return after reranking.

        Returns:
            Up to ``top_k`` hits sorted by descending cross-encoder score.
            The ``score`` field of each returned hit reflects the raw
            cross-encoder logit (not normalised).
        """
        if not hits:
            return hits

        model = self._get_model()
        if model is None:
            logger.warning("Reranker: model unavailable - returning original RRF order")
            return hits[:top_k]

        try:
            from memorymesh.core.models import SearchHit as _Hit

            pairs = [(query, h.preview) for h in hits]
            scores: list[float] = model.predict(pairs).tolist()  # type: ignore[attr-defined,union-attr]

            ranked = sorted(zip(hits, scores, strict=False), key=lambda x: x[1], reverse=True)

            result: list[SearchHit] = []
            for hit, score in ranked[:top_k]:
                result.append(
                    _Hit(
                        path=hit.path,
                        chunk_index=hit.chunk_index,
                        score=round(float(score), 6),
                        preview=hit.preview,
                        file_type=hit.file_type,
                        mtime=hit.mtime,
                        source_root=hit.source_root,
                        start_char=hit.start_char,
                        end_char=hit.end_char,
                        extended_preview=hit.extended_preview,
                    )
                )

            logger.info(
                f"Reranker: reranked={len(hits)} returned={len(result)} "
                f"model={self._config.model!r}"
            )
            return result

        except Exception as exc:
            logger.warning(f"Reranker: prediction failed, returning original order: {exc}")
            return hits[:top_k]

    def _get_model(self) -> object | None:
        """Return the loaded CrossEncoder, loading lazily on first call."""
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self._config.model)
            logger.info(f"Reranker: loaded model={self._config.model!r}")
        except Exception as exc:
            logger.warning(f"Reranker: failed to load model={self._config.model!r}: {exc}")
            self._load_failed = True

        return self._model
