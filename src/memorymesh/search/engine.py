"""Hybrid search engine: dense (ChromaDB) + sparse (BM25) fused via RRF.

The engine is stateless between queries - it delegates to the vector store and
BM25 index for retrieval, then fuses results.

Search modes:
- ``"hybrid"`` (default) - dense + sparse + RRF.
- ``"dense"``  - dense vector search only.
- ``"sparse"`` - BM25 only.

Optional post-processing (controlled by ``SearchConfig``):
- Cross-encoder reranking: re-scores an over-fetched candidate pool.
- Query expansion: generates lexical variants and/or HyDE vectors for
  parallel retrieval, merging all candidate lists via RRF before reranking.

Privacy invariant: query text is never logged - only mode, top_k, and latency.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

from memorymesh.core.models import SearchHit, SearchResponse
from memorymesh.search.rank_fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from memorymesh.core.models import OllamaConfig, SearchConfig
    from memorymesh.embeddings.base import EmbeddingProvider
    from memorymesh.search.query_expander import QueryExpander
    from memorymesh.search.reranker import CrossEncoderReranker
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.vector_store import VectorStore

SearchMode = Literal["hybrid", "dense", "sparse"]


class SearchEngine:
    """Orchestrates dense + sparse search with optional RRF fusion, query
    expansion, and cross-encoder reranking.

    Args:
        vector_store: ChromaDB vector store for dense retrieval.
        bm25_index: BM25 index for sparse retrieval.
        embedding_provider: Converts query text to a vector.
        config: Search configuration (top_k, RRF k, over-fetch factor,
            reranker, query expansion).
        ollama_config: Ollama settings for HyDE-based query expansion.
            ``None`` disables HyDE regardless of ``config.query_expansion``.
        metadata_store: Optional SQLite store used to filter out suppressed
            chunks (``forgotten_chunks`` table).  ``None`` disables suppression
            filtering (safe default for tests that don't wire this dep).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        config: SearchConfig | None = None,
        ollama_config: OllamaConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        from memorymesh.core.models import SearchConfig as _Cfg

        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._provider = embedding_provider
        self._cfg: SearchConfig = config or _Cfg()
        self._ollama_cfg = ollama_config
        self._metadata_store = metadata_store

        self._reranker: CrossEncoderReranker | None = None
        self._expander: QueryExpander | None = None

        if self._cfg.reranker.enabled:
            from memorymesh.search.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(self._cfg.reranker)

        if self._cfg.query_expansion.enabled:
            from memorymesh.search.query_expander import QueryExpander

            self._expander = QueryExpander(
                config=self._cfg.query_expansion,
                embedding_provider=embedding_provider,
                ollama_config=ollama_config,
            )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | SearchMode = "hybrid",
        filter_: dict[str, object] | None = None,
        source: str | None = None,
        file_type: str | None = None,
        after_ts: float | None = None,
        before_ts: float | None = None,
        modality: str = "all",
    ) -> SearchResponse:
        """Run a search query and return fused results.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return.  Defaults to
                ``config.default_top_k``.
            mode: Search mode - ``"hybrid"``, ``"dense"``, or ``"sparse"``.
            filter_: Optional ChromaDB ``where`` filter applied to dense
                search only.
            source: If set, keep only hits whose ``source_root`` equals this.
            file_type: If set, keep only hits whose ``file_type`` equals this.
            after_ts: If set, keep only hits whose ``mtime`` >= this timestamp.
            before_ts: If set, keep only hits whose ``mtime`` <= this timestamp.
            modality: ``"text"`` excludes image hits; ``"image"`` keeps only
                image hits; ``"all"`` (default) applies no modality filter.

        Returns:
            :class:`~memorymesh.core.models.SearchResponse` with ranked hits.
        """
        t0 = time.perf_counter()
        k = top_k if top_k is not None else self._cfg.default_top_k

        effective_mode: SearchMode = cast(SearchMode, mode)
        if mode == "hybrid" and not self._cfg.hybrid.enabled:
            effective_mode = "dense"

        # Enlarge candidate pool when reranker is active
        candidate_k = k
        if self._reranker is not None:
            candidate_k = max(k, self._cfg.reranker.top_k_before_rerank)

        # Build all query strings (original + lexical variants)
        queries: list[str] = [query]
        extra_vecs: list[list[float]] = []
        if self._expander is not None:
            queries = self._expander.expand_queries(query)
            if self._cfg.query_expansion.use_hyde:
                extra_vecs = self._expander.expand_vectors(query)

        # Retrieve candidates for each query variant in parallel
        all_hit_lists: list[list[SearchHit]] = []

        if len(queries) == 1 and not extra_vecs:
            hits = self._retrieve_single(queries[0], effective_mode, candidate_k, filter_)
            all_hit_lists.append(hits)
        else:
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                futs = {
                    pool.submit(self._retrieve_single, q, effective_mode, candidate_k, filter_): q
                    for q in queries
                }
                for fut in as_completed(futs):
                    try:
                        all_hit_lists.append(fut.result())
                    except Exception as exc:
                        logger.warning(f"SearchEngine: variant retrieval failed: {exc}")

            # Extra HyDE vectors: embed-then-search in dense mode
            for vec in extra_vecs:
                try:
                    hyde_hits = self._vector_store.search(vec, top_k=candidate_k, filter_=filter_)
                    all_hit_lists.append(hyde_hits)
                except Exception as exc:
                    logger.warning(f"SearchEngine: HyDE dense search failed: {exc}")

        # Fuse all candidate lists
        if len(all_hit_lists) == 1:
            fused = all_hit_lists[0][:candidate_k]
        else:
            fused = self._fuse_all(all_hit_lists, candidate_k)

        # Rerank if enabled
        if self._reranker is not None and fused:
            final_hits = self._reranker.rerank(query, fused, k)
        else:
            final_hits = fused[:k]

        # Modality post-filter
        _image_exts = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
        if modality == "image":
            final_hits = [h for h in final_hits if h.file_type in _image_exts]
        elif modality == "text":
            final_hits = [h for h in final_hits if h.file_type not in _image_exts]

        # Faceted post-filtering
        if source or file_type or after_ts is not None or before_ts is not None:
            filtered: list[SearchHit] = []
            for h in final_hits:
                if source and h.source_root != source:
                    continue
                if file_type and h.file_type != file_type:
                    continue
                if after_ts is not None and (h.mtime or 0) < after_ts:
                    continue
                if before_ts is not None and (h.mtime or 0) > before_ts:
                    continue
                filtered.append(h)
            final_hits = filtered

        # Suppression filter - remove chunks in the forgotten_chunks table
        if self._metadata_store is not None:
            try:
                forgotten: frozenset[str] = frozenset(self._metadata_store.list_forgotten())
                if forgotten:
                    final_hits = [
                        h for h in final_hits if f"{h.path}:{h.chunk_index}" not in forgotten
                    ]
            except Exception as exc:
                logger.warning(f"SearchEngine: could not load suppression list: {exc}")

        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"SearchEngine: mode={effective_mode!r} top_k={k} variants={len(queries)} "
            f"candidates={len(fused)} returned={len(final_hits)} latency={duration_ms:.1f}ms"
        )
        from memorymesh.observability.metrics import get_metrics

        get_metrics().record_search(duration_ms, mode=effective_mode, returned=len(final_hits))
        return SearchResponse(
            hits=final_hits,
            mode=effective_mode,
            duration_ms=round(duration_ms, 2),
        )

    def _retrieve_single(
        self,
        query: str,
        mode: SearchMode,
        top_k: int,
        filter_: dict[str, object] | None,
    ) -> list[SearchHit]:
        """Run dense + sparse retrieval for a single query and fuse."""
        over_fetch = top_k * self._cfg.hybrid.over_fetch_factor

        dense_hits: list[SearchHit] = []
        sparse_hits: list[SearchHit] = []

        if mode in ("hybrid", "dense"):
            query_vec = self._provider.embed_query(query)
            dense_hits_raw = self._vector_store.search(query_vec, top_k=over_fetch, filter_=filter_)
            # Summary chunks (chunk_index=-1) are used only to improve retrieval
            # recall - they must not appear in results shown to the user.
            dense_hits = [h for h in dense_hits_raw if h.chunk_index != -1]

        if mode in ("hybrid", "sparse"):
            # Summary chunks are stored with chunk_index=-1 - exclude them.
            sparse_hits = [
                h for h in self._bm25.search(query, top_k=over_fetch) if h.chunk_index != -1
            ]

        if mode == "dense":
            return dense_hits[:top_k]
        if mode == "sparse":
            return sparse_hits[:top_k]
        return self._fuse(dense_hits, sparse_hits, top_k)

    def _fuse(
        self,
        dense_hits: list[SearchHit],
        sparse_hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        """Fuse dense and sparse hits via RRF, return top_k merged results."""
        dense_ids = [self._hit_id(h) for h in dense_hits]
        sparse_ids = [self._hit_id(h) for h in sparse_hits]

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._cfg.hybrid.rrf_k)

        # Build a lookup from hit_id -> SearchHit, preferring dense (richer metadata)
        hit_map: dict[str, SearchHit] = {}
        for hit in reversed(sparse_hits):
            hit_map[self._hit_id(hit)] = hit
        for hit in reversed(dense_hits):
            hit_map[self._hit_id(hit)] = hit

        results: list[SearchHit] = []
        for doc_id, rrf_score in fused[:top_k]:
            if doc_id not in hit_map:
                continue
            hit = hit_map[doc_id]
            results.append(
                SearchHit(
                    path=hit.path,
                    chunk_index=hit.chunk_index,
                    score=round(rrf_score, 6),
                    preview=hit.preview,
                    file_type=hit.file_type,
                    mtime=hit.mtime,
                    source_root=hit.source_root,
                    start_char=hit.start_char,
                    end_char=hit.end_char,
                )
            )
        return results

    def _fuse_all(self, hit_lists: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
        """Fuse multiple candidate lists via RRF (multi-query expansion case)."""
        id_lists = [[self._hit_id(h) for h in hits] for hits in hit_lists]
        fused = reciprocal_rank_fusion(id_lists, k=self._cfg.hybrid.rrf_k)

        # Build a global hit_map; first-seen wins (earlier lists have priority)
        hit_map: dict[str, SearchHit] = {}
        for hits in reversed(hit_lists):
            for hit in hits:
                hit_map[self._hit_id(hit)] = hit

        results: list[SearchHit] = []
        for doc_id, rrf_score in fused[:top_k]:
            if doc_id not in hit_map:
                continue
            hit = hit_map[doc_id]
            results.append(
                SearchHit(
                    path=hit.path,
                    chunk_index=hit.chunk_index,
                    score=round(rrf_score, 6),
                    preview=hit.preview,
                    file_type=hit.file_type,
                    mtime=hit.mtime,
                    source_root=hit.source_root,
                    start_char=hit.start_char,
                    end_char=hit.end_char,
                )
            )
        return results

    @staticmethod
    def _hit_id(hit: SearchHit) -> str:
        return f"{hit.path}:{hit.chunk_index}"
