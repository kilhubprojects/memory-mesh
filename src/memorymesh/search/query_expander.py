"""Query expansion for hybrid search.

Two expansion modes are available and can be combined:

**Mode A - Lexical expansion (always available):**
    Rule-based stemming and synonym substitution over a small technical
    vocabulary.  Generates alternative phrasings of the query without any
    external model.  Example: ``"indexing flow"`` -> ``["index flow",
    "file indexer flow"]``.

**Mode B - HyDE (Hypothetical Document Embeddings, opt-in):**
    Generates a hypothetical answer document via a local Ollama model, then
    embeds it and uses the result as an additional dense query vector.  This
    improves recall for vague or abstract queries whose surface form differs
    from how relevant documents are written.  Requires ``ollama.enabled: true``
    in the config.

Both modes are opt-in via ``search.query_expansion`` config.  All failures
fall back silently - the daemon never crashes due to an expansion error.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import OllamaConfig, QueryExpansionConfig, SearchHit
    from memorymesh.embeddings.base import EmbeddingProvider


# Maps a root/variant term to a list of alternative terms to substitute.
# Keys should be lowercase; values are listed most-specific first.
_SYNONYMS: dict[str, list[str]] = {
    "index": ["indexing", "indexed", "file indexer", "indexer"],
    "indexing": ["index", "indexed", "file indexer"],
    "indexed": ["index", "indexing"],
    "indexer": ["file indexer", "indexing pipeline"],
    "search": ["query", "retrieve", "lookup", "find"],
    "query": ["search", "retrieve", "find"],
    "retrieve": ["search", "fetch", "query"],
    "chunk": ["segment", "passage", "text chunk"],
    "chunking": ["segmentation", "splitting", "text splitting"],
    "embed": ["embedding", "vector", "encode"],
    "embedding": ["vector", "encode", "dense representation"],
    "store": ["storage", "persist", "save"],
    "storage": ["store", "persist", "database"],
    "watch": ["monitor", "detect", "observe"],
    "watcher": ["file watcher", "monitor", "observer"],
    "parse": ["extract", "read", "process"],
    "parsing": ["extraction", "reading", "processing"],
    "file": ["document", "path", "source file"],
    "document": ["file", "text", "content"],
    "delete": ["remove", "removal", "deletion"],
    "hash": ["sha256", "checksum", "fingerprint"],
    "dedup": ["deduplication", "duplicate detection", "hash check"],
    "fuse": ["merge", "combine", "blend"],
    "fusion": ["merging", "combination", "rank fusion"],
    "dense": ["vector search", "semantic search", "embedding search"],
    "sparse": ["keyword search", "bm25", "lexical search"],
    "hybrid": ["dense and sparse", "combined search", "fusion search"],
    "flow": ["pipeline", "process", "workflow"],
    "boot": ["startup", "initialise", "start"],
    "crash": ["fail", "error", "exception"],
    "daemon": ["server", "background process", "service"],
}

# Simple suffix rules for stemming: (suffix_to_strip, replacement).
# Applied in order; first match wins.
_STEM_RULES: list[tuple[str, str]] = [
    ("tion", ""),
    ("ing", ""),
    ("tion", ""),
    ("ed", ""),
    ("er", ""),
    ("ion", ""),
    ("s", ""),
]


def _stem(word: str) -> str:
    """Return a rough stem for ``word`` using suffix stripping."""
    w = word.lower()
    for suffix, replacement in _STEM_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + replacement
    return w


class QueryExpander:
    """Generates alternative query formulations to improve search recall.

    Args:
        config: Query expansion settings.
        embedding_provider: Used to embed hypothetical documents (HyDE mode).
        ollama_config: Ollama connection settings.  Required only when
            ``config.use_hyde`` is ``True``.
    """

    def __init__(
        self,
        config: QueryExpansionConfig,
        embedding_provider: EmbeddingProvider,
        ollama_config: OllamaConfig | None = None,
    ) -> None:
        self._config = config
        self._provider = embedding_provider
        self._ollama = ollama_config

    def expand_queries(self, query: str) -> list[str]:
        """Return a list of query strings, starting with the original.

        Args:
            query: The original user query.

        Returns:
            A deduplicated list ``[original, variant_1, ..., variant_n]``.
            Always contains at least the original query.
        """
        variants: list[str] = [query]

        if self._config.n_lexical_variants > 0:
            lex = self._lexical_variants(query, self._config.n_lexical_variants)
            for v in lex:
                if v not in variants:
                    variants.append(v)

        logger.debug(f"QueryExpander: lexical variants={len(variants) - 1}")
        return variants

    def expand_vectors(self, query: str) -> list[list[float]]:
        """Return additional query vectors beyond the original embedding.

        Currently produces one hypothetical-document vector when HyDE is
        enabled.  Returns an empty list when HyDE is disabled or Ollama is
        unavailable.

        Args:
            query: The original user query.

        Returns:
            A list of additional embedding vectors (may be empty).
        """
        if not (self._config.use_hyde and self._ollama and self._ollama.enabled):
            return []

        hypo = self._generate_hypothetical(query)
        if hypo is None:
            return []

        try:
            vec = self._provider.embed_query(hypo)
            logger.debug("QueryExpander: HyDE vector generated")
            return [vec]
        except Exception as exc:
            logger.warning(f"QueryExpander: HyDE embedding failed: {exc}")
            return []

    def _lexical_variants(self, query: str, n: int) -> list[str]:
        """Generate up to ``n`` lexical variants of ``query``."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        variants: list[str] = []

        for i, token in enumerate(tokens):
            candidates = _SYNONYMS.get(token, [])
            if not candidates:
                stem = _stem(token)
                candidates = _SYNONYMS.get(stem, [])

            for candidate in candidates[:2]:
                new_tokens = [*tokens[:i], candidate, *tokens[i + 1 :]]
                variant = " ".join(new_tokens)
                if variant != query.lower() and variant not in variants:
                    variants.append(variant)
                if len(variants) >= n:
                    return variants

        return variants

    def _generate_hypothetical(self, query: str) -> str | None:
        """Generate a hypothetical passage that would answer ``query`` via Ollama."""
        assert self._ollama is not None  # guarded by caller

        prompt = (
            f"Write a short technical passage (2-3 sentences) that directly "
            f"answers the following question. Write only the passage, no preamble:\n\n"
            f"{query}"
        )
        payload = json.dumps(
            {
                "model": self._ollama.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 120, "temperature": 0.1},
            }
        ).encode()

        url = f"{self._ollama.base_url.rstrip('/')}/api/generate"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._ollama.timeout_s) as resp:
                body = json.loads(resp.read())
                text = body.get("response", "").strip()
                if not text:
                    logger.warning("QueryExpander: Ollama returned empty HyDE response")
                    return None
                return text
        except urllib.error.URLError as exc:
            logger.warning(f"QueryExpander: Ollama unreachable for HyDE: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"QueryExpander: HyDE generation failed: {exc}")
            return None


def merge_hit_lists(
    hit_lists: list[list[SearchHit]],
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Merge multiple ranked hit lists via RRF, preserving hit objects.

    Args:
        hit_lists: Each inner list is a ranked sequence of
            :class:`~memorymesh.core.models.SearchHit` objects.
        rrf_k: RRF smoothing constant.

    Returns:
        Deduplicated list of hits ordered by descending fused RRF score.
        The ``score`` field of each hit is replaced with the fused RRF score.
    """
    from memorymesh.core.models import SearchHit
    from memorymesh.search.rank_fusion import reciprocal_rank_fusion

    def _hit_id(h: object) -> str:
        assert isinstance(h, SearchHit)
        return f"{h.path}:{h.chunk_index}"

    rankings = [[_hit_id(h) for h in lst] for lst in hit_lists]
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)

    # Build lookup: prefer first list (original query) for metadata
    hit_map: dict[str, SearchHit] = {}
    for lst in reversed(hit_lists):
        for h in lst:
            assert isinstance(h, SearchHit)
            hit_map[_hit_id(h)] = h

    result: list[SearchHit] = []
    for doc_id, rrf_score in fused:
        if doc_id not in hit_map:
            continue
        h = hit_map[doc_id]
        result.append(
            SearchHit(
                path=h.path,
                chunk_index=h.chunk_index,
                score=round(rrf_score, 6),
                preview=h.preview,
                file_type=h.file_type,
                mtime=h.mtime,
                source_root=h.source_root,
                start_char=h.start_char,
                end_char=h.end_char,
                extended_preview=h.extended_preview,
            )
        )
    return result
