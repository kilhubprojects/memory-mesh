"""Wave 0 eval: compare baseline vs reranker+expansion on field-test queries.

Ground truth is file-level relevance labels derived from the Wave 0 field test.
Run from the repo root:

    uv run python scripts/eval_wave0.py

Requires the live index at ~/.memorymesh/ (produced by `memorymesh start`).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ── Ground truth ──────────────────────────────────────────────────────────────
# relevance[query_id][basename] → grade (3=primary, 2=relevant, 1=marginal, 0=noise)

QUERIES: list[tuple[str, dict[str, int]]] = [
    (
        "how does the BM25 index work",
        {
            "bm25_index.py": 3,
            "engine.py": 2,
            "ARCHITECTURE.md": 2,
            "rank_fusion.py": 1,
        },
    ),
    (
        "reciprocal rank fusion implementation",
        {
            "rank_fusion.py": 3,
            "engine.py": 2,
            "bm25_index.py": 1,
        },
    ),
    (
        "what happens when a file is modified — watcher debounce flow until chunk is stored",
        {
            "watcher.py": 3,
            "file_indexer.py": 3,
            "ARCHITECTURE.md": 2,
            "bm25_index.py": 1,
        },
    ),
]

TOP_K = 10
IDEAL_DCG_CACHE: dict[tuple[int, ...], float] = {}


def dcg(grades: list[int]) -> float:
    """Compute DCG for a ranked list of grades."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg(hits_grades: list[int], all_grades: dict[str, int]) -> float:
    """Compute NDCG@k given hit grades and all relevance labels."""
    ideal = sorted(all_grades.values(), reverse=True)[:TOP_K]
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 1.0
    return dcg(hits_grades) / ideal_score


def grade_hits(hits: list[object], relevance: dict[str, int]) -> list[int]:
    """Map a list of SearchHit to relevance grades by basename.

    Each file is counted at most once — subsequent chunks from the same file
    get grade 0 to avoid inflating DCG.
    """
    grades = []
    seen: set[str] = set()
    for hit in hits[:TOP_K]:
        basename = Path(hit.path).name  # type: ignore[attr-defined]
        if basename in seen:
            grades.append(0)
        else:
            seen.add(basename)
            grades.append(relevance.get(basename, 0))
    return grades


def run_search(engine: object, query: str) -> list[object]:
    """Run search and return hits."""
    response = engine.search(query, top_k=TOP_K)  # type: ignore[attr-defined]
    return response.hits


def build_engine(
    cfg: object,
    reranker_enabled: bool,
    expansion_enabled: bool,
) -> object:
    """Bootstrap a SearchEngine against the live index."""
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from memorymesh.search.engine import SearchEngine
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.vector_store import ChromaVectorStore

    cfg = cfg  # type: ignore[assignment]

    provider = SentenceTransformersProvider(config=cfg.embeddings)  # type: ignore[attr-defined]
    vector_store = ChromaVectorStore(
        db_path=cfg.storage.vector_db_path,  # type: ignore[attr-defined]
        embedding_model_id=provider.model_id,
    )
    bm25 = BM25Index(cfg.storage.bm25_pickle_path)  # type: ignore[attr-defined]
    bm25.load()

    from memorymesh.core.models import (
        QueryExpansionConfig,
        SearchConfig,
        SearchHybridConfig,
        SearchRerankerConfig,
    )

    search_cfg = SearchConfig(
        default_top_k=TOP_K,
        hybrid=SearchHybridConfig(enabled=True, rrf_k=60, over_fetch_factor=3),
        reranker=SearchRerankerConfig(
            enabled=reranker_enabled,
            model="BAAI/bge-reranker-v2-m3",
            top_k_before_rerank=35,
        ),
        query_expansion=QueryExpansionConfig(
            enabled=expansion_enabled,
            use_hyde=False,
            n_lexical_variants=1,
        ),
    )

    return SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
        config=search_cfg,
        ollama_config=cfg.ollama,  # type: ignore[attr-defined]
    )


def print_hits(hits: list[object], relevance: dict[str, int]) -> None:
    for i, hit in enumerate(hits[:TOP_K], 1):
        basename = Path(hit.path).name  # type: ignore[attr-defined]
        grade = relevance.get(basename, 0)
        mark = ["  ", "+ ", "* ", "**"][min(grade, 3)]
        score = hit.score  # type: ignore[attr-defined]
        print(f"  {i:2}. {mark}{basename:<35} score={score:.4f}  grade={grade}")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memorymesh.config import load_config
    from memorymesh.observability.logging import setup_logging

    setup_logging(level="WARNING")  # suppress INFO spam during eval

    cfg = load_config()

    print("=" * 70)
    print(f"Wave 0 Eval — MemoryMesh  (top_k={TOP_K})")
    print("=" * 70)

    configs = [
        ("Baseline (RRF only)", False, False),
        ("+ Reranker + Query Expansion", True, True),
    ]

    results: dict[str, list[float]] = {label: [] for label, _, _ in configs}

    for label, reranker_on, expansion_on in configs:
        print(f"\n{'-' * 70}")
        print(f"  Config: {label}")
        print(f"{'-' * 70}")
        engine = build_engine(cfg, reranker_on, expansion_on)

        for q_idx, (query, relevance) in enumerate(QUERIES):
            hits = run_search(engine, query)
            grades = grade_hits(hits, relevance)
            score = ndcg(grades, relevance)
            results[label].append(score)

            print(f"\n  Q{q_idx + 1}: {query!r}")
            print(f"  NDCG@{TOP_K} = {score:.4f}")
            print_hits(hits, relevance)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  NDCG@10 COMPARISON SUMMARY")
    print(f"{'-' * 70}")
    print(f"  {'Config':<38} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Mean':>6}")
    print(f"  {'-' * 60}")

    baseline_scores = None
    for label, scores in results.items():
        mean = sum(scores) / len(scores)
        "  ".join(f"{s:.3f}" for s in scores)
        print(f"  {label:<38} {scores[0]:>6.3f} {scores[1]:>6.3f} {scores[2]:>6.3f} {mean:>6.3f}")
        if baseline_scores is None:
            baseline_scores = scores

    if baseline_scores and len(results) == 2:
        new_scores = list(results.values())[1]
        deltas = [n - b for n, b in zip(new_scores, baseline_scores, strict=False)]
        mean_delta = sum(deltas) / len(deltas)
        "  ".join(f"{d:+.3f}" for d in deltas)
        d0, d1, d2 = deltas[0], deltas[1], deltas[2]
        label = "Delta (improvement)"
        print(f"  {label:<38} {d0:>+6.3f} {d1:>+6.3f} {d2:>+6.3f} {mean_delta:>+6.3f}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
