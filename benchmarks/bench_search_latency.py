"""Search latency benchmark.

Measures p50 / p95 / p99 end-to-end search latency across hybrid, dense,
and sparse modes.  A synthetic corpus is indexed once; then each query is
run *n_queries* times and latencies are collected.

All real components are used (ChromaDB, BM25, RRF) with a mock embedding
provider so the benchmark measures the search stack, not inference speed.

Usage
-----
    uv run python benchmarks/bench_search_latency.py
    uv run python benchmarks/bench_search_latency.py --n-docs 500 --n-queries 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorymesh.core.models import MemoryMeshConfig, StorageConfig
from memorymesh.indexer.file_indexer import FileIndexer
from memorymesh.search.engine import SearchEngine
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.metadata_store import MetadataStore
from memorymesh.storage.vector_store import ChromaVectorStore

_DIM = 384

_SAMPLE_QUERIES = [
    "hybrid search engine architecture",
    "how does BM25 work with embeddings",
    "file watcher debounce configuration",
    "python function that returns string",
    "machine learning neural networks",
    "calendar event start date location",
    "email archive inbox search",
    "obsidian wikilink frontmatter tags",
    "reconciliation startup crash recovery",
    "chunking strategy markdown headings",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_provider(dim: int = _DIM) -> MagicMock:
    rng = random.Random(99)

    def _rand_unit(texts: list[str]) -> list[list[float]]:
        results = []
        for _ in texts:
            v = [rng.gauss(0, 1) for _ in range(dim)]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            results.append([x / norm for x in v])
        return results

    provider = MagicMock()
    provider.model_id = f"mock-{dim}d"
    provider.embed_documents.side_effect = _rand_unit
    provider.embed_query.side_effect = lambda t: _rand_unit([t])[0]
    return provider


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _build_corpus_and_stack(tmp: Path, n_docs: int) -> tuple[SearchEngine, BM25Index]:
    """Index *n_docs* synthetic files and return a ready SearchEngine."""
    words = [
        "memory",
        "mesh",
        "index",
        "search",
        "query",
        "embedding",
        "chunk",
        "parse",
        "document",
        "file",
        "hybrid",
        "dense",
        "sparse",
        "vector",
        "bm25",
        "reranker",
        "ollama",
        "config",
        "yaml",
        "python",
        "rust",
        "go",
        "typescript",
        "markdown",
        "pdf",
        "wikilink",
        "calendar",
        "email",
        "agent",
    ]
    rng = random.Random(7)

    corpus = tmp / "corpus"
    corpus.mkdir()
    for i in range(n_docs):
        p = corpus / f"doc_{i:05d}.txt"
        content = " ".join(rng.choice(words) for _ in range(rng.randint(30, 150)))
        p.write_text(content, encoding="utf-8")

    cfg = MemoryMeshConfig(storage=StorageConfig(base_dir=tmp / "state"))
    provider = _make_provider()
    vector_store = ChromaVectorStore(db_path=tmp / "chroma", embedding_model_id=provider.model_id)
    metadata_store = MetadataStore(tmp / "meta.sqlite3")
    bm25 = BM25Index(tmp / "bm25.pkl")

    indexer = FileIndexer(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
    )
    indexer.index_directory(corpus, source_name="bench")

    engine = SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
    )
    return engine, bm25


def _bench_mode(
    engine: SearchEngine,
    mode: str,
    queries: list[str],
    n_queries: int,
    top_k: int,
) -> dict:
    """Run *n_queries* search calls and return latency stats (ms)."""
    rng = random.Random(0)
    latencies: list[float] = []

    for _ in range(n_queries):
        q = rng.choice(queries)
        t0 = time.perf_counter()
        engine.search(q, top_k=top_k, mode=mode)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": mode,
        "n_queries": n_queries,
        "mean_ms": round(sum(latencies) / len(latencies), 2),
        "p50_ms": round(_percentile(latencies, 50), 2),
        "p95_ms": round(_percentile(latencies, 95), 2),
        "p99_ms": round(_percentile(latencies, 99), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def _print_row(stats: dict) -> None:
    print(
        f"  {stats['mode']:<8}  "
        f"mean={stats['mean_ms']:>6.1f}ms  "
        f"p50={stats['p50_ms']:>6.1f}ms  "
        f"p95={stats['p95_ms']:>6.1f}ms  "
        f"p99={stats['p99_ms']:>6.1f}ms  "
        f"min={stats['min_ms']:>5.1f}ms  "
        f"max={stats['max_ms']:>6.1f}ms"
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="MemoryMesh search latency benchmark.")
    parser.add_argument("--n-docs", type=int, default=200, help="Corpus size.")
    parser.add_argument("--n-queries", type=int, default=50, help="Queries per mode.")
    parser.add_argument("--top-k", type=int, default=10, help="Results per query.")
    parser.add_argument("--modes", nargs="+", default=["hybrid", "dense", "sparse"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with TemporaryDirectory(prefix="mm_lat_") as tmpdir:
        tmp = Path(tmpdir)
        print(f"Building corpus ({args.n_docs} docs) … ", end="", flush=True)
        engine, _bm25 = _build_corpus_and_stack(tmp, args.n_docs)
        print("done\n")

        print("MemoryMesh — Search Latency Benchmark")
        print(f"  corpus={args.n_docs} docs  n_queries={args.n_queries}  top_k={args.top_k}\n")

        all_stats: list[dict] = []
        for mode in args.modes:
            stats = _bench_mode(engine, mode, _SAMPLE_QUERIES, args.n_queries, args.top_k)
            _print_row(stats)
            all_stats.append(stats)

        if args.output:
            args.output.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
