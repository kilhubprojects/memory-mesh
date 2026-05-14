"""Indexing throughput benchmark.

Measures how many files/chunks MemoryMesh can index per second using a
synthetic corpus of configurable size.  All real components are used
(ChromaDB, BM25, MetadataStore) except the embedding provider, which is
replaced with a deterministic mock to decouple indexing speed from GPU/CPU
model inference speed.

Usage
-----
    uv run python benchmarks/bench_indexing.py
    uv run python benchmarks/bench_indexing.py --n-files 500 --workers 4

Results are printed as a human-readable table and optionally saved to a
JSON file for tracking over time.
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

# Ensure src/ is in path when run without uv install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memorymesh.core.models import MemoryMeshConfig, StorageConfig
from memorymesh.indexer.file_indexer import FileIndexer
from memorymesh.storage.bm25_index import BM25Index
from memorymesh.storage.metadata_store import MetadataStore
from memorymesh.storage.vector_store import ChromaVectorStore

_DIM = 384  # matches all-MiniLM-L6-v2 output dimension


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_provider(dim: int = _DIM) -> MagicMock:
    """Mock embedding provider that returns random unit-normalised vectors."""
    rng = random.Random(42)

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


def _generate_corpus(directory: Path, n_files: int) -> list[Path]:
    """Write *n_files* synthetic text files to *directory*."""
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
        "docx",
        "wikilink",
        "calendar",
        "email",
        "archive",
        "source",
        "watcher",
        "debounce",
        "daemon",
        "transport",
        "stdio",
        "http",
        "agent",
    ]

    rng = random.Random(0)
    paths: list[Path] = []
    extensions = [".txt", ".md", ".py"]

    for i in range(n_files):
        ext = extensions[i % len(extensions)]
        path = directory / f"file_{i:05d}{ext}"
        n_words = rng.randint(50, 300)
        content = " ".join(rng.choice(words) for _ in range(n_words))
        if ext == ".md":
            content = f"# Document {i}\n\n{content}\n"
        elif ext == ".py":
            content = (
                f'"""Module {i}."""\n\n'
                f"def func_{i}() -> str:\n"
                f'    """Return a string."""\n'
                f'    return "{content[:40]}"\n'
            )
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    return paths


# ── Benchmark ─────────────────────────────────────────────────────────────────


def run_bench(
    n_files: int = 200,
    workers: int = 1,
    warmup: int = 5,
) -> dict:
    """Run the indexing benchmark and return a results dict.

    Args:
        n_files: Number of synthetic files to index.
        workers: Not currently used by FileIndexer (it's single-threaded per
            call); included for future parallel indexer support.
        warmup: Number of files to index before timing starts.

    Returns:
        Dict with timing and throughput metrics.
    """
    with TemporaryDirectory(prefix="mm_bench_") as tmpdir:
        tmp = Path(tmpdir)
        corpus_dir = tmp / "corpus"
        corpus_dir.mkdir()
        state_dir = tmp / "state"

        print(f"Generating {n_files} synthetic files … ", end="", flush=True)
        paths = _generate_corpus(corpus_dir, n_files)
        total_bytes = sum(p.stat().st_size for p in paths)
        print(f"done ({total_bytes / 1024:.1f} KB total)")

        cfg = MemoryMeshConfig(storage=StorageConfig(base_dir=state_dir))
        provider = _make_provider()
        vector_store = ChromaVectorStore(
            db_path=tmp / "chroma", embedding_model_id=provider.model_id
        )
        metadata_store = MetadataStore(tmp / "metadata.sqlite3")
        bm25 = BM25Index(tmp / "bm25.pkl")

        indexer = FileIndexer(
            config=cfg,
            vector_store=vector_store,
            metadata_store=metadata_store,
            embedding_provider=provider,
            bm25_index=bm25,
        )

        # Warmup
        if warmup > 0:
            print(f"Warming up ({warmup} files) … ", end="", flush=True)
            for p in paths[:warmup]:
                indexer.index_file(p, source_name="bench")
            print("done")

        # Timed run (remaining files)
        timed_paths = paths[warmup:]
        n_timed = len(timed_paths)
        print(f"Timing {n_timed} files … ", end="", flush=True)

        t0 = time.perf_counter()
        results = [indexer.index_file(p, source_name="bench") for p in timed_paths]
        elapsed = time.perf_counter() - t0

        n_indexed = sum(1 for r in results if r.status == "indexed")
        n_chunks = sum(r.n_chunks for r in results if r.status == "indexed")
        timed_bytes = sum(
            p.stat().st_size
            for p, r in zip(timed_paths, results, strict=False)
            if r.status == "indexed"
        )

        files_per_sec = n_indexed / elapsed if elapsed > 0 else 0
        chunks_per_sec = n_chunks / elapsed if elapsed > 0 else 0
        mb_per_sec = (timed_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0

        print("done\n")

        row = {
            "n_files_total": n_files,
            "n_files_timed": n_indexed,
            "n_chunks": n_chunks,
            "elapsed_s": round(elapsed, 3),
            "files_per_sec": round(files_per_sec, 1),
            "chunks_per_sec": round(chunks_per_sec, 1),
            "mb_per_sec": round(mb_per_sec, 3),
            "workers": workers,
        }

        return row


def _print_table(row: dict) -> None:
    print("┌─────────────────────────────────────┐")
    print("│   MemoryMesh — Indexing Benchmark    │")
    print("├─────────────────────────────────────┤")
    print(f"│  Files indexed   : {row['n_files_timed']:>6}            │")
    print(f"│  Chunks produced : {row['n_chunks']:>6}            │")
    print(f"│  Elapsed         : {row['elapsed_s']:>6.2f} s          │")
    print(f"│  Files/sec       : {row['files_per_sec']:>6.1f}            │")
    print(f"│  Chunks/sec      : {row['chunks_per_sec']:>6.1f}            │")
    print(f"│  MB/sec          : {row['mb_per_sec']:>6.3f}            │")
    print("└─────────────────────────────────────┘")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="MemoryMesh indexing throughput benchmark.")
    parser.add_argument("--n-files", type=int, default=200, help="Number of files to index.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup files (not timed).")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads (future use).")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    args = parser.parse_args()

    row = run_bench(n_files=args.n_files, workers=args.workers, warmup=args.warmup)
    _print_table(row)

    if args.output:
        args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
