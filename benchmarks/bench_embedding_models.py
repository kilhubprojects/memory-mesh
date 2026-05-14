"""Embedding model comparison benchmark.

Measures indexing throughput and search-quality proxy metrics for
different sentence-transformers models.  Useful for choosing the right
embedding model for a given hardware / quality trade-off.

Quality proxy: for each model, the benchmark indexes a small labelled
corpus, runs 5 fixed queries, and checks whether the top-1 result
matches the expected document.  This is a coarse recall@1 estimate, not
a full NDCG evaluation — for that, use the eval scripts.

Hardware note: the actual model inference runs here (unlike the other
benchmarks which use mock providers).  If you have a GPU/MPS device,
set ``--device cuda`` or ``--device mps`` to get meaningful speed numbers.
Without a GPU, MPS, or large RAM, stick to the two smallest models.

Usage
-----
    uv run python benchmarks/bench_embedding_models.py
    uv run python benchmarks/bench_embedding_models.py --models all-MiniLM-L6-v2
    uv run python benchmarks/bench_embedding_models.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Model catalogue ───────────────────────────────────────────────────────────

_MODELS: list[dict] = [
    {
        "id": "all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Default — fast, good English quality",
        "multilingual": False,
    },
    {
        "id": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "description": "Multilingual (PT, EN, ES, ...) — same dim, ~2x slower",
        "multilingual": True,
    },
    {
        "id": "all-mpnet-base-v2",
        "dim": 768,
        "description": "English, higher quality, ~4x slower, 2x larger dim",
        "multilingual": False,
    },
]

# ── Labelled test corpus ──────────────────────────────────────────────────────

_CORPUS: list[tuple[str, str]] = [
    ("python_tutorial.txt", "Python is a high-level programming language."),
    ("ml_notes.txt", "Neural networks and deep learning use gradient descent."),
    ("architecture.txt", "MemoryMesh uses hybrid search with BM25 and ChromaDB."),
    ("calendar.txt", "Meeting scheduled for next Monday to discuss deployment."),
    ("email.txt", "Please review the pull request before the end of the sprint."),
    ("obsidian.txt", "# Research\n\nWikilinks connect ideas across notes."),
    ("search_engine.txt", "Reciprocal Rank Fusion combines dense and sparse results."),
    ("config.txt", "Set debounce_ms to 1500 and worker_concurrency to 2 in config.yaml."),
]

_QUERIES: list[tuple[str, str]] = [
    ("programming language", "python_tutorial.txt"),
    ("hybrid search bm25", "search_engine.txt"),
    ("meeting scheduled deployment", "calendar.txt"),
    ("pull request sprint review", "email.txt"),
    ("debounce watcher config", "config.txt"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 0 else 0.0


def _bench_model(
    model_id: str,
    device: str,
    corpus: list[tuple[str, str]],
    queries: list[tuple[str, str]],
) -> dict:
    """Load *model_id*, embed corpus + queries, measure speed and recall@1."""
    from memorymesh.core.models import EmbeddingsConfig
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    cfg = EmbeddingsConfig(model=model_id, device=device, batch_size=32, normalize=True)

    print(f"  Loading {model_id} … ", end="", flush=True)
    t_load0 = time.perf_counter()
    provider = SentenceTransformersProvider(config=cfg)
    # Force model load by embedding a dummy text
    provider.embed_query("warmup")
    t_load = time.perf_counter() - t_load0
    print(f"loaded in {t_load:.1f}s")

    # Embed corpus
    texts = [text for _, text in corpus]
    filenames = [fn for fn, _ in corpus]
    t_emb0 = time.perf_counter()
    doc_vecs = provider.embed_documents(texts)
    t_emb = time.perf_counter() - t_emb0

    # Embed queries and compute recall@1
    hits = 0
    for query_text, expected_fn in queries:
        q_vec = provider.embed_query(query_text)
        scores = [_cosine(q_vec, dv) for dv in doc_vecs]
        top_fn = filenames[scores.index(max(scores))]
        if top_fn == expected_fn:
            hits += 1

    recall_at_1 = hits / len(queries) if queries else 0.0
    docs_per_sec = len(texts) / t_emb if t_emb > 0 else 0.0

    return {
        "model": model_id,
        "load_s": round(t_load, 2),
        "embed_corpus_s": round(t_emb, 3),
        "docs_per_sec": round(docs_per_sec, 1),
        "recall_at_1": round(recall_at_1, 2),
        "n_queries": len(queries),
        "n_docs": len(texts),
        "device": device,
    }


def _print_results(rows: list[dict]) -> None:
    print("\n  MemoryMesh — Embedding Model Benchmark")
    print(f"  {'Model':<45} {'Load':>6}  {'Docs/s':>7}  {'Recall@1':>9}  {'Device':<8}")
    print("  " + "-" * 80)
    for r in rows:
        m = r["model"]
        if len(m) > 44:
            m = m[:41] + "…"
        print(
            f"  {m:<45} {r['load_s']:>5.1f}s  {r['docs_per_sec']:>7.1f}  "
            f"{r['recall_at_1']:>8.0%}  {r['device']:<8}"
        )


def main() -> None:
    """Entry point."""
    available_ids = [m["id"] for m in _MODELS]
    parser = argparse.ArgumentParser(description="MemoryMesh embedding model benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=available_ids,
        choices=available_ids,
        metavar="MODEL_ID",
        help="Model IDs to benchmark. Defaults to all three.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' selects cuda > mps > cpu.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with TemporaryDirectory(prefix="mm_emb_") as tmpdir:
        # Write corpus to temp dir so providers can optionally read from disk
        tmp = Path(tmpdir)
        for fn, text in _CORPUS:
            (tmp / fn).write_text(text, encoding="utf-8")

        rows: list[dict] = []
        for model_id in args.models:
            meta = next((m for m in _MODELS if m["id"] == model_id), {})
            print(f"\n[{model_id}]  {meta.get('description', '')}")
            try:
                row = _bench_model(model_id, args.device, _CORPUS, _QUERIES)
                rows.append(row)
            except Exception as exc:
                print(f"  ERROR: {exc}")

    _print_results(rows)

    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
