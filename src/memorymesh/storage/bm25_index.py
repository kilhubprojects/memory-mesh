"""BM25 sparse index backed by rank-bm25.

Persists the index as a pickle file at ``storage.bm25_pickle_path``.  The
index is rebuilt from scratch whenever the caller detects drift between SQLite
and the on-disk pickle (see :class:`~memorymesh.indexer.file_indexer.FileIndexer`).

Tokenisation: lowercase -> ``\\w+`` regex -> remove one-character tokens and a
small PT+EN stopword set.  Keeps the approach simple and dependency-free.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import sys
from pathlib import Path

from loguru import logger

from memorymesh.core.models import SearchHit

_STOPWORDS: frozenset[str] = frozenset(
    {
        # EN
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "so",
        # PT
        "o",
        "os",
        "um",
        "uma",
        "de",
        "da",
        "das",
        "dos",
        "em",
        "na",
        "nas",
        "nos",
        "para",
        "por",
        "com",
        "que",
        "se",
        "n?o",
        "mais",
        "mas",
        "ou",
        "e",
        "?",
        "ao",
        "aos",
    }
)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, remove stopwords and short tokens."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 1 and tok not in _STOPWORDS
    ]


class BM25Index:
    """In-memory BM25 index with pickle persistence.

    Args:
        pickle_path: Where to save / load the serialised index.  The parent
            directory is created on first save.
    """

    def __init__(self, pickle_path: Path) -> None:
        self._pickle_path = pickle_path
        # corpus: list of token lists, parallel to _paths and _chunk_ids
        self._corpus: list[list[str]] = []
        self._paths: list[str] = []  # file path for each document
        self._chunk_ids: list[str] = []  # "<path>:<idx>" for each document
        self._previews: list[str] = []  # first 200 chars of each doc
        self._bm25: object | None = None  # rank_bm25.BM25Okapi, built lazily

    def rebuild(self, docs: list[tuple[str, str, str]]) -> None:
        """Rebuild the index from scratch.

        Args:
            docs: List of ``(chunk_id, path, text)`` tuples.  Replaces the
                entire existing index.
        """
        self._corpus = []
        self._paths = []
        self._chunk_ids = []
        self._previews = []
        self._bm25 = None

        for chunk_id, path, text in docs:
            self._corpus.append(_tokenize(text))
            self._chunk_ids.append(chunk_id)
            self._paths.append(path)
            self._previews.append(text[:200])

        if self._corpus:
            self._bm25 = self._build_bm25(self._corpus)

        logger.info(f"BM25Index: rebuilt with {len(self._corpus)} document(s)")
        self.save()

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the *top_k* best BM25 matches for *query*.

        Args:
            query: Raw query string.
            top_k: Maximum number of hits to return.

        Returns:
            List of :class:`~memorymesh.core.models.SearchHit`, best-score first.
            Empty list if the index is empty or the query has no matching tokens.
        """
        if self._bm25 is None or not self._corpus:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores: list[float] = self._bm25.get_scores(tokens)  # type: ignore[attr-defined]

        # rank_bm25 0.2.x can return negative scores for small corpora.
        # Shift scores so the minimum is 0, then normalise by the range.
        float_scores = [float(s) for s in scores]
        min_score = min(float_scores)
        shifted = [s - min_score for s in float_scores]
        max_shifted = max(shifted)
        if max_shifted <= 0.0:
            return []

        ranked = sorted(enumerate(shifted), key=lambda x: x[1], reverse=True)
        hits: list[SearchHit] = []
        for idx, raw_score in ranked[:top_k]:
            if raw_score <= 0.0:
                break
            normalised = min(raw_score / max_shifted, 1.0)
            chunk_id = self._chunk_ids[idx]
            path_str = self._paths[idx]
            # chunk_id format: "<path>:<chunk_index>"
            parts = chunk_id.rsplit(":", 1)
            chunk_index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
            hits.append(
                SearchHit(
                    path=path_str,
                    chunk_index=chunk_index,
                    score=round(normalised, 6),
                    preview=self._previews[idx],
                    file_type=Path(path_str).suffix.lower(),
                    mtime=0.0,
                    source_root="",
                )
            )
        return hits

    def add_chunks(self, chunks: list[tuple[str, str, str]]) -> None:
        """Append new chunks to the index without a full rebuild.

        Args:
            chunks: List of ``(chunk_id, path, text)`` tuples.
        """
        for chunk_id, path, text in chunks:
            self._corpus.append(_tokenize(text))
            self._chunk_ids.append(chunk_id)
            self._paths.append(path)
            self._previews.append(text[:200])
        if self._corpus:
            self._bm25 = self._build_bm25(self._corpus)
        self.save()

    def delete_by_path(self, path: str) -> int:
        """Remove all entries whose path matches *path*.

        Args:
            path: Absolute path string.

        Returns:
            Number of entries removed.
        """
        keep = [i for i, p in enumerate(self._paths) if p != path]
        removed = len(self._paths) - len(keep)
        if removed == 0:
            return 0

        self._corpus = [self._corpus[i] for i in keep]
        self._paths = [self._paths[i] for i in keep]
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._previews = [self._previews[i] for i in keep]
        self._bm25 = self._build_bm25(self._corpus) if self._corpus else None
        logger.debug(f"BM25Index: removed {removed} entry/entries for path={path!r}")
        return removed

    def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._corpus)

    def save(self) -> None:
        """Persist the index to the pickle file.

        Also writes a ``<name>.pkl.sha256`` sidecar so that :meth:`load` can
        detect accidental corruption or tampering before deserialising.
        """
        try:
            self._pickle_path.parent.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    self._pickle_path.parent.chmod(0o700)
                except OSError as exc:
                    logger.warning(
                        f"BM25Index: could not set permissions on {self._pickle_path.parent}: {exc}"
                    )
            tmp = self._pickle_path.with_suffix(".tmp")
            state = {
                "corpus": self._corpus,
                "paths": self._paths,
                "chunk_ids": self._chunk_ids,
                "previews": self._previews,
            }
            with tmp.open("wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._pickle_path)
            # Write SHA-256 sidecar for integrity verification on load.
            sha = hashlib.sha256(self._pickle_path.read_bytes()).hexdigest()
            self._pickle_path.with_suffix(".pkl.sha256").write_text(sha, encoding="utf-8")
            logger.debug(f"BM25Index: saved {len(self._corpus)} docs to {self._pickle_path}")
        except OSError as exc:
            logger.warning(f"BM25Index: failed to save pickle: {exc}")

    def load(self) -> bool:
        """Load the index from the pickle file.

        Returns:
            ``True`` if loaded successfully, ``False`` if the file does not
            exist or is corrupt.
        """
        if not self._pickle_path.exists():
            return False
        # Integrity check: compare SHA-256 against sidecar before deserialising.
        sha_path = self._pickle_path.with_suffix(".pkl.sha256")
        if sha_path.exists():
            try:
                expected = sha_path.read_text(encoding="utf-8").strip()
                actual = hashlib.sha256(self._pickle_path.read_bytes()).hexdigest()
                if actual != expected:
                    logger.warning(
                        f"BM25Index: pickle integrity check failed for {self._pickle_path}; "
                        "rebuilding from scratch"
                    )
                    return False
            except OSError as exc:
                logger.warning(f"BM25Index: could not verify pickle integrity: {exc}")
        try:
            with self._pickle_path.open("rb") as fh:
                state: dict[str, object] = pickle.load(fh)
            self._corpus = state["corpus"]  # type: ignore[assignment]
            self._paths = state["paths"]  # type: ignore[assignment]
            self._chunk_ids = state["chunk_ids"]  # type: ignore[assignment]
            self._previews = state["previews"]  # type: ignore[assignment]
            self._bm25 = self._build_bm25(self._corpus) if self._corpus else None
            logger.info(f"BM25Index: loaded {len(self._corpus)} docs from {self._pickle_path}")
            return True
        except Exception as exc:
            logger.warning(f"BM25Index: failed to load pickle ({exc}); index will be empty")
            return False

    @staticmethod
    def _build_bm25(corpus: list[list[str]]) -> object:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        return BM25Okapi(corpus)
