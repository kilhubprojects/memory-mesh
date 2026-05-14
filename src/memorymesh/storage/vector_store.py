"""Vector store abstraction and ChromaDB implementation.

The :class:`VectorStore` ABC defines the interface the rest of the codebase
depends on.  :class:`ChromaVectorStore` is the MVP implementation backed by
ChromaDB embedded (no external server required).

Boot-time invariant
-------------------
On construction, :class:`ChromaVectorStore` compares the ``embedding_model_id``
stored in the Chroma collection metadata against the one passed by the caller.
A mismatch means the index was built with a different model and **must** be
rebuilt before the daemon can serve queries.  This raises
:class:`~memorymesh.core.exceptions.EmbeddingModelMismatchError` with
actionable instructions.

Privacy invariant: this module never logs chunk text - only IDs, counts, and
operational events.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from memorymesh.core.exceptions import EmbeddingModelMismatchError, StorageError
from memorymesh.core.models import ChunkMetadata, ChunkWithEmbedding, SearchHit

_COLLECTION_NAME = "memorymesh_chunks"


class VectorStore(ABC):
    """Abstract interface for a dense-vector store.

    All methods are synchronous; the async wrapper lives in the search engine.
    """

    @abstractmethod
    def upsert(self, chunks: list[ChunkWithEmbedding]) -> None:
        """Insert or update *chunks* (keyed by ``chunk.id``).

        Args:
            chunks: Chunks with pre-computed embeddings to store.
        """

    @abstractmethod
    def delete_by_path(self, path: str) -> None:
        """Remove all chunks whose ``path`` metadata equals *path*.

        Args:
            path: Absolute path string of the file to remove.
        """

    @abstractmethod
    def search(
        self,
        query_vec: list[float],
        top_k: int,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """Return the *top_k* most similar chunks to *query_vec*.

        Args:
            query_vec: Dense query embedding.
            top_k: Maximum number of results.
            filter_: Optional ChromaDB ``where`` filter (e.g.
                ``{"file_type": ".py"}``).

        Returns:
            List of :class:`~memorymesh.core.models.SearchHit`, highest score first.
        """

    @abstractmethod
    def get_by_path(
        self,
        path: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return raw chunk records (text + metadata) for all chunks at *path*.

        Args:
            path: Absolute path string whose chunks should be fetched.
            limit: Maximum number of chunks to return.

        Returns:
            List of dicts, each with at least ``"text"`` and ``"path"`` keys.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently stored."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The embedding model ID this store was built with."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store using an embedded (in-process) client.

    Args:
        db_path: Directory for the persistent Chroma database.  Pass ``None``
            to use an in-memory ephemeral client (useful in tests).
        collection_name: Name of the ChromaDB collection.
        embedding_model_id: ID of the embedding model in use.  Must match the
            ID stored in the collection or a
            :class:`~memorymesh.core.exceptions.EmbeddingModelMismatchError`
            is raised.
    """

    def __init__(
        self,
        db_path: Path | None,
        collection_name: str = _COLLECTION_NAME,
        embedding_model_id: str = "all-MiniLM-L6-v2",
    ) -> None:
        import chromadb  # deferred - heavy import, only needed when storage is active

        if db_path is None:
            self._client = chromadb.EphemeralClient()
            logger.debug("ChromaVectorStore: using in-memory ephemeral client")
        else:
            db_path.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    db_path.chmod(0o700)
                except OSError as exc:
                    logger.warning(
                        f"ChromaVectorStore: could not set permissions on {db_path}: {exc}"
                    )
            self._client = chromadb.PersistentClient(path=str(db_path))
            logger.debug(f"ChromaVectorStore: persistent client at {db_path}")

        self._embedding_model_id = embedding_model_id
        self._collection = self._get_or_create_collection(collection_name, embedding_model_id)

    def _get_or_create_collection(self, name: str, embedding_model_id: str) -> Any:
        """Return the collection, creating it with correct metadata if new.

        Raises:
            EmbeddingModelMismatchError: If the collection exists but was built
                with a different embedding model.
        """

        collection = self._client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model_id": embedding_model_id,
            },
        )

        stored_model_id: str = collection.metadata.get("embedding_model_id", "")
        if stored_model_id and stored_model_id != embedding_model_id:
            raise EmbeddingModelMismatchError(
                f"Vector store collection '{name}' was built with embedding model "
                f"'{stored_model_id}', but the current config specifies "
                f"'{embedding_model_id}'.  Run 'memorymesh reindex --all' to rebuild."
            )

        logger.debug(
            f"Collection '{name}' ready (model={embedding_model_id!r}, count={collection.count()})"
        )
        return collection

    def upsert(self, chunks: list[ChunkWithEmbedding]) -> None:
        """Insert or update chunks in the collection.

        Args:
            chunks: Non-empty list of chunks with embeddings.
        """
        if not chunks:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, object]] = []

        for chunk in chunks:
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(_chunk_to_metadata(chunk))

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(chunks)} chunk(s) from path={chunks[0].path!s}")
        except Exception as exc:
            raise StorageError(f"ChromaDB upsert failed: {exc}") from exc

    def delete_by_path(self, path: str) -> None:
        """Remove all chunks for *path*.

        Args:
            path: Absolute path string.
        """
        try:
            results = self._collection.get(
                where={"path": {"$eq": path}},
                include=[],
            )
            ids: list[str] = results["ids"]
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunk(s) for path={path!r}")
            else:
                logger.debug(f"No chunks found for path={path!r}; nothing deleted")
        except Exception as exc:
            logger.warning(f"Failed to delete vectors for path={path!r}: {exc}")

    def search(
        self,
        query_vec: list[float],
        top_k: int,
        filter_: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """Query the collection for the *top_k* nearest neighbours.

        Args:
            query_vec: Dense query embedding.
            top_k: Maximum number of hits to return.
            filter_: Optional ChromaDB ``where`` filter.

        Returns:
            List of :class:`~memorymesh.core.models.SearchHit`, best-score first.
        """
        n = self.count()
        if n == 0:
            return []

        actual_k = min(top_k, n)

        try:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=actual_k,
                where=filter_,
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:
            raise StorageError(f"ChromaDB query failed: {exc}") from exc

        hits: list[SearchHit] = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for _id, distance, meta, doc in zip(ids, distances, metadatas, documents, strict=True):
            score = max(0.0, 1.0 - float(distance))
            preview = (doc or "")[:200]
            hits.append(
                SearchHit(
                    path=str(meta.get("path", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=round(score, 6),
                    preview=preview,
                    file_type=str(meta.get("file_type", "")),
                    mtime=float(meta.get("mtime", 0.0)),
                    source_root=str(meta.get("source_root", "")),
                    start_char=int(meta.get("start_char", 0)),
                    end_char=int(meta.get("end_char", 0)),
                )
            )

        return hits

    def get_by_path(
        self,
        path: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return raw chunk records for all chunks at *path*.

        Args:
            path: Absolute path string.
            limit: Maximum number of chunks to return.

        Returns:
            List of dicts with ``"text"``, ``"path"``, and other metadata keys.
        """
        try:
            results = self._collection.get(
                where={"path": {"$eq": path}},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:
            logger.warning(f"ChromaVectorStore.get_by_path failed for {path!r}: {exc}")
            return []

        chunks: list[dict[str, object]] = []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        for doc, meta in zip(documents, metadatas, strict=False):
            record: dict[str, object] = dict(meta or {})
            record["text"] = doc or ""
            chunks.append(record)
        return chunks

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return int(self._collection.count())

    @property
    def model_id(self) -> str:
        """The embedding model ID stored in collection metadata."""
        return self._embedding_model_id


def _chunk_to_metadata(chunk: ChunkWithEmbedding) -> dict[str, object]:
    """Convert a chunk's fields to a flat ChromaDB metadata dict.

    ChromaDB metadata values must be str, int, float, or bool.
    """
    meta: dict[str, object] = {
        "path": str(chunk.path),
        "chunk_index": chunk.chunk_index,
        "file_type": chunk.file_type,
        "mtime": chunk.mtime,
        "source_root": chunk.source_root,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    cm: ChunkMetadata = chunk.metadata
    if cm.heading_path is not None:
        meta["heading_path"] = json.dumps(cm.heading_path)
    if cm.function_name is not None:
        meta["function_name"] = cm.function_name
    if cm.class_name is not None:
        meta["class_name"] = cm.class_name
    if cm.language is not None:
        meta["language"] = cm.language
    # chunk_type distinguishes content chunks ("chunk") from LLM summaries ("summary").
    meta["chunk_type"] = cm.chunk_type
    return meta
