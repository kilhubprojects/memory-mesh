"""File indexer - the heart of MemoryMesh.

Pipeline per file:
1. Compute SHA-256 (streaming, 64 KB chunks).
2. Compare with :class:`~memorymesh.storage.metadata_store.MetadataStore` -
   skip if hash unchanged.
3. Write ``indexing`` intent to SQLite.
4. Parse -> Chunk -> Embed -> Upsert Chroma -> Upsert BM25 -> mark SQLite
   ``indexed``.
5. On storage/index failures: log.warning + write ``failed`` record + continue.

Reconciliation on dirty shutdown
---------------------------------
If :meth:`MetadataStore.is_clean_state` returns ``False``, :meth:`reconcile`
removes Chroma entries not present in SQLite (orphan vectors) and queues
reindexing for SQLite entries not present in Chroma (missing vectors).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from memorymesh.core.models import (
    FileRecord,
    IndexResult,
)

if TYPE_CHECKING:
    from memorymesh.core.models import MemoryMeshConfig, ParsedDocument
    from memorymesh.embeddings.base import EmbeddingProvider
    from memorymesh.embeddings.multimodal.clip_provider import CLIPProvider
    from memorymesh.embeddings.multimodal.whisper_provider import WhisperProvider
    from memorymesh.llm.ollama_client import OllamaClient
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.vector_store import VectorStore

# File extensions handled via CLIP image embeddings (Wave 5).
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
)

# File extensions handled via Whisper transcription (Wave 5).
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
)

# Maximum characters of document text sent to Ollama for summary generation.
_SUMMARY_MAX_CHARS = 2_000

# Prompt template for document summarisation.
_SUMMARY_PROMPT = (
    "Summarize this document in 2-3 sentences optimized for search indexing:\n\n{text}"
)

# Maximum characters of chunk text sent to Ollama for entity extraction.
_ENTITY_EXCERPT_MAX_CHARS = 1_500

# Prompt template for entity extraction - expects a JSON array response.
_ENTITY_PROMPT = (
    "Extract named entities from the text below. "
    "Return ONLY a compact JSON array of objects, each with exactly two keys: "
    '"name" (string, lowercase) and "type" (one of: {types}). '
    "Extract at most 10 entities. Return [] if none found. "
    "Do NOT include any explanation outside the JSON.\n\n"
    "Text:\n{text}\n\n"
    "JSON:"
)


class FileIndexer:
    """Orchestrates the full parse -> chunk -> embed -> store pipeline.

    When ``config.indexing.generate_summaries`` is ``True`` and an
    :class:`~memorymesh.llm.ollama_client.OllamaClient` is provided, the indexer
    generates an LLM summary for each document after normal chunking and stores
    it as a dedicated ``chunk_type="summary"`` chunk.  Summary chunks are used
    only for retrieval - the :class:`~memorymesh.search.engine.SearchEngine`
    filters them out of final results returned to the user.

    Args:
        config: Root configuration (used for source filters and OCR settings).
        vector_store: ChromaDB wrapper for dense vectors.
        metadata_store: SQLite metadata store (source of truth).
        embedding_provider: Converts chunk text to embeddings.
        bm25_index: Sparse BM25 index.
        ollama_client: Optional Ollama LLM client for summary generation.
            ``None`` disables summaries regardless of config.
    """

    def __init__(
        self,
        config: MemoryMeshConfig,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        embedding_provider: EmbeddingProvider,
        bm25_index: BM25Index,
        ollama_client: OllamaClient | None = None,
        clip_provider: CLIPProvider | None = None,
        whisper_provider: WhisperProvider | None = None,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._provider = embedding_provider
        self._bm25 = bm25_index
        self._ollama = ollama_client
        self._clip_provider = clip_provider
        self._whisper_provider = whisper_provider

        # Build registries once and reuse across all files
        from memorymesh.chunking.registry import get_chunker
        from memorymesh.parsing.registry import build_registry

        self._parser_registry = build_registry(ocr_config=config.ocr)
        self._get_chunker = get_chunker

        # Optional semantic deduplicator
        self._deduplicator = None
        if config.indexing.dedup_enabled:
            from memorymesh.indexer.deduplicator import SemanticDeduplicator

            self._deduplicator = SemanticDeduplicator(threshold=config.indexing.dedup_threshold)

        # Optional PII filter
        self._pii_filter = None
        if config.indexing.pii_detection:
            from memorymesh.indexer.pii_filter import PIIFilter

            self._pii_filter = PIIFilter(redact=config.indexing.pii_redact)

        # Ensure schema exists (idempotent - safe to call every startup)
        self._metadata_store.init_schema()

    def startup(self) -> None:
        """Mark daemon startup in SQLite; triggers reconciliation if dirty."""
        was_clean = self._metadata_store.is_clean_state()
        self._metadata_store.mark_startup()
        if not was_clean:
            logger.warning("FileIndexer: dirty shutdown detected - starting reconciliation")
            self.reconcile()

    def shutdown(self) -> None:
        """Mark clean shutdown in SQLite."""
        self._metadata_store.mark_clean_shutdown()

    def index_file(
        self,
        path: Path,
        source_name: str = "",
        *,
        force: bool = False,
    ) -> IndexResult:
        """Index a single file through the full pipeline.

        Args:
            path: Absolute path to the file.
            source_name: Name of the owning source (for metadata).
            force: Skip hash comparison and always re-index.

        Returns:
            :class:`~memorymesh.core.models.IndexResult` with the outcome.
        """
        t0 = time.perf_counter()

        ext = path.suffix.lower()

        if ext in _IMAGE_EXTENSIONS:
            return self._index_image(path, source_name, t0, force=force)
        if ext in _AUDIO_EXTENSIONS:
            return self._index_audio(path, source_name, t0, force=force)

        parser = self._parser_registry.get(ext)
        if parser is None:
            logger.debug(f"FileIndexer: unsupported extension {ext!r} - {path.name!r}")
            return IndexResult(
                path=path,
                status="unsupported",
                duration_ms=_elapsed_ms(t0),
            )

        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning(f"FileIndexer: cannot stat {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        if not force:
            try:
                current_hash = compute_hash(path)
            except OSError as exc:
                logger.warning(f"FileIndexer: cannot hash {path}: {exc}")
                return IndexResult(
                    path=path,
                    status="parse_error",
                    error=str(exc),
                    duration_ms=_elapsed_ms(t0),
                )

            existing = self._metadata_store.get_file(str(path))
            if existing is not None and existing.sha256 == current_hash:
                logger.debug(f"FileIndexer: skipping unchanged {path.name!r}")
                return IndexResult(
                    path=path,
                    status="skipped",
                    duration_ms=_elapsed_ms(t0),
                )
        else:
            current_hash = compute_hash(path)

        try:
            doc = parser.parse(path)
        except Exception as exc:
            logger.warning(f"FileIndexer: parse failed for {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        if doc.metadata.get("error"):
            err = str(doc.metadata["error"])
            logger.warning(f"FileIndexer: parser returned error for {path.name!r}: {err}")
            self._record_error(path, source_name, stat, current_hash, err)
            return IndexResult(
                path=path,
                status="parse_error",
                error=err,
                duration_ms=_elapsed_ms(t0),
            )

        try:
            chunker = self._get_chunker(ext, self._config.chunking)
            chunks = chunker.chunk(doc)
        except Exception as exc:
            logger.warning(f"FileIndexer: chunking failed for {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        # Attach source_root and mtime to chunks
        for chunk in chunks:
            chunk.source_root = source_name
            chunk.mtime = stat.st_mtime

        try:
            texts = [c.text for c in chunks]
            embeddings = self._provider.embed_documents(texts)
        except Exception as exc:
            logger.warning(f"FileIndexer: embedding failed for {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        from memorymesh.core.models import ChunkWithEmbedding

        chunks_with_embeddings = [
            ChunkWithEmbedding(**chunk.model_dump(), embedding=emb)
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        if self._deduplicator is not None:
            chunks_with_embeddings = self._deduplicator.filter(chunks_with_embeddings)

        if self._pii_filter is not None:
            chunks_with_embeddings = self._pii_filter.filter(chunks_with_embeddings)

        # Writing "indexing" here ensures that a crash during Chroma/BM25 is
        # detectable at next startup via is_clean_state() / reconcile().
        _intent_record = FileRecord(
            path=str(path),
            source_name=source_name,
            sha256=current_hash,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            file_type=ext,
            n_chunks=len(chunks),
            status="indexing",
            indexed_at=time.time(),
            embedding_model_id=self._provider.model_id,
        )
        try:
            self._metadata_store.upsert_file(_intent_record)
        except Exception as exc:
            logger.warning(f"FileIndexer: pre-index SQLite write failed for {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        try:
            self._vector_store.delete_by_path(str(path))
            self._vector_store.upsert(chunks_with_embeddings)
        except Exception as exc:
            logger.warning(f"FileIndexer: vector upsert failed for {path}: {exc}")
            # Write "failed" (not "parse_error") - parse succeeded, the store write failed.
            _fail_record = FileRecord(
                path=str(path),
                source_name=source_name,
                sha256=current_hash,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
                file_type=ext,
                n_chunks=0,
                status="failed",
                error_message=str(exc)[:500],
                indexed_at=time.time(),
                embedding_model_id=self._provider.model_id,
            )
            try:
                self._metadata_store.upsert_file(_fail_record)
            except Exception as meta_exc:
                logger.warning(
                    f"FileIndexer: failed to write 'failed' record for {path}: {meta_exc}"
                )
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        try:
            self._bm25.delete_by_path(str(path))
            self._bm25.add_chunks([(f"{path}:{c.chunk_index}", str(path), c.text) for c in chunks])
        except Exception as exc:
            logger.warning(f"FileIndexer: BM25 update failed for {path}: {exc}")
            _fail_record = FileRecord(
                path=str(path),
                source_name=source_name,
                sha256=current_hash,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
                file_type=ext,
                n_chunks=0,
                status="failed",
                error_message=str(exc)[:500],
                indexed_at=time.time(),
                embedding_model_id=self._provider.model_id,
            )
            try:
                self._metadata_store.upsert_file(_fail_record)
            except Exception as meta_exc:
                logger.warning(
                    f"FileIndexer: failed to write 'failed' record for {path}: {meta_exc}"
                )
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        if self._config.memory.entity_extraction.enabled and self._ollama is not None:
            self._maybe_extract_entities(
                path=path,
                text=doc.text,
                chunks=chunks,
                extract_types=self._config.memory.entity_extraction.extract_types,
            )

        if self._config.indexing.generate_summaries and self._ollama is not None:
            self._maybe_index_summary(path, doc.text, source_name, stat)

        record = FileRecord(
            path=str(path),
            source_name=source_name,
            sha256=current_hash,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            file_type=ext,
            n_chunks=len(chunks),
            status="indexed",
            indexed_at=time.time(),
            embedding_model_id=self._provider.model_id,
        )
        self._metadata_store.upsert_file(record)

        duration = _elapsed_ms(t0)
        logger.info(
            f"FileIndexer: indexed {path.name!r} chunks={len(chunks)} duration={duration:.0f}ms"
        )
        return IndexResult(
            path=path,
            status="indexed",
            n_chunks=len(chunks),
            duration_ms=duration,
        )

    def index_directory(
        self,
        root: Path,
        source_name: str = "",
        recursive: bool = True,
        extensions: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        *,
        force: bool = False,
    ) -> list[IndexResult]:
        """Index all supported files in *root*.

        Args:
            root: Directory to scan.
            source_name: Owning source name for all files.
            recursive: Whether to descend into subdirectories.
            extensions: Whitelist of lowercase extensions.  ``None`` means all
                supported types.
            ignore_patterns: Glob patterns relative to *root* to skip.
            force: Re-index even unchanged files.

        Returns:
            List of :class:`~memorymesh.core.models.IndexResult`, one per file.
        """
        results: list[IndexResult] = []
        all_ignore = list(self._config.global_ignore) + (ignore_patterns or [])
        walker = root.rglob("*") if recursive else root.glob("*")

        for path in walker:
            if not path.is_file():
                continue
            if _should_ignore(path, root, all_ignore):
                continue
            ext = path.suffix.lower()
            if extensions and ext not in extensions:
                continue
            if (
                ext not in self._parser_registry
                and ext not in _IMAGE_EXTENSIONS
                and ext not in _AUDIO_EXTENSIONS
            ):
                continue

            result = self.index_file(path, source_name=source_name, force=force)
            results.append(result)

        return results

    def index_parsed_document(
        self,
        doc: ParsedDocument,
        source_name: str = "",
    ) -> IndexResult:
        """Index a pre-parsed document, skipping the parse step.

        Used by the ``sync`` command to index documents produced by external
        data connectors.  The document's ``path`` is used as the unique key
        in all stores (it may be a synthetic URI such as ``jira://ENG-1.jira``).

        Args:
            doc: Pre-parsed document from a connector.
            source_name: Owning source name (for metadata).

        Returns:
            :class:`~memorymesh.core.models.IndexResult` with the outcome.
        """
        t0 = time.perf_counter()
        path = doc.path

        try:
            chunker = self._get_chunker(doc.file_type, self._config.chunking)
            chunks = chunker.chunk(doc)
        except Exception as exc:
            logger.warning(f"FileIndexer: chunking failed for connector doc {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        mtime = time.time()
        for chunk in chunks:
            chunk.source_root = source_name or doc.metadata.get("source", "")
            chunk.mtime = mtime

        try:
            texts = [c.text for c in chunks]
            embeddings = self._provider.embed_documents(texts)
        except Exception as exc:
            logger.warning(f"FileIndexer: embedding failed for connector doc {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        from memorymesh.core.models import ChunkWithEmbedding

        chunks_with_embeddings = [
            ChunkWithEmbedding(**chunk.model_dump(), embedding=emb)
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        if self._deduplicator is not None:
            chunks_with_embeddings = self._deduplicator.filter(chunks_with_embeddings)

        if self._pii_filter is not None:
            chunks_with_embeddings = self._pii_filter.filter(chunks_with_embeddings)

        try:
            self._vector_store.delete_by_path(str(path))
            self._vector_store.upsert(chunks_with_embeddings)
        except Exception as exc:
            logger.warning(f"FileIndexer: vector upsert failed for connector doc {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        try:
            self._bm25.delete_by_path(str(path))
            self._bm25.add_chunks([(f"{path}:{c.chunk_index}", str(path), c.text) for c in chunks])
        except Exception as exc:
            logger.warning(f"FileIndexer: BM25 update failed for connector doc {path}: {exc}")

        record = FileRecord(
            path=str(path),
            source_name=source_name or doc.metadata.get("source", ""),
            sha256="",
            mtime=mtime,
            size_bytes=len(doc.text.encode()),
            file_type=doc.file_type,
            n_chunks=len(chunks),
            status="indexed",
            indexed_at=mtime,
            embedding_model_id=self._provider.model_id,
        )
        self._metadata_store.upsert_file(record)

        duration = _elapsed_ms(t0)
        logger.info(
            f"FileIndexer: connector doc {path.name!r} "
            f"chunks={len(chunks)} duration={duration:.0f}ms"
        )
        return IndexResult(
            path=path,
            status="indexed",
            n_chunks=len(chunks),
            duration_ms=duration,
        )

    def delete_file(self, path: Path) -> None:
        """Remove a file from all indexes.

        Args:
            path: Absolute path to the file.
        """
        str_path = str(path)
        self._vector_store.delete_by_path(str_path)
        self._bm25.delete_by_path(str_path)
        self._metadata_store.mark_deleted(str_path)
        logger.info(f"FileIndexer: deleted {path.name!r} from all indexes")

    def reconcile(self) -> None:
        """Reconcile SQLite state against ChromaDB after a dirty shutdown.

        - Entries in SQLite with status ``"indexed"`` but missing in Chroma
          -> reindex.
        - Chroma chunks with no corresponding SQLite entry are tolerated for
          now (they will be cleaned up on the next full scan).
        """
        logger.info("FileIndexer: reconciliation started")
        records = self._metadata_store.list_files(status="indexed")
        reindexed = 0
        for record in records:
            path = Path(record.path)
            if not path.exists():
                self._metadata_store.mark_deleted(record.path)
                continue
            result = self.index_file(path, source_name=record.source_name, force=True)
            if result.status == "indexed":
                reindexed += 1

        logger.info(f"FileIndexer: reconciliation complete (reindexed={reindexed})")

    def _maybe_extract_entities(
        self,
        path: Path,
        text: str,
        chunks: list,
        extract_types: list[str],
    ) -> None:
        """Run LLM entity extraction over the first chunk of *text* and persist results.

        Sends a compact excerpt to Ollama and parses the JSON response into
        :class:`~memorymesh.core.models.Entity` + mention records.  All errors
        are swallowed - entity extraction is best-effort and must never break
        normal indexing.

        Args:
            path: Source file path (used to build chunk IDs for mentions).
            text: Full document text (only the first
                ``_ENTITY_EXCERPT_MAX_CHARS`` characters are sent to the LLM).
            chunks: Already-indexed chunks (used to record mention chunk IDs).
            extract_types: Entity type whitelist (e.g. ``["person", "project"]``).
        """
        import json

        assert self._ollama is not None  # guarded by caller

        excerpt = text[:_ENTITY_EXCERPT_MAX_CHARS].strip()
        if not excerpt:
            return

        types_str = ", ".join(extract_types) if extract_types else "person, project, concept"
        prompt = _ENTITY_PROMPT.format(text=excerpt, types=types_str)

        try:
            raw_response = self._ollama.generate(prompt)
        except Exception as exc:
            logger.warning(
                f"FileIndexer: entity extraction Ollama call failed for {path.name!r}: {exc}"
            )
            return

        if not raw_response.strip():
            return

        # Parse the JSON array; Ollama sometimes wraps in markdown, so we
        # try to find the first '[' and last ']'.
        try:
            start = raw_response.index("[")
            end = raw_response.rindex("]") + 1
            entities_raw: list[dict] = json.loads(raw_response[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                f"FileIndexer: entity extraction JSON parse failed for "
                f"{path.name!r}: {exc!r} | raw={raw_response[:200]!r}"
            )
            return

        from memorymesh.core.models import Entity

        chunk_ids = [f"{path}:{c.chunk_index}" for c in chunks]

        for item in entities_raw[:20]:  # cap to avoid runaway responses
            name = str(item.get("name", "")).strip().lower()
            etype = str(item.get("type", "")).strip().lower()
            if not name or not etype:
                continue
            if extract_types and etype not in extract_types:
                continue

            try:
                entity = Entity(name=name, entity_type=etype)
                self._metadata_store.upsert_entity(entity)
                for cid in chunk_ids:
                    self._metadata_store.add_entity_mention(name, etype, cid)
            except Exception as exc:
                logger.warning(
                    f"FileIndexer: failed to persist entity {name!r} from {path.name!r}: {exc}"
                )

        logger.debug(f"FileIndexer: extracted {len(entities_raw)} entities from {path.name!r}")

    def _maybe_index_summary(
        self,
        path: Path,
        full_text: str,
        source_name: str,
        stat: os.stat_result,
    ) -> None:
        """Generate an LLM summary chunk and upsert it into the vector store.

        The summary chunk uses ``chunk_index=-1`` and
        ``ChunkMetadata.chunk_type="summary"`` so the search engine can
        distinguish it from regular content chunks and filter it from results.

        This method is a no-op on any error - the main indexing result is
        not affected.

        Args:
            path: Source file path.
            full_text: Full document text (truncated to ``_SUMMARY_MAX_CHARS``
                before sending to Ollama).
            source_name: Owning source name.
            stat: ``os.stat_result`` for the file (used for mtime/size).
        """
        assert self._ollama is not None  # guarded by caller

        text_snippet = full_text[:_SUMMARY_MAX_CHARS]
        if not text_snippet.strip():
            return

        try:
            summary = self._ollama.generate(_SUMMARY_PROMPT.format(text=text_snippet))
        except Exception as exc:
            logger.warning(f"FileIndexer: summary generation failed for {path.name!r}: {exc}")
            return

        if not summary.strip():
            logger.warning(f"FileIndexer: Ollama returned empty summary for {path.name!r}")
            return

        from memorymesh.core.models import Chunk, ChunkMetadata, ChunkWithEmbedding

        summary_chunk = Chunk(
            path=path,
            chunk_index=-1,
            text=summary,
            start_char=0,
            end_char=len(summary),
            file_type=path.suffix.lower(),
            mtime=stat.st_mtime,
            source_root=source_name,
            metadata=ChunkMetadata(chunk_type="summary"),
        )

        try:
            vec = self._provider.embed_query(summary)
            cwe = ChunkWithEmbedding(**summary_chunk.model_dump(), embedding=vec)
            self._vector_store.upsert([cwe])
            logger.debug(
                f"FileIndexer: summary chunk indexed for {path.name!r} ({len(summary)} chars)"
            )
        except Exception as exc:
            logger.warning(f"FileIndexer: failed to upsert summary chunk for {path.name!r}: {exc}")

    def _index_image(
        self,
        path: Path,
        source_name: str,
        t0: float,
        *,
        force: bool = False,
    ) -> IndexResult:
        """Index an image file using CLIP embeddings (Wave 5).

        Uses :class:`~memorymesh.embeddings.multimodal.clip_provider.CLIPProvider`
        if available.  Falls back to ``"unsupported"`` when ``open-clip-torch``
        is not installed.

        The image is represented as a single chunk whose text is a minimal
        placeholder and whose embedding comes from the CLIP visual encoder.

        Args:
            path: Absolute path to the image file.
            source_name: Owning source name.
            t0: ``time.perf_counter()`` timestamp from the caller.
            force: Skip hash check and always re-index.

        Returns:
            :class:`~memorymesh.core.models.IndexResult` with status
            ``"indexed"`` on success or ``"unsupported"``/``"parse_error"``
            on failure.
        """
        if self._clip_provider is None:
            logger.debug(
                f"FileIndexer: CLIP provider not configured - skipping image {path.name!r}"
            )
            return IndexResult(path=path, status="unsupported", duration_ms=_elapsed_ms(t0))

        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning(f"FileIndexer: cannot stat image {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        if not force:
            try:
                current_hash = compute_hash(path)
            except OSError as exc:
                logger.warning(f"FileIndexer: cannot hash image {path}: {exc}")
                return IndexResult(
                    path=path,
                    status="parse_error",
                    error=str(exc),
                    duration_ms=_elapsed_ms(t0),
                )
            existing = self._metadata_store.get_file(str(path))
            if existing is not None and existing.sha256 == current_hash:
                logger.debug(f"FileIndexer: skipping unchanged image {path.name!r}")
                return IndexResult(path=path, status="skipped", duration_ms=_elapsed_ms(t0))
        else:
            current_hash = compute_hash(path)

        vec = self._clip_provider.embed_image(path)
        if vec is None:
            logger.debug(f"FileIndexer: CLIP embed_image returned None for {path.name!r}")
            return IndexResult(path=path, status="unsupported", duration_ms=_elapsed_ms(t0))

        file_size_kb = stat.st_size / 1024

        from memorymesh.core.models import Chunk, ChunkMetadata, ChunkWithEmbedding

        text = f"Image: {path.name}\nPath: {path}\nSize: {file_size_kb:.1f}KB"
        chunk = Chunk(
            path=path,
            chunk_index=0,
            text=text,
            start_char=0,
            end_char=len(text),
            file_type=path.suffix.lower(),
            mtime=stat.st_mtime,
            source_root=source_name,
            metadata=ChunkMetadata(chunk_type="image"),
        )
        cwe = ChunkWithEmbedding(**chunk.model_dump(), embedding=vec)

        try:
            self._vector_store.delete_by_path(str(path))
            self._vector_store.upsert([cwe])
        except Exception as exc:
            logger.warning(f"FileIndexer: CLIP upsert failed for {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        record = FileRecord(
            path=str(path),
            source_name=source_name,
            sha256=current_hash,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            file_type=path.suffix.lower(),
            n_chunks=1,
            status="indexed",
            indexed_at=time.time(),
            embedding_model_id=f"clip:{self._clip_provider.model_id}",
        )
        self._metadata_store.upsert_file(record)

        duration = _elapsed_ms(t0)
        logger.info(f"FileIndexer: indexed image {path.name!r} via CLIP duration={duration:.0f}ms")
        return IndexResult(path=path, status="indexed", n_chunks=1, duration_ms=duration)

    def _index_audio(
        self,
        path: Path,
        source_name: str,
        t0: float,
        *,
        force: bool = False,
    ) -> IndexResult:
        """Transcribe an audio file with Whisper, then index the transcript (Wave 5).

        Uses :class:`~memorymesh.embeddings.multimodal.whisper_provider.WhisperProvider`
        if available.  Falls back to ``"unsupported"`` when ``openai-whisper``
        is not installed.  The resulting transcript is passed through the normal
        text chunking + embedding pipeline so it is fully searchable.

        Args:
            path: Absolute path to the audio file.
            source_name: Owning source name.
            t0: ``time.perf_counter()`` timestamp from the caller.
            force: Skip hash check and always re-index.

        Returns:
            :class:`~memorymesh.core.models.IndexResult`.
        """
        if self._whisper_provider is None:
            logger.debug(
                f"FileIndexer: Whisper provider not configured - skipping audio {path.name!r}"
            )
            return IndexResult(path=path, status="unsupported", duration_ms=_elapsed_ms(t0))

        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning(f"FileIndexer: cannot stat audio {path}: {exc}")
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        if not force:
            try:
                current_hash = compute_hash(path)
            except OSError as exc:
                logger.warning(f"FileIndexer: cannot hash audio {path}: {exc}")
                return IndexResult(
                    path=path,
                    status="parse_error",
                    error=str(exc),
                    duration_ms=_elapsed_ms(t0),
                )
            existing = self._metadata_store.get_file(str(path))
            if existing is not None and existing.sha256 == current_hash:
                logger.debug(f"FileIndexer: skipping unchanged audio {path.name!r}")
                return IndexResult(path=path, status="skipped", duration_ms=_elapsed_ms(t0))
        else:
            current_hash = compute_hash(path)

        transcript = self._whisper_provider.transcribe(path)

        if not transcript or not transcript.strip():
            logger.warning(f"FileIndexer: Whisper returned empty transcript for {path.name!r}")
            # Still record the file so it is not retried endlessly
            record = FileRecord(
                path=str(path),
                source_name=source_name,
                sha256=current_hash,
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
                file_type=path.suffix.lower(),
                n_chunks=0,
                status="indexed",
                indexed_at=time.time(),
                embedding_model_id=self._provider.model_id,
            )
            self._metadata_store.upsert_file(record)
            return IndexResult(path=path, status="indexed", n_chunks=0, duration_ms=_elapsed_ms(t0))

        # Build a synthetic ParsedDocument and run through the normal pipeline
        from memorymesh.core.models import ParsedDocument

        doc = ParsedDocument(
            path=path,
            text=transcript,
            file_type=path.suffix.lower(),
            encoding="utf-8",
            metadata={"whisper_transcript": True},
        )

        from memorymesh.chunking.registry import get_chunker

        try:
            chunker = get_chunker(path.suffix.lower(), self._config.chunking)
            chunks = chunker.chunk(doc)
        except Exception as exc:
            logger.warning(f"FileIndexer: chunking failed for audio {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        for chunk in chunks:
            chunk.source_root = source_name
            chunk.mtime = stat.st_mtime

        try:
            texts = [c.text for c in chunks]
            embeddings = self._provider.embed_documents(texts)
        except Exception as exc:
            logger.warning(f"FileIndexer: embedding failed for audio {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        from memorymesh.core.models import ChunkWithEmbedding

        chunks_with_embeddings = [
            ChunkWithEmbedding(**chunk.model_dump(), embedding=emb)
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        try:
            self._vector_store.delete_by_path(str(path))
            self._vector_store.upsert(chunks_with_embeddings)
        except Exception as exc:
            logger.warning(f"FileIndexer: upsert failed for audio {path}: {exc}")
            self._record_error(path, source_name, stat, current_hash, str(exc))
            return IndexResult(
                path=path,
                status="parse_error",
                error=str(exc),
                duration_ms=_elapsed_ms(t0),
            )

        try:
            self._bm25.delete_by_path(str(path))
            self._bm25.add_chunks([(f"{path}:{c.chunk_index}", str(path), c.text) for c in chunks])
        except Exception as exc:
            logger.warning(f"FileIndexer: BM25 update failed for audio {path}: {exc}")

        record = FileRecord(
            path=str(path),
            source_name=source_name,
            sha256=current_hash,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            file_type=path.suffix.lower(),
            n_chunks=len(chunks),
            status="indexed",
            indexed_at=time.time(),
            embedding_model_id=self._provider.model_id,
        )
        self._metadata_store.upsert_file(record)

        duration = _elapsed_ms(t0)
        logger.info(
            f"FileIndexer: indexed audio {path.name!r} via Whisper "
            f"chunks={len(chunks)} duration={duration:.0f}ms"
        )
        return IndexResult(path=path, status="indexed", n_chunks=len(chunks), duration_ms=duration)

    def _record_error(
        self,
        path: Path,
        source_name: str,
        stat: os.stat_result,
        sha256: str,
        error: str,
    ) -> None:
        record = FileRecord(
            path=str(path),
            source_name=source_name,
            sha256=sha256,
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            file_type=path.suffix.lower(),
            n_chunks=0,
            status="parse_error",
            error_message=error[:500],
            indexed_at=time.time(),
            embedding_model_id=self._provider.model_id,
        )
        try:
            self._metadata_store.upsert_file(record)
        except Exception as exc:
            logger.warning(f"FileIndexer: failed to write error record for {path}: {exc}")


def compute_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file using streaming reads.

    Args:
        path: Absolute path to the file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Lowercase hex digest string.

    Raises:
        OSError: If the file cannot be read.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _should_ignore(path: Path, root: Path, patterns: list[str]) -> bool:
    """Return True if *path* matches any of *patterns* relative to *root*."""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    full = str(path)
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(full, pat):
            return True
        # Support glob patterns like **/node_modules/**
        parts = path.parts
        for part in parts:
            clean_pat = pat.replace("**/", "").replace("/**", "").replace("**", "")
            if clean_pat and fnmatch.fnmatch(part, clean_pat):
                return True
    return False
