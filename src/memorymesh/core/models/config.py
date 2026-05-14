"""Configuration models for MemoryMesh.

All Pydantic models that mirror ``config.yaml`` keys live here.  Every field
has a sane default so the daemon can boot without a config file present.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

FileStatusLiteral = Literal[
    "indexed",
    "indexing",  # in-progress write; signals unclean state if found at startup
    "failed",  # Chroma/BM25 write failed after parse succeeded
    "parse_error",
    "unsupported",
    "deleted",
    "pending_reindex",
]


_DEFAULT_GLOBAL_IGNORE: list[str] = [
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.env",
    "**/.env.*",
    "**/*.key",
    "**/*.pem",
    "**/id_rsa*",
    "**/secrets/**",
    "**/.ssh/**",
    "**/.aws/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
]


class MemoryTier(StrEnum):
    """Memory tier for hierarchical storage.

    * ``hot``  - recently accessed or manually pinned; kept in RAM cache.
    * ``warm`` - normal indexed content (default).
    * ``cold`` - rarely accessed; score is decayed; may be compressed.
    """

    hot = "hot"
    warm = "warm"
    cold = "cold"


class AgentPermission(StrEnum):
    """Access level granted to a named agent client.

    Permissions are ordered: each level includes all lower ones.
    """

    read = "read"
    read_index = "read+index"
    read_index_delete = "read+index+delete"
    admin = "admin"


# CONFIG MODELS


class SourceConfig(BaseModel):
    """A single monitored directory.

    Args:
        name: Human-readable identifier (optional).
        path: Absolute or ``~``-prefixed path to index.
        recursive: Whether to descend into subdirectories.
        extensions: Whitelist of file extensions (e.g. ``[".py", ".md"]``).
            Empty list means "use the global default extension list".
        ignore: Additional glob patterns to skip inside this source.
    """

    name: str = ""
    path: Path
    recursive: bool = True
    extensions: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(default_factory=list)

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, v: Any) -> Path:
        return Path(str(v)).expanduser()


class EmbeddingsConfig(BaseModel):
    """Embedding provider settings.

    Args:
        provider: Provider key - ``"sentence_transformers"`` in the MVP.
        model: Model name passed to the provider.
        device: Compute device - ``"auto"`` lets the provider decide.
        batch_size: Number of texts embedded per forward pass.
        normalize: Whether to L2-normalise vectors (enables cosine via dot product).
    """

    provider: str = "sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = 32
    normalize: bool = True


class RecursiveChunkingConfig(BaseModel):
    """Settings for the recursive text splitter."""

    strategy: Literal["recursive"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


class MarkdownChunkingConfig(BaseModel):
    """Settings for the heading-based markdown splitter."""

    strategy: Literal["by_heading"] = "by_heading"
    max_chunk_size: int = 800


class CodeChunkingConfig(BaseModel):
    """Settings for the tree-sitter AST-aware code splitter."""

    strategy: Literal["tree_sitter"] = "tree_sitter"
    max_chunk_size: int = 1024
    fallback: str = "recursive"


class ChunkingConfig(BaseModel):
    """Per-content-type chunking configuration."""

    default: RecursiveChunkingConfig = Field(default_factory=RecursiveChunkingConfig)
    markdown: MarkdownChunkingConfig = Field(default_factory=MarkdownChunkingConfig)
    code: CodeChunkingConfig = Field(default_factory=CodeChunkingConfig)


class OcrConfig(BaseModel):
    """Optional OCR fallback for scanned PDFs.

    Args:
        enabled: Master switch; OCR is never called when ``False``.
        backend: ``"tesseract"`` (default) or ``"easyocr"``.
        languages: Language codes passed to the backend.
        trigger: When to invoke OCR - ``"empty_text_only"`` means only when
            the native parser returned fewer than 50 characters.
        max_file_size_mb: Files larger than this are skipped by OCR.
    """

    enabled: bool = False
    backend: Literal["tesseract", "easyocr"] = "tesseract"
    languages: list[str] = Field(default_factory=lambda: ["eng"])
    trigger: Literal["empty_text_only"] = "empty_text_only"
    max_file_size_mb: int = 50


class SearchHybridConfig(BaseModel):
    """Hybrid (dense + sparse) fusion settings."""

    enabled: bool = True
    rrf_k: int = 60
    over_fetch_factor: int = 3


class SearchRerankerConfig(BaseModel):
    """Cross-encoder reranker settings."""

    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    top_k_before_rerank: int = 35


class QueryExpansionConfig(BaseModel):
    """Query expansion settings (lexical variants + HyDE).

    Args:
        enabled: Master switch.
        use_hyde: Generate a hypothetical document via Ollama and embed it.
        n_lexical_variants: How many stemming/synonym variants to generate.
    """

    enabled: bool = False
    use_hyde: bool = False
    n_lexical_variants: int = 1


class OllamaConfig(BaseModel):
    """Ollama local LLM settings (used for HyDE query expansion).

    Args:
        enabled: Whether Ollama is available for use.
        base_url: Base URL of the Ollama HTTP API.
        model: Model name to call for HyDE generation.
        timeout_s: Request timeout in seconds.
    """

    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    timeout_s: int = 10


class SearchConfig(BaseModel):
    """Query-time search settings."""

    default_top_k: int = 10
    hybrid: SearchHybridConfig = Field(default_factory=SearchHybridConfig)
    reranker: SearchRerankerConfig = Field(default_factory=SearchRerankerConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    parent_window_chars: int = Field(
        default=0,
        description=(
            "When > 0, search_memory returns an extended_preview with this many "
            "additional characters of context around each hit. 0 = disabled."
        ),
    )


class StorageConfig(BaseModel):
    """Paths for all persistent state managed by MemoryMesh.

    All sub-paths (``metadata_db``, ``bm25_pickle``, ``audit_log``) are
    *relative to* ``base_dir``.  Use the ``*_path`` properties to get the
    resolved absolute :class:`~pathlib.Path`.

    Note: the ``*_path`` computed properties are Python ``@property`` methods and
    do **not** appear in ``model_dump()`` / ``model_json_schema()`` output.
    """

    base_dir: Path = Field(default_factory=lambda: Path("~/.memorymesh").expanduser())
    vector_db_subdir: str = "chroma_db"
    metadata_db: str = "metadata.sqlite3"
    bm25_pickle: str = "bm25.pkl"
    audit_log: str = "audit.jsonl"

    @field_validator("base_dir", mode="before")
    @classmethod
    def _expand_base_dir(cls, v: Any) -> Path:
        return Path(str(v)).expanduser()

    @property
    def vector_db_path(self) -> Path:
        """Absolute path to the ChromaDB directory."""
        return self.base_dir / self.vector_db_subdir

    @property
    def metadata_db_path(self) -> Path:
        """Absolute path to the SQLite metadata database."""
        return self.base_dir / self.metadata_db

    @property
    def bm25_pickle_path(self) -> Path:
        """Absolute path to the BM25 pickle file."""
        return self.base_dir / self.bm25_pickle

    @property
    def audit_log_path(self) -> Path:
        """Absolute path to the JSONL audit log."""
        return self.base_dir / self.audit_log


class ServerHttpConfig(BaseModel):
    """HTTP listener settings for the streamable-http transport."""

    host: str = "127.0.0.1"
    port: int = 8765


class ServerConfig(BaseModel):
    """MCP server transport configuration."""

    transport: Literal["stdio", "streamable-http"] = "stdio"
    http: ServerHttpConfig = Field(default_factory=ServerHttpConfig)


class WatcherConfig(BaseModel):
    """Filesystem watcher settings."""

    enabled: bool = True
    debounce_ms: int = 1500
    worker_concurrency: int = 2
    use_polling: bool = False


class LoggingConfig(BaseModel):
    """Logging sink configuration.

    Args:
        level: Minimum log level string.
        format: ``"pretty"`` for coloured human-readable, ``"json"`` for NDJSON.
        file: Path to the rotating log file.  ``None`` disables file logging.
        rotation: loguru rotation spec (e.g. ``"10 MB"``).
        retention: loguru retention spec (e.g. ``"14 days"``).
    """

    level: str = "INFO"
    format: Literal["pretty", "json"] = "pretty"
    file: Path | None = Field(
        default_factory=lambda: Path("~/.memorymesh/logs/memorymesh.log").expanduser()
    )
    rotation: str = "10 MB"
    retention: str = "14 days"

    @field_validator("file", mode="before")
    @classmethod
    def _expand_file(cls, v: Any) -> Path | None:
        if v is None:
            return None
        return Path(str(v)).expanduser()


class MemoryTierConfig(BaseModel):
    """Thresholds that drive automatic tier promotion and demotion.

    Args:
        hot_tier_days: Chunks accessed within this many days stay in *hot*.
        cold_tier_days: Chunks not accessed for this many days fall to *cold*.
        hot_max_chunks: Maximum chunks held in the hot in-memory cache.
    """

    hot_tier_days: int = 7
    cold_tier_days: int = 90
    hot_max_chunks: int = 500


class ForgettingConfig(BaseModel):
    """Controls time-based score decay for cold-tier chunks.

    Args:
        enabled: Master switch for decay.
        decay_half_life_days: Days after which a chunk's effective score halves.
        min_score_floor: Effective score never drops below this fraction.
    """

    enabled: bool = False
    decay_half_life_days: float = 90.0
    min_score_floor: float = 0.1


class EpisodicMemoryConfig(BaseModel):
    """Settings for the episodic events table.

    Args:
        enabled: Whether the episodic events system is active.
        auto_record: Automatically record a ``retrieval`` event on each search hit.
    """

    enabled: bool = True
    auto_record: bool = True


class EntityExtractionConfig(BaseModel):
    """Settings for optional LLM-based entity extraction.

    Args:
        enabled: Requires ``ollama.enabled: true``.
        extract_types: Entity types to extract (person, project, concept, location).
    """

    enabled: bool = False
    extract_types: list[str] = Field(default_factory=lambda: ["person", "project", "concept"])


class MemoryConfig(BaseModel):
    """Root config for Wave-3 memory primitives."""

    tiers: MemoryTierConfig = Field(default_factory=MemoryTierConfig)
    forgetting: ForgettingConfig = Field(default_factory=ForgettingConfig)
    episodic: EpisodicMemoryConfig = Field(default_factory=EpisodicMemoryConfig)
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)


class AgentConfig(BaseModel):
    """Per-agent identity and access rules.

    Args:
        client_id: Stable identifier sent in ``X-MemoryMesh-Client`` header or
            MCP ``initialize`` metadata.
        name: Human-readable label.
        permission: Highest operation level allowed.
        sources: Allowlist of source names (empty = all sources).
        rate_limit_per_min: Token-bucket refill rate per minute. 0 = unlimited.
    """

    client_id: str
    name: str = ""
    permission: AgentPermission = AgentPermission.read
    sources: list[str] = Field(default_factory=list)
    rate_limit_per_min: int = 60


class AuthConfig(BaseModel):
    """Per-client identity and ACL settings.

    Args:
        enabled: When ``False``, all clients get ``default_permission`` with no
            source restrictions and no rate limiting.  Safe default for solo use.
        agents: List of named agents with explicit permissions.
        default_permission: Permission granted to unidentified clients when
            ``enabled`` is ``True``.
        default_rate_limit_per_min: Rate limit for unidentified clients.
    """

    enabled: bool = False
    agents: list[AgentConfig] = Field(default_factory=list)
    default_permission: AgentPermission = AgentPermission.read
    default_rate_limit_per_min: int = 120


class EmailSourceConfig(BaseModel):
    """Extra settings for mbox email sources.

    Args:
        max_messages: Maximum number of email messages to index per mbox file.
    """

    max_messages: int = 10_000


class IndexingConfig(BaseModel):
    """Controls optional enrichment passes run after normal chunking.

    Args:
        generate_summaries: When ``True`` *and* Ollama is available, the indexer
            generates a 2-3 sentence LLM summary for each document and stores it
            as a dedicated chunk (``chunk_type="summary"``) used only for retrieval
            and never surfaced directly in search results.
        dedup_enabled: When ``True``, near-duplicate chunks are dropped before
            storing.  Reduces index bloat when content is re-ingested or when
            connectors return overlapping documents.
        dedup_threshold: Cosine-similarity threshold above which two chunks are
            considered duplicates.  Default ``0.97`` is intentionally strict.
        pii_detection: When ``True``, run the PII filter over chunk text before
            indexing.  Detected entities are either redacted or block the chunk
            depending on ``pii_redact``.
        pii_redact: When ``True`` (and ``pii_detection`` is on), replace detected
            PII spans with ``[REDACTED]`` placeholders.  When ``False``, chunks
            that contain PII are skipped entirely.
    """

    generate_summaries: bool = False
    dedup_enabled: bool = False
    dedup_threshold: float = 0.97
    pii_detection: bool = False
    pii_redact: bool = True


class CLIPConfig(BaseModel):
    """Configuration for the CLIP image embedding provider.

    Args:
        model_name: Open-CLIP model architecture.
        pretrained: Pre-trained weights tag.
        device: ``"auto"`` selects CUDA when available, else CPU.
    """

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"


class WhisperConfig(BaseModel):
    """Configuration for the Whisper audio transcription provider.

    Args:
        model_size: Faster-whisper model size.
        device: ``"auto"`` selects CUDA when available, else CPU.
        language: ISO-639-1 code, or ``None`` for auto-detect.
    """

    model_size: str = "base"
    device: str = "auto"
    language: str | None = None


class MultimodalConfig(BaseModel):
    """Configuration for optional CLIP and Whisper multimodal providers.

    Args:
        clip: CLIP image embedding configuration.
        whisper: Whisper audio transcription configuration.
    """

    clip: CLIPConfig = Field(default_factory=CLIPConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)


class EncryptionConfig(BaseModel):
    """Configuration for Fernet symmetric encryption at rest.

    Args:
        enabled: When ``True``, audit log entries are encrypted.
        key_file: Path to the Fernet key file.
    """

    enabled: bool = False
    key_file: Path = Field(default_factory=lambda: Path("~/.memorymesh/secret.key").expanduser())


class ConnectorConfig(BaseModel):
    """Configuration entry for a single external data connector.

    Args:
        type: Connector type identifier, e.g. ``"jira"``, ``"slack"``.
        enabled: When ``False`` the connector is never run.
        config: Free-form dict of connector-specific settings; validated and
            passed to the connector's own ``*Config`` model at sync time.
    """

    type: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class MemoryMeshConfig(BaseModel):
    """Root configuration model for MemoryMesh.

    Mirrors the top-level keys of ``config.yaml``.  All fields have defaults
    so the daemon can boot without any config file present.
    """

    sources: list[SourceConfig] = Field(default_factory=list)
    global_ignore: list[str] = Field(default_factory=lambda: list(_DEFAULT_GLOBAL_IGNORE))
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    email: EmailSourceConfig = Field(default_factory=EmailSourceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    connectors: list[ConnectorConfig] = Field(default_factory=list)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)
