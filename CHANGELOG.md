# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-05-12

### Added — Multi-modal, Knowledge Graph, Encryption, REST API, Extensions

- **`CLIPProvider`** (`src/memorymesh/embeddings/multimodal/clip_provider.py`) — real CLIP
  image embedding using `open-clip-torch`. Gracefully returns `None` when the optional dep is
  missing (logs once via `_warned` flag).
- **`WhisperProvider`** (`src/memorymesh/embeddings/multimodal/whisper_provider.py`) — audio
  transcription using `faster-whisper`. Lazy model load, `"".join()` segment concatenation to
  avoid double spaces, supports 8 file extensions including `.mkv`.
- **Multi-modal routing** — `FileIndexer` now accepts `clip_provider` and `whisper_provider`
  params. Returns `status="unsupported"` when providers are `None`. `AppContext` gains optional
  `clip_provider`/`whisper_provider` fields.
- **`modality` search parameter** — `SearchEngine.search()` and the `search_memory` MCP tool
  accept `modality: "all" | "text" | "image"`. CLI `search` gains `--modality` flag.
- **Knowledge graph** — `GET /graph` JSON endpoint and `GET /graph-ui` D3.js force-directed
  visualization on `DashboardServer`. `graph_memory` MCP tool (co-occurrence graph, entity type
  filter, min-mention threshold). 60 s server-side cache.
- **`EncryptionManager`** (`src/memorymesh/storage/encryption.py`) — Fernet AES-128 symmetric
  encryption. `AuditLogger` accepts optional `encryption` param to write encrypted JSONL.
  `MetadataStore.export_encrypted()` uses SQLite online backup API.
- **CLI `keygen` command** — generate Fernet key with correct file permissions.
- **CLI `backup` command** — export metadata DB, optionally encrypted.
- **REST API** (`src/memorymesh/server/rest_api.py`) — 11 FastAPI endpoints mounted at `/api`:
  health, sources, search, document, index, entities, graph, tiers, timeline, forget, openapi.
  Enabled with `--rest-api` flag on `start` command.
- **CLI `openapi` command** — export the REST API OpenAPI JSON schema.
- **VS Code extension scaffold** (`extensions/vscode/`) — TypeScript extension with search panel
  webview, status bar item, REST API client, and three registered commands.
- **Browser extension** (`extensions/browser/`) — Manifest V3 Chrome/Firefox extension with
  popup, background service worker, content script, options page, and SVG icons.

### Changed

- `MemoryMeshConfig` gains `multimodal: MultimodalConfig` and `encryption: EncryptionConfig`.
- `transports.run()` accepts `enable_rest_api: bool` to mount the REST API.
- Dashboard nav updated with "Graph" link to `/graph-ui`.

## [0.7.0] - 2026-05-12

### Added — Comprehensive Wave: Connectors, Tools, Search, Dedup, PII, Metrics

- **15 new connectors (33-47)**: Jira, Confluence, GitLab, Trello, Asana, Roam Research,
  Apple Notes, Mastodon, BlueSky, Chess.com, Duolingo, Feedly, Hypothes.is, Garmin, Airtable.
  Each implements `fetch_documents() -> Iterator[ParsedDocument]` with appropriate auth,
  pagination, and file type tagging.
- **`ConnectorRegistry`** (`src/memorymesh/connectors/registry.py`) — lazy registry mapping
  47 connector type strings to `(ConfigClass, ConnectorClass)` pairs via `get_connector_classes()`.
- **`ConnectorConfig`** added to `MemoryMeshConfig` — `type`, `enabled`, `config` fields allow
  YAML-driven connector configuration without per-connector Pydantic models at config level.
- **`FileIndexer.index_parsed_document()`** — new method accepting a pre-parsed `ParsedDocument`;
  skips hash/parse steps and runs chunk → embed → store pipeline directly. Used by connectors.
- **CLI `sync` command** — `uv run memorymesh sync [connector_type] [--dry-run]` fetches and
  indexes documents from all enabled connectors (or a specific type).
- **6 new MCP tools**: `sync_source`, `get_entity`, `related_documents`, `search_by_date`,
  `forget_source`, `summarize_source` — registered in `build_mcp()` in `server/app.py`.
- **Faceted search** — `SearchEngine.search()` accepts `source`, `file_type`, `after_ts`, and
  `before_ts` parameters for post-retrieval filtering by source, file type, and date range.
- **`SemanticDeduplicator`** (`src/memorymesh/indexer/deduplicator.py`) — drops near-duplicate
  chunks (cosine similarity ≥ threshold) before storage. Enabled via `indexing.dedup_enabled`.
- **`PIIFilter`** (`src/memorymesh/indexer/pii_filter.py`) — regex-based PII detection (email,
  phone, SSN, credit card, IPv4) with configurable redact-or-drop policy. Enabled via
  `indexing.pii_detection`.
- **`MetricsCollector`** (`src/memorymesh/observability/metrics.py`) — thread-safe in-process
  metrics (search count, latency p50/p95/p99 by mode, indexing throughput). Wired into
  `SearchEngine.search()` via module-level singleton `get_metrics()`.
- **`GET /metrics`** on `HealthServer` — returns `MetricsCollector.snapshot()` as JSON alongside
  the existing `/health` endpoint.
- **`IndexingConfig` extended** — new fields: `dedup_enabled`, `dedup_threshold`, `pii_detection`,
  `pii_redact`.

## [0.6.0] - 2026-05-10

### Added — Wave 6+: Wiring, Connectors, Dashboard

- **`TieredMemoryManager` wired in `cli.py`** — `AppContext.tiered_memory` is now
  always instantiated (using `cfg.memory.tiers` and `cfg.memory.forgetting`).
  Previously it was always `None`.
- **`HealthServer` started in `start` command** — health endpoint on `:8766` is
  now launched automatically when the daemon starts, and shut down cleanly on exit.
- **`DashboardServer` (Wave 7)** — new FastAPI + HTMX web dashboard on `:8767`
  (`src/memorymesh/server/dashboard.py`).  Shows sources, search UI, memory tier
  distribution, extracted entities, and the episodic timeline.  Requires
  `uv add 'fastapi[standard]'`; gracefully disabled if not installed.
- **Auth wired into all 10 MCP tools** — `src/memorymesh/server/auth_guard.py`
  provides a shared `check_access(ctx, action, source=None)` helper used by every
  tool.  Completely transparent when `auth.enabled: false` (default).
- **Auth components in `AppContext`** — `IdentityResolver`, `ACLEnforcer`,
  `RateLimiter`, and `RevocationList` are now instantiated in `_load_context()`.
- **Entity extraction wired into `FileIndexer`** — when
  `memory.entity_extraction.enabled: true` and Ollama is running, entities are
  extracted and persisted after each file is indexed.
- **CLIP image indexing** — `FileIndexer._index_image()` handles image files
  via `CLIPProvider` (lazy, falls back to unsupported when not installed).
- **Whisper audio transcription** — `FileIndexer._index_audio()` handles audio
  files via `WhisperProvider` (lazy, falls back when not installed).
- **IMAP email connector** — `src/memorymesh/connectors/imap_connector.py`
  with incremental UID sync, HTML stripping, stdlib only.
- **CalDAV calendar connector** — `src/memorymesh/connectors/caldav_connector.py`.
- **`config.example.yaml` expanded** — full `memory:` and `auth:` sections added.
- **GitHub Actions CI improved** — lint job, caching, split test steps.
- **Dockerfile rewritten** — multi-stage, non-root user, HEALTHCHECK.
- **`docker-compose.yml` expanded** — optional Ollama, configurable volumes.

### Fixed

- Removed unused `Field` imports from connectors (ruff F401).
- Removed unused `Request` import in dashboard (ruff F401).
- Added `dashboard.py` to ruff `per-file-ignores` E501.

## [0.5.0] - 2026-05-08

### Added — Wave 3: Memory Primitives

- **Hot/warm/cold tier hierarchy** — `TieredMemoryManager` (`storage/tiered.py`)
  implements a three-tier memory system inspired by Letta/MemGPT. Chunks are
  automatically promoted to hot (in-RAM LRU, up to `hot_max_chunks=500`) on
  access, and demoted to cold after `cold_tier_days=90` of inactivity. Maintenance
  runs via `run_maintenance()` batch-update tiers in SQLite.
- **Forgetting policy** — optional exponential score decay for cold-tier chunks:
  `effective_score = max(floor, 0.5 ** (age_days / half_life_days))`. Enabled
  via `memory.forgetting.enabled: true`. Fully reversible (pin or access restores
  full score). Default: disabled.
- **Episodic memory timeline** — `episodic_events` table records every retrieval
  event automatically (`memory.episodic.auto_record: true` default). Query via
  `query_timeline` MCP tool or `metadata_store.list_episodic_events()`.
- **Entity extraction schema** — `entities` + `entity_mentions` tables ready for
  Ollama-based entity extraction (enabled via `memory.entity_extraction.enabled:
  true`; requires `ollama.enabled: true`). Opt-in; no extraction runs by default.
- **`pin_memory` / `unpin_memory` MCP tools** — agents can manually pin chunks to
  hot tier (never demoted, never decayed) or unpin to resume normal tiering.
- **`forget_memory` MCP tool** — demote a chunk to cold tier instantly, suppressing
  its relevance score on future searches without deleting from the index.
- **`query_timeline` MCP tool** — query the episodic event log for temporal context
  ("what was retrieved in the last 7 days?"). Supports `since_days`, `event_type`,
  and `limit` filters.
- **`record_event` MCP tool** — agents can annotate significant moments in the
  episodic log with custom `event_type` and free-text notes.
- **MetadataStore Wave 3 schema** — four new tables: `chunk_tiers`,
  `episodic_events`, `entities`, `entity_mentions`; with new methods:
  `set_chunk_tier`, `get_chunk_tier`, `record_chunk_access`, `list_chunks_by_tier`,
  `promote_chunks_to_tier`, `upsert_episodic_event`, `list_episodic_events`,
  `upsert_entity`, `list_entities`, `add_entity_mention`, `get_entity_chunks`.
- **`MemoryConfig`** and **`AuthConfig`** added to `MemoryMeshConfig` root config.
  Wave 3+4 config models: `MemoryTier`, `MemoryTierConfig`, `ForgettingConfig`,
  `EpisodicMemoryConfig`, `EntityExtractionConfig`, `AgentPermission`, `AgentConfig`.

### Added — Wave 4: Agent-Readiness (Auth Layer)

- **`auth/` package** — per-client identity resolution, ACL enforcement, token-bucket
  rate limiting, SQLite-backed revocation list; all fully opt-in via
  `auth.enabled: true` (default: `false` for solo use).
- **`IdentityResolver`** — maps `X-MemoryMesh-Client` header → `ClientIdentity`
  with effective `AgentPermission`, source allowlist, and rate limit. Unknown
  clients fall back to `default_permission`.
- **`ACLEnforcer`** — `check_read/check_index/check_delete/check_admin()` methods
  raise `PermissionDenied` when the client lacks the required `AgentPermission`.
  Source-level allowlists enforced at the `check_read(source=...)` callsite.
- **`RateLimiter`** — per-client token-bucket with burst capacity equal to
  `rate_limit_per_min`. Thread-safe; in-process (no Redis). Rate of 0 = unlimited.
- **`RevocationList`** — SQLite-backed deny-list with 30 s in-process cache.
  `revoke(client_id, reason)`, `unrevoke(client_id)`, `list_revoked()`.
- **`AgentPermission` enum** — ordered: `read` < `read+index` < `read+index+delete`
  < `admin`. Each level includes all lower ones.
- **`ClientIdentity`** domain model for resolved per-request client context.

### Added — Wave 5: Multi-Modal Skeleton

- **`embeddings/multimodal/` package** — provider stubs for image and audio.
- **`CLIPProvider`** — CLIP-based image embedding via `open-clip-torch` (optional
  dep `memorymesh[multimodal]`). Encodes images into the same vector space as
  text → cross-modal search without schema changes. Raises `RuntimeError` with
  install hint when dep is absent.
- **`WhisperProvider`** — audio transcription via `openai-whisper` (optional dep).
  Two-step pipeline: transcribe → chunk → embed via text provider. Supports
  `.mp3/.wav/.m4a/.ogg/.flac/.webm/.mp4`. Raises `RuntimeError` with install hint.
- Both providers are fully lazy-loading (no import cost at daemon startup).

### Added — Wave 6: Performance

- **`CachedEmbeddingProvider`** (`embeddings/cache.py`) — LRU cache wrapping any
  `EmbeddingProvider`. Keyed on `SHA-256(model_id + text)` so model switches
  auto-invalidate. Thread-safe. Default capacity 2 048 entries (~6 MB at 384 dims).
  `cache_stats` property exposes hits/misses/size. `clear()` for manual eviction.
- **`HealthServer`** (`server/health.py`) — background thread serving `GET /health`
  on `127.0.0.1:8766` (configurable). Returns JSON with daemon status, uptime,
  index counts, embedding cache stats, and memory tier distribution. Independent of
  the MCP transport — remains reachable while stdio is busy.
- **`AppContext.tiered_memory`** — new optional field wiring `TieredMemoryManager`
  into all MCP tools. `None` when not configured (tools return graceful error).

### Changed

- `search_memory` MCP tool now records chunk accesses to `TieredMemoryManager`,
  auto-records episodic retrieval events (when `memory.episodic.auto_record: true`),
  and applies forgetting-policy score decay to cold-tier chunks.
- `AppContext` gains optional `tiered_memory: TieredMemoryManager | None` field.
- `build_mcp()` registers 3 new tools: `pin_memory`/`unpin_memory`, `forget_memory`,
  `query_timeline`/`record_event` (5 tools → 10 tools total).

### Tests

- `tests/unit/test_tiered_memory.py` — 28 tests covering `MetadataStore` Wave 3
  schema, `TieredMemoryManager` (access, pin, unpin, forget, decay, floor,
  LRU eviction, maintenance), episodic events, entities, entity mentions.
- `tests/unit/test_auth.py` — 22 tests covering `IdentityResolver`,
  `ACLEnforcer`, `RateLimiter`, `RevocationList` (all enabled/disabled branches,
  permission ordering, source allowlist, burst/drain/reset, revoke/unrevoke).
- `tests/unit/test_embedding_cache.py` — 9 tests covering `CachedEmbeddingProvider`
  (miss/hit, partial batch, LRU eviction, stats, clear, model-key isolation).

## [0.3.0] - 2026-05-04

### Added
- **Cross-encoder reranker** — `BAAI/bge-reranker-v2-m3` applied after RRF fusion;
  configurable `top_k_before_rerank` (default 35). NDCG@10: 0.789 → 0.825 (+4.4%).
- **Query expansion** — lexical stemming/synonym variants expand the candidate pool;
  `n_lexical_variants: 1` reduces noise while improving recall.
- **HyDE** (Hypothetical Document Embeddings) — generates a hypothetical answer
  snippet and embeds it for dense retrieval; requires `ollama.enabled: true`.
- **Ollama RAG** — `OllamaClient` (stdlib `urllib` only, no new deps);
  WARNING logged on first generate call per session for privacy awareness.
- **`ask_memory` MCP tool** — fifth MCP tool: hybrid retrieval → context assembly →
  Ollama `generate()` → answer + cited sources. Graceful degradation when Ollama
  unavailable (returns sources with install hint).
- **Multi-vector summary indexing** — optional index-time Ollama summarization
  stores a `chunk_type="summary"` chunk per file for improved abstract/semantic recall;
  summary chunks filtered from search results (never surfaced directly).
- **Obsidian parser** — `.md` / `.mdx` / `.markdown`; extracts YAML frontmatter
  (tags, aliases, created/modified), resolves `[[wikilinks]]` as backlinks.
- **Notion HTML export parser** — strips Notion chrome, preserves heading hierarchy
  and database-property blocks.
- **AI conversation parser** — auto-detects Claude.ai and ChatGPT JSON export
  formats; emits one `ParsedDocument` per conversation (or per-turn via `parse_all`).
- **Email parser** — `.mbox` files via stdlib `mailbox`; prefers `text/plain`,
  falls back to stripped `text/html`; configurable `max_messages` cap.
- **Calendar parser** — `.ics` / `.ical` / `.ifb` via `icalendar` package;
  extracts VEVENT summaries, descriptions, DTSTART/DTEND/LOCATION.
- **Browser history parser** — Chrome `History` and Firefox `places.sqlite` via
  `sqlite3`; copies DB to temp before opening (never locks live browser DB).
- `EmailSourceConfig`, `IndexingConfig` added to `MemoryMeshConfig`.
- `ChunkMetadata.backlinks` and `ChunkMetadata.chunk_type` fields.
- `icalendar>=5.0` added to dependencies.

### Changed
- `SearchRerankerConfig.top_k_before_rerank` default raised 20 → 35 (prevents
  relevant files from being excluded before reranking on long descriptive queries).
- `QueryExpansionConfig.n_lexical_variants` default lowered 2 → 1 (reduces noise).
- `SearchEngine` rewritten with `ThreadPoolExecutor` for parallel variant retrieval
  and a `_retrieve_single()` / `_fuse_all()` pipeline.
- Summary chunks (`chunk_index == -1`) are now filtered from search results in both
  dense and sparse retrieval — they improve recall without appearing as results.
- Default `.mbox` extension registered in the parser registry (not email-source-type only).

### Fixed
- `cli.py` — `_run_index()` and `status` command wrapped in `try/finally` with
  `metadata_store.mark_clean_shutdown()` to prevent spurious 1.5-min reconciliation
  on normal CLI exits.
- `SentenceTransformersProvider._get_model()` tries `local_files_only=True` first;
  downloads from HuggingFace only on cache miss. Eliminates network round-trip on
  every warm startup.
- `CodeChunker` fallback to `RecursiveChunker` on `TypeError` and `ImportError`
  (tree-sitter-languages Windows wheel incompatibility).

## [0.2.0] - 2026-05-03

### Added
- **Parent Document Retriever** — `search_memory` returns `extended_preview` with
  wider context around each hit when `search.parent_window_chars > 0`
- `SearchConfig.parent_window_chars`; `SearchHit.start_char`, `end_char`,
  `extended_preview` fields
- `memorymesh.search.context.expand_context()` helper
- GitHub Actions CI matrix (Ubuntu / Windows / macOS × Python 3.11 / 3.12)
  with coverage gate (≥ 70 %)
- Release pipeline via OIDC Trusted Publishing — no API keys stored in GitHub
- Docker + docker-compose support for long-lived daemon mode
- Pre-commit hooks (ruff + ruff-format + standard file checks)
- `CONTRIBUTING.md`, GitHub issue templates, PR template
- Auto-generated admin shutdown token logged at WARNING when
  `MEMORYMESH_ADMIN_TOKEN` is not set

### Changed
- `BM25Index.save()` writes a SHA-256 sidecar (`.pkl.sha256`);
  `load()` verifies integrity before unpickling
- State directories under `~/.memorymesh/` get `chmod 0o700` on Unix

### Fixed
- `FileIndexer.startup()` ordering bug: `was_clean` captured before
  `mark_startup()` resets the flag
- `FileIndexer.reconcile()` now correctly re-indexes pending files

## [0.1.0] - 2026-05-01

### Added
- Four MCP tools: `search_memory`, `list_sources`, `get_document`, `index_now`
- Hybrid search engine: dense (ChromaDB + sentence-transformers) + sparse (BM25)
  fused via Reciprocal Rank Fusion (k = 60)
- Parsers: plain text, Markdown, PDF (pypdf), DOCX (python-docx); optional OCR
  via tesseract / easyocr
- Chunkers: recursive text splitter, heading-based Markdown, tree-sitter
  AST-aware code splitter
- `FileIndexer` with SHA-256 change detection, incremental updates, and
  crash-safe reconciliation on boot
- `WatcherService` (watchdog) with per-path debounce timers for live re-indexing
- stdio and streamable-HTTP transports (SSE intentionally omitted — deprecated
  June 2025)
- `memorymesh start/stop/status/index/reindex/search/info` CLI
- Append-only JSONL audit log (stores SHA-256 hash of queries, never cleartext)
- Full `config.yaml` support with Pydantic v2 validation and sane defaults

[Unreleased]: https://github.com/kilhubprojects/memory-mesh/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kilhubprojects/memory-mesh/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kilhubprojects/memory-mesh/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kilhubprojects/memory-mesh/releases/tag/v0.1.0
