# MemoryMesh

[![PyPI](https://img.shields.io/pypi/v/memorymesh-mcp)](https://pypi.org/project/memorymesh-mcp/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/kilhubprojects/memory-mesh/ci.yml?label=CI)](https://github.com/kilhubprojects/memory-mesh/actions)
[![Tests](https://img.shields.io/badge/tests-CI%20validated-brightgreen)]()
[![v0.8.0](https://img.shields.io/badge/version-v0.8.0-orange)](https://github.com/kilhubprojects/memory-mesh/releases/tag/v0.8.0)

**SQLite for AI memory.** Persistent memory layer for MCP agents and coding copilots — local-first, zero cloud, works in 5 minutes.

<!-- demo GIF goes here -->
> **See it in action:** Claude Code remembering project decisions across sessions.

| 💻 **Coding copilots** | 🤖 **MCP agents** | 📚 **Research assistants** |
|---|---|---|
| Remember architecture decisions, bugs, and preferences across sessions | Persistent memory across any MCP-compatible client | Semantic recall over your notes, papers, and docs |

---

## Up and running in 5 minutes

```bash
pip install memorymesh-mcp
cp config.example.yaml ~/.memorymesh/config.yaml
# edit config.yaml — point at your folders
memorymesh index ~/Documents
memorymesh search "how did I configure the debounce"
```

Wire it into Claude Desktop. Find the config file at:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memorymesh": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/memory-mesh",
        "memorymesh", "start"
      ]
    }
  }
}
```

Restart Claude Desktop. The 15 tools appear automatically.

---

## Why it exists

Every AI conversation starts from zero. Claude doesn't know which architecture decision you made last week. Cursor doesn't remember the bug you fixed yesterday. The context dies when the session ends.

Mem0 requires a cloud account. Zep needs a running server and a database. LangMem ties you to the LangChain ecosystem. None of them speak MCP natively.

MemoryMesh runs entirely on your machine. It indexes your files into a local SQLite + ChromaDB store and exposes them through 15 MCP tools. It never touches the network unless you configure a connector. Any MCP client — Claude Desktop, Cursor, your own agent — gets persistent memory with one config change.

---

## How it works

The indexer watches your files, chunks them with format-aware parsers, and stores embeddings locally. The search engine fuses dense and sparse results, then a cross-encoder reranker scores the candidates.

```
                   ┌──────────────────────────────┐
  MCP clients ───▶ │         MemoryMesh           │
(Claude Desktop,   │  ┌────────────────────────┐  │
 Cursor, agents)   │  │ MCP Tools (FastMCP):   │  │
                   │  │  search_memory         │  │
                   │  │  list_sources          │  │
                   │  │  get_document          │  │
                   │  │  index_now             │  │
                   │  └──────────┬─────────────┘  │
                   │             ▼                 │
                   │     Search Engine             │
                   │   dense + BM25 → RRF          │
                   │             │                 │
                   │   ┌─────────┴──────────┐      │
                   │   ▼                    ▼      │
                   │ ChromaDB            BM25      │
                   │ (embeddings)     (sparse)     │
                   │   ▲                    ▲      │
                   │   └──────── Indexer ───┘      │
                   │                ▲              │
                   │           Watchdog            │
                   └────────────────┬──────────────┘
                                    ▼
                             Your filesystem
```

**Indexing:** file watcher detects changes → SHA-256 dedup skips unchanged files → parser (txt/md/pdf/docx/code/obsidian/email/calendar/browser) → chunker (tree-sitter for code, by-heading for markdown, recursive for text) → `sentence-transformers` embeddings → ChromaDB + BM25.

**Search:** query → query expansion (lexical variants + HyDE) → parallel dense + sparse search → Reciprocal Rank Fusion (k=60) → `bge-reranker-v2-m3` reranker → top-k results with path, preview, score, and metadata.

**RAG (optional):** `ask_memory` → `search_memory` retrieval → Ollama `generate()` → grounded answer with cited sources.

---

## What's included

MemoryMesh ships with hybrid search (dense embeddings + BM25 + RRF + cross-encoder reranker), hot/warm/cold memory tiers with configurable forgetting decay, and an episodic event timeline. 47 connectors pull data from Jira, Notion, GitHub, Slack, email, browser history, Spotify, and more. 15 MCP tools expose everything to any MCP-compatible client. A real-time file watcher re-indexes changed files within seconds, without a manual trigger.

<details>
<summary>Full feature list</summary>

| Feature | Status |
|---|---|
| Local file indexing (txt, md, code, pdf, docx) | ✅ |
| Obsidian vault parser (frontmatter + wikilinks) | ✅ |
| Notion HTML export parser | ✅ |
| AI conversation exports (Claude, ChatGPT JSON) | ✅ |
| Email indexing (`.mbox` via stdlib) | ✅ |
| Calendar indexing (`.ics` / iCalendar) | ✅ |
| Browser history (Chrome / Firefox / Brave SQLite) | ✅ |
| Hybrid search — dense + BM25 + RRF | ✅ |
| Cross-encoder reranker (`bge-reranker-v2-m3`) | ✅ |
| Query expansion — lexical variants + HyDE | ✅ |
| RAG with local LLM via Ollama (`ask_memory` tool) | ✅ |
| Multi-vector summary indexing for abstract recall | ✅ |
| MCP server — 15 tools, stdio + streamable-http | ✅ |
| Real-time incremental indexing (watchdog + debounce) | ✅ |
| Tree-sitter code chunking (Python, JS, TS, Go, Rust…) | ✅ |
| Parent Document Retriever (`extended_preview`) | ✅ |
| Cross-platform — Windows / Linux / macOS | ✅ |
| Post-crash reconciliation | ✅ |
| Optional OCR for scanned PDFs (Tesseract / EasyOCR) | ✅ |
| Privacy audit log (query hashes only, no cleartext) | ✅ |
| GitHub Actions CI (Ubuntu / Windows / macOS) | ✅ |
| Docker + docker-compose | ✅ |
| Per-agent permission layer (ACL + rate limiting + revocation) | ✅ |
| Hierarchical memory (hot / warm / cold tiers + forgetting policy) | ✅ |
| Episodic memory timeline (`query_timeline`, `record_event` tools) | ✅ |
| Memory control tools (`pin_memory`, `forget_memory`) | ✅ |
| Embedding LRU cache (`CachedEmbeddingProvider`) | ✅ |
| Health endpoint (`GET /health` on :8766) | ✅ |
| Real CLIP image embeddings (`memorymesh[multimodal]`) | ✅ |
| Real Whisper audio transcription (`memorymesh[multimodal]`) | ✅ |
| Knowledge Graph — entity co-occurrence (`/graph`, `graph_memory` tool) | ✅ |
| Encryption at rest (Fernet AES-128, `memorymesh keygen`) | ✅ |
| REST API (11 endpoints at `/api`, OpenAPI docs at `/api/docs`) | ✅ |
| VS Code extension (`extensions/vscode/`) | ✅ |
| Browser extension — Manifest V3 (`extensions/browser/`) | ✅ |
| 47 data source connectors (Jira, Notion, GitHub, Slack, Spotify…) | ✅ |
| Unit + integration test suite | ✅ |

</details>

---

## MCP Tools

Once running, these tools are available to any MCP-compatible client:

| Tool | Description |
|---|---|
| `search_memory(query, top_k, mode, source)` | Hybrid search over all indexed content. Returns path, preview, score, file type, source, and optional `extended_preview` for wider context. |
| `list_sources()` | List all configured sources with file counts and index status. |
| `get_document(path, max_bytes)` | Read the full content of an indexed file (up to 1 MB by default). |
| `index_now(path)` | Force immediate re-index of a file or directory, bypassing the watcher. |
| `ask_memory(question, top_k, model)` | RAG: retrieves relevant passages and sends them to a local Ollama model for a grounded answer. Requires Ollama running locally. |
| `pin_memory(chunk_id)` | Pin a chunk to the hot tier — never demoted, never score-decayed. |
| `forget_memory(chunk_id)` | Suppress a chunk from future search results without deleting the source file. |
| `query_timeline(since_days, event_type, limit)` | Query the episodic event log: what was retrieved / indexed in the last N days? |
| `sync_source(source_type, dry_run)` | Pull and index documents from a configured external connector (Jira, Notion, GitHub…). |
| `get_entity(name, entity_type)` | Look up a named entity (person, project, concept) and its associated chunk IDs. |
| `related_documents(path, top_k, exclude_self)` | Find documents semantically similar to the given file path. |
| `search_by_date(since_days, until_days, source, limit)` | Search indexed chunks by last-modified date range. |
| `forget_source(source, dry_run)` | Remove all indexed data for a named source from the index. |
| `summarize_source(source, max_chunks)` | Generate a brief summary of the most recent content in a source (requires Ollama). |
| `graph_memory(min_mentions, entity_type)` | Return the entity co-occurrence knowledge graph as nodes and edges. |

All tools are backwards-compatible — new fields are added without changing existing signatures.

---

## How MemoryMesh compares

How MemoryMesh compares to similar projects:

| Feature | MemoryMesh | LangChain | LlamaIndex | PrivateGPT | AnythingLLM | MemGPT | Haystack |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MCP native** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hybrid search (dense + BM25 + RRF)** | ✅ | Partial | Partial | ❌ | ❌ | ❌ | ✅ |
| **Real-time watcher + SHA-256 dedup** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Post-crash reconciliation** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **100% local, zero telemetry** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cross-platform (Win/Linux/Mac)** | ✅ | ✅ | ✅ | Partial | Partial | ✅ | ✅ |
| **No framework dependency** | ✅ | — | — | ❌ | ❌ | ❌ | — |
| **Per-agent permissions** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**MCP native** means it was built for MCP from day one — not bolted on after. The 15 tools follow additive versioning — new fields are added without removing existing ones.

**Per-agent permissions** means per-client identity, ACL by source and operation, token-bucket rate limiting, and token revocation are built into the core — not added as middleware.

---

## Configuration

Everything lives in `config.yaml`. See [`config.example.yaml`](./config.example.yaml) for a fully commented reference. Key highlights:

```yaml
sources:
  - name: documents
    path: ~/Documents
    recursive: true
    extensions: [.txt, .md, .pdf, .docx]

  - name: projects
    path: ~/Projects
    recursive: true
    extensions: [.py, .js, .ts, .go, .rs, .md]

  - name: obsidian
    path: ~/obsidian-vault
    source_type: obsidian     # activates wikilink + frontmatter parser

  - name: emails
    path: ~/Mail
    source_type: email        # parses .mbox files

embeddings:
  model: all-MiniLM-L6-v2    # swap to paraphrase-multilingual-MiniLM-L12-v2 for PT/EN

search:
  default_top_k: 10
  hybrid:
    enabled: true
  reranker:
    enabled: true             # cross-encoder reranker (recommended)
    model: BAAI/bge-reranker-v2-m3
  query_expansion:
    enabled: true
    n_lexical_variants: 1

# Optional: local LLM for ask_memory tool + HyDE query expansion
ollama:
  enabled: false              # set true after: ollama pull llama3
  model: llama3

server:
  transport: stdio            # stdio | streamable-http
```

**Global ignore list** protects sensitive paths by default: `.env`, `*.key`, `id_rsa*`, `secrets/`, `.ssh/`, `.aws/`, `.git/`, `node_modules/`.

---

## Benchmarks

> Benchmarks will be published after CI runs across all three platforms. The goal is reproducible numbers — not "fast on my machine."

Scripts are in `benchmarks/` and runnable locally:

- `bench_indexing.py` — indexing throughput (chunks/s, MB/s) on a synthetic corpus
- `bench_search_latency.py` — p50/p95/p99 search latency across hybrid/dense/sparse modes
- `bench_embedding_models.py` — speed vs. quality comparison across three embedding models

---

## Privacy & security

Three commitments that do not change across versions:

1. **No data leaves your machine.** No telemetry. No external API calls unless you explicitly opt in — and even then, there is a `WARNING` in the log.
2. **HTTP listener binds to `127.0.0.1` only** by default. Exposing to other interfaces requires an explicit config override.
3. **Logs never contain document content or queries in cleartext.** The audit log records query *hashes*, not queries.

Encryption at rest is available as of v0.8.0. Run `memorymesh keygen` to generate a key, then enable `encryption.enabled: true` in `config.yaml`. The SQLite metadata store can be exported as an encrypted backup with `memorymesh backup`.

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| **v0.1** | Core: hybrid search, 4 MCP tools, stdio transport, indexer | ✅ shipped |
| **v0.2** | CI/CD, Parent Document Retriever, Docker, security hardening | ✅ shipped |
| **v0.3** | Reranker, query expansion + HyDE, RAG (Ollama), 6 new parsers, eval framework | ✅ shipped |
| **v0.5** | Per-agent permissions (ACL/rate-limit/revocation), hot/warm/cold tiers, episodic timeline, memory control tools, embedding cache, health endpoint, CLIP/Whisper stubs | ✅ shipped |
| **v0.8** | Real CLIP+Whisper, Knowledge Graph, Encryption at rest, REST API (11 endpoints), VS Code + browser extensions, 47 connectors, 15 MCP tools | ✅ shipped |
| **v1.0** | Agent OS integration — memory layer for multi-agent systems | ~6 months |
| **v2.0** | Hardware agents — ESP32/Arduino querying the hub over BLE/WiFi | ~12 months |

Full details in [`ROADMAP.md`](./ROADMAP.md).

---

## Troubleshooting

- **`UnicodeDecodeError` on a text file** — MemoryMesh tries UTF-8, UTF-8 BOM, cp1252, latin-1 in order. If a file still fails, it is logged and skipped.
- **Watcher doesn't fire on a network drive / WSL mount** — set `watcher.use_polling: true` in `config.yaml`.
- **Tesseract not found** — install it system-wide and ensure it is in `PATH`. Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki).
- **Embedding model mismatch after changing config** — run `memorymesh reindex --all`. The CLI refuses to start if the model ID stored in ChromaDB does not match the config.

---

## Contributing

Contributions are welcome — bug reports, new connectors, integration examples, and documentation improvements all help. Open an issue to discuss before submitting a large PR.

---

## Acknowledgements

Architecture informed by studying [LlamaIndex](https://github.com/run-llama/llama_index), [LangChain](https://github.com/langchain-ai/langchain), [PrivateGPT](https://github.com/zylon-ai/private-gpt), [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm), [MemGPT](https://github.com/cpacker/MemGPT), and [Haystack](https://github.com/deepset-ai/haystack) — understanding what each does well and what it does not. And to [chroma-mcp](https://github.com/chroma-core/chroma-mcp) and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for showing what MCP-native looks like in practice.

---

MIT. See [`LICENSE`](./LICENSE).
