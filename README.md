# MemoryMesh

> Universal MCP hub for personal data. Local-first, private by default, designed to be the memory layer of the agents you'll build next.

[![CI](https://img.shields.io/github/actions/workflow/status/kilhubprojects/memory-mesh/ci.yml?label=CI)](https://github.com/kilhubprojects/memory-mesh/actions)
[![PyPI](https://img.shields.io/pypi/v/memorymesh-mcp)](https://pypi.org/project/memorymesh-mcp/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20streamable--http-purple)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/tests-CI%20validated-brightgreen)]()
[![v0.8.0](https://img.shields.io/badge/version-v0.8.0-orange)](https://github.com/kilhubprojects/memory-mesh/releases/tag/v0.8.0)

MemoryMesh indexes your local files — and in future versions, your emails, calendar, browser history, and chat logs — and exposes them through the [Model Context Protocol](https://modelcontextprotocol.io/). Any MCP-aware client (Claude Desktop, Cursor, Claude Code, or your own agent) can ask semantic questions over the things you actually own, without sending a single byte to the cloud.

It is a **hub**, not a single-purpose RAG. The transport, embedding model, parser, and chunking strategy are all swappable behind clean interfaces — so the same hub can grow from "search my notes" to "remember everything for my Agent OS."

---

## Why this exists

Personal data is fragmented across dozens of apps. No AI agent can reach all of it in a unified, private way. Anthropic's MCP defined the protocol; MemoryMesh fills the gap of the hub that wires everything up — locally, with privacy as a precondition rather than a setting.

---

## How it works

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

**Indexing pipeline:** file watcher detects changes → SHA-256 dedup skips unchanged files → parser (txt/md/pdf/docx/code/obsidian/notion/conversations/email/calendar/browser) → smart chunker (tree-sitter for code, by-heading for markdown, recursive for text) → embeddings via `sentence-transformers` → upsert into ChromaDB + BM25 index → optional Ollama summary chunk for abstract recall.

**Search pipeline:** query → optional query expansion (lexical variants + HyDE) → parallel dense search (ChromaDB) + sparse search (BM25) over-fetch → Reciprocal Rank Fusion (k=60) → cross-encoder reranker (`bge-reranker-v2-m3`) → top-k results with path, preview, score, and metadata.

**RAG pipeline (optional):** `ask_memory` tool → `search_memory` retrieval → context assembly → Ollama `generate()` → grounded answer with cited sources.

---

## What makes it different

Most comparable tools pick one dimension to optimize. MemoryMesh targets all of them simultaneously:

| Feature | MemoryMesh | LangChain | LlamaIndex | PrivateGPT | AnythingLLM | MemGPT | Haystack |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MCP native** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hybrid search (dense + BM25 + RRF)** | ✅ | Partial | Partial | ❌ | ❌ | ❌ | ✅ |
| **Real-time watcher + SHA-256 dedup** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Post-crash reconciliation** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **100% local, zero telemetry** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cross-platform (Win/Linux/Mac)** | ✅ | ✅ | ✅ | Partial | Partial | ✅ | ✅ |
| **No framework dependency** | ✅ | — | — | ❌ | ❌ | ❌ | — |
| **Multi-agent ready (ACL, rate limits, per-client identity)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**MCP native** means it was built for MCP from day one — not bolted on after. The 15 tools (`search_memory`, `list_sources`, `get_document`, `index_now`, `ask_memory`, `pin_memory`, `forget_memory`, `query_timeline`, `sync_source`, `get_entity`, `related_documents`, `search_by_date`, `forget_source`, `summarize_source`, `graph_memory`) follow additive versioning — new fields are added without changing existing signatures.

**Multi-agent ready** means per-client identity, ACL by source and operation, token-bucket rate limiting, and token revocation are built into the core — not added as middleware. The same architecture supports hardware agents (ESP32, Arduino) querying the hub. See [Roadmap](#roadmap).

---

## Status

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

---

## Quickstart

> **Prerequisite:** Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
# Install from PyPI
pip install memorymesh-mcp
```

Or clone for development:

```bash
# Clone and install
git clone https://github.com/kilhubprojects/memory-mesh.git
cd memory-mesh
uv sync

# Copy and edit config — point it at the folders you want indexed
cp config.example.yaml ~/.memorymesh/config.yaml
$EDITOR ~/.memorymesh/config.yaml

# Index a folder
uv run memorymesh index ~/Documents

# Test a search
uv run memorymesh search "how did I configure the debounce"
```

### Enable RAG with Ollama (optional)

```bash
# Install Ollama from https://ollama.ai, then pull a model
ollama pull llama3

# Enable in config.yaml:
#   ollama:
#     enabled: true
#     model: llama3

# Now ask questions answered from your own documents
uv run memorymesh search "summarise my architecture notes"
# ... or use the ask_memory MCP tool in Claude Desktop
```

### Run as daemon (real-time indexing)

```bash
uv run memorymesh start --transport streamable-http --detach
uv run memorymesh status
# edit a file in one of your sources — it gets indexed within ~2s
uv run memorymesh search "the sentence you just typed"
uv run memorymesh stop
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `search_memory(query, top_k, mode, source)` | Hybrid search over all indexed content. Returns path, preview, score, file type, source, and optional `extended_preview` for wider context. |
| `list_sources()` | List all configured sources with file counts and index status. |
| `get_document(path, max_bytes)` | Read the full content of an indexed file (up to 1 MB by default). |
| `index_now(path)` | Force immediate re-index of a file or directory, bypassing the watcher. |
| `ask_memory(question, top_k, model)` | RAG: retrieves relevant passages and sends them to a local Ollama model for a grounded answer. Requires Ollama running locally. |
| `pin_memory(chunk_id)` | Pin a chunk to the hot tier — never demoted, never score-decayed. |
| `forget_memory(chunk_id)` | Force a chunk to cold tier, suppressing its relevance on future searches without deleting it. |
| `query_timeline(since_days, event_type, limit)` | Query the episodic event log: what was retrieved / indexed in the last N days? |
| `sync_source(source_type, dry_run)` | Pull and index documents from a configured external connector (Jira, Notion, GitHub…). |
| `get_entity(name, entity_type)` | Look up a named entity (person, project, concept) and its associated chunk IDs. |
| `related_documents(path, top_k, exclude_self)` | Find documents semantically similar to the given file path. |
| `search_by_date(since_days, until_days, source, limit)` | Search indexed chunks by last-modified date range. |
| `forget_source(source, dry_run)` | Remove all indexed data for a named source from the index. |
| `summarize_source(source, max_chunks)` | Generate a brief LLM summary of the most recent content in a source. |
| `graph_memory(min_mentions, entity_type)` | Return the entity co-occurrence knowledge graph as nodes and edges. |

All tools are backwards-compatible — new fields are added without changing existing signatures.

### Wire it into Claude Desktop

Add to your Claude Desktop config:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

**stdio (simplest — Claude Desktop spawns MemoryMesh):**

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

**Daemon mode (pre-started HTTP server — faster cold start):**

```bash
# Start the daemon once
uv run memorymesh start --transport streamable-http --port 8765 --detach
```

```json
{
  "mcpServers": {
    "memorymesh": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Restart Claude Desktop after editing. The tools appear automatically.

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

> Benchmarks will be published here after v0.2 lands CI across all three platforms. The goal is reproducible numbers — not "fast on my machine."

Scripts are already in `benchmarks/` and runnable locally:

- `bench_indexing.py` — indexing throughput (chunks/s, MB/s) on a synthetic corpus
- `bench_search_latency.py` — p50/p95/p99 search latency across hybrid/dense/sparse modes
- `bench_embedding_models.py` — speed vs. quality comparison across three embedding models

---

## Privacy & security

Three hard commitments that do not change across versions:

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

- **`UnicodeDecodeError` on a text file** — MemoryMesh tries UTF-8, UTF-8 BOM, cp1252, latin-1 in order. If a file still fails, it is logged and skipped, not crashed.
- **Watcher doesn't fire on a network drive / WSL mount** — set `watcher.use_polling: true` in `config.yaml`.
- **Tesseract not found** — install it system-wide and ensure it is in `PATH`. Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki).
- **Embedding model mismatch after changing config** — run `memorymesh reindex --all`. The CLI refuses to start if the model ID stored in ChromaDB does not match the config.

---

## About

MemoryMesh is an independent open-source project. Contributions, bug reports, and
architectural feedback are welcome via GitHub Issues.

---

## Contributing

Contributions are welcome. Please open an issue to discuss the change before submitting a pull request.

---

## License

MIT. See [`LICENSE`](./LICENSE).

---

## Acknowledgements

Architecture informed by studying [LlamaIndex](https://github.com/run-llama/llama_index), [LangChain](https://github.com/langchain-ai/langchain), [PrivateGPT](https://github.com/zylon-ai/private-gpt), [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm), [MemGPT](https://github.com/cpacker/MemGPT), and [Haystack](https://github.com/deepset-ai/haystack) — understanding what each does well and what it does not. And to [chroma-mcp](https://github.com/chroma-core/chroma-mcp) and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for showing what MCP-native looks like in practice.
