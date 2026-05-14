# ARCHITECTURE.md — MemoryMesh

Arquitetura técnica do MVP. Toda decisão aqui já foi travada (ver `MVP_SCOPE.md` para o "porquê"). Tudo é **modular e abstraído** porque a versão pós-MVP vai trocar componentes (embeddings melhores, novos parsers, novos transports).

---

## 1. Visão de alto nível

```
                          ┌──────────────────────────────┐
                          │   MCP Clients                │
                          │  Claude Desktop / Cursor /   │
                          │  Claude Code / Agent OS      │
                          └──────────────┬───────────────┘
                                         │ stdio  /  streamable-http
                                         ▼
        ┌──────────────────────────────────────────────────────────┐
        │                   MemoryMesh Server                       │
        │                                                           │
        │   ┌────────────────────────────────────────────────┐     │
        │   │            FastMCP App  (server/app.py)         │     │
        │   │                                                 │     │
        │   │  Tools:  search_memory  list_sources            │     │
        │   │          get_document   index_now  ask_memory   │     │
        │   └─────────────────────┬───────────────────────────┘     │
        │                         │                                  │
        │                         ▼                                  │
        │   ┌────────────────────────────────────────────────┐     │
        │   │        Search Engine  (search/engine.py)        │     │
        │   │   query expansion (lexical + HyDE)              │     │
        │   │   dense (Chroma)  +  sparse (BM25)  →  RRF      │     │
        │   │   cross-encoder reranker (bge-reranker-v2-m3)   │     │
        │   └─────────────────────┬───────────────────────────┘     │
        │                         │                                  │
        │   ┌─────────────────────┼───────────────────────────┐     │
        │   ▼                     ▼                            ▼     │
        │ ┌──────────┐  ┌────────────────┐  ┌──────────────────┐    │
        │ │ Chroma   │  │ Metadata SQLite│  │ BM25 in-memory   │    │
        │ │ vectors  │  │ paths/hash/mt  │  │ (rebuild on boot)│    │
        │ └──────────┘  └────────────────┘  └──────────────────┘    │
        │      ▲                ▲                       ▲           │
        │      └────────────────┼───────────────────────┘           │
        │                       │                                   │
        │   ┌───────────────────┴────────────────────────┐         │
        │   │       File Indexer  (indexer/file_indexer) │         │
        │   │  parse → chunk → embed → store             │         │
        │   └────────────────────┬───────────────────────┘         │
        │                        │                                  │
        │   ┌────────────────────┴───────────────────────┐         │
        │   │       Watchdog Watcher (indexer/watcher)    │         │
        │   │   debounced events → enqueue indexer        │         │
        │   └────────────────────┬───────────────────────┘         │
        └────────────────────────┼─────────────────────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │   File system    │
                       │   ~/Documents,   │
                       │   ~/Projects, …  │
                       └──────────────────┘
```

---

## 2. Modos de execução

Hibrido configurável (decidido com você):

### 2.1. On-demand (stdio)

- MCP client (Claude Desktop, Cursor) faz `spawn` do processo via stdio.
- Servidor inicia rápido, **não** roda o watcher (economia de RAM).
- Lê o índice existente e serve queries.
- Ao desconectar, processo morre.

### 2.2. Daemon (streamable-http)

- `memorymesh start` sobe processo persistente.
- Watcher ativo: indexação incremental em tempo real.
- Listener HTTP em `127.0.0.1:8765` (default).
- Lock file em `~/.memorymesh/daemon.pid` para impedir múltiplas instâncias.

### 2.3. Compartilhamento de código

Toda a lógica de indexer, search, storage, parsing é **transport-agnostic**. Os dois modos compartilham 95% do código. A diferença é só:

- Que tools são registradas (todas, sempre).
- Se o `WatcherService` é iniciado.
- Que transport é passado para `mcp.run(transport=...)`.

---

## 3. Abstrações principais

### 3.1. `EmbeddingProvider` (ABC)

```python
class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...      # serve pra invalidar índice em caso de troca

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
```

Implementação MVP: `SentenceTransformersProvider` (default `all-MiniLM-L6-v2`, configurável).

Pós-MVP: `BGE_M3_Provider`, `OllamaEmbeddingProvider`, `OpenAIEmbeddingProvider` (opt-in, viola privacidade — fica desabilitado por default).

**Regra crítica:** o `model_id` é gravado nos metadados do Chroma collection. Se trocar de modelo, o boot detecta mismatch e exige reindexação (com confirmação no CLI).

### 3.2. `Parser` (ABC)

```python
class Parser(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument: ...
```

`ParsedDocument` carrega texto + metadata estrutural opcional (headings pra md, função/classe pra código).

Implementações:
- `TextParser` (txt/md/código), `PdfParser`, `DocxParser` — parsers base
- `ObsidianParser` — `.md` com YAML frontmatter + `[[wikilinks]]` → backlinks
- `NotionParser` — HTML exports do Notion, extrai hierarquia de headings
- `ConversationParser` — JSON do Claude.ai e ChatGPT (auto-detect)
- `EmailParser` — `.mbox` via stdlib, prefere `text/plain`, cap de mensagens
- `CalendarParser` — `.ics` via `icalendar`, extrai VEVENT
- `BrowserHistoryParser` — `History` (Chrome) e `places.sqlite` (Firefox) via cópia segura
- `OcrFallbackParser` — wrapper que ativa só se `ocr.enabled=true` E parsing nativo retornou texto vazio

### 3.3. `Chunker` (ABC)

```python
class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...
```

Implementações MVP:
- `RecursiveTextChunker` — fallback. Usa separadores `["\n\n", "\n", ". ", " "]`.
- `MarkdownChunker` — divide por heading H1/H2/H3 preservando contexto pai.
- `CodeChunker` — usa `tree-sitter` pra cortar em fronteiras de função/classe/método. Linguagens MVP: Python, JS/TS, Java, C, C++, Rust, Go.

`ChunkerRegistry` faz o roteamento `extensão → chunker`.

### 3.4. `VectorStore` (ABC)

ChromaDB é o default, mas a interface permite trocar (LanceDB no futuro).

```python
class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[ChunkWithEmbedding]) -> None: ...

    @abstractmethod
    def delete_by_path(self, path: str) -> None: ...

    @abstractmethod
    def search(self, query_vec: list[float], top_k: int, filter_: dict | None = None) -> list[SearchHit]: ...
```

### 3.5. `SparseIndex` (BM25)

Implementação MVP: `rank_bm25.BM25Okapi`. Persistido como pickle em `~/.memorymesh/bm25.pkl`. Reconstruído em boot se hash do corpus mudou.

---

## 4. Fluxo de dados

### 4.1. Indexação (caminho feliz)

```
1. Watcher detecta event ou indexação inicial varre o diretório.
2. Indexer recebe Path.
3. Lê metadata atual do SQLite. Se hash(arquivo_atual) == hash(armazenado) → skip.
4. Calcula novo hash SHA-256.
5. Roteia para Parser apropriado pelo MIME/extensão.
6. Parser retorna ParsedDocument.
7. Chunker apropriado divide em Chunks.
8. EmbeddingProvider.embed_documents(chunks_text) em batches.
9. VectorStore.upsert(chunks_com_embedding).
10. SparseIndex registra os mesmos chunks.
11. MetadataStore atualiza (path, hash, mtime, n_chunks, status='indexed', updated_at).
12. Log INFO: "Indexed <path> in <X>ms (<N> chunks)".
```

### 4.2. Indexação (erros)

Cada step entre 5 e 11 tem try/except específico:

| Step | Falha possível | Ação |
|------|----------------|------|
| 5 | MIME desconhecido | log.warning + status='unsupported' + skip |
| 6 | PDF corrompido, encoding ruim | log.warning + status='parse_error' + skip + métrica |
| 8 | OOM no batch | reduzir batch e retry; persistir falha se ainda falhar |
| 9 | Disco cheio | log.error + parar daemon (CRITICAL) |
| 10 | Índice esparso corrompido | rebuild on next boot |

### 4.3. Busca (caminho feliz)

```
1. MCP tool search_memory(query, top_k, filters?) ou ask_memory(question, top_k).
2. QueryExpander.expand(query):
    a. lexical_variants = stem/synonym variants (n_lexical_variants=1 default)
    b. hyde_query = OllamaClient.generate(hyde_prompt) se ollama.enabled=true
    c. queries = [query] + lexical_variants + ([hyde_query] se habilitado)
3. SearchEngine._retrieve_single() para cada query em paralelo (ThreadPoolExecutor):
    a. dense_hits = vector_store.search(embed_query(q), top_k=top_k * over_fetch_factor)
    b. sparse_hits = bm25.search(q, top_k=top_k * over_fetch_factor)
    c. Filtra chunk_index == -1 (summary chunks nunca aparecem nos resultados)
4. _fuse_all(): RRF(k=60) sobre todos os hit_lists de todas as queries
5. CrossEncoderReranker.rerank(fused[:top_k_before_rerank]) → re-ordena
6. fused = fused[:top_k]
7. Para cada hit, expande com preview (200 chars) + path + score.
   Se parent_window_chars > 0, adiciona extended_preview.
8. Audit log: query_hash, n_results, latency. (Não loga o conteúdo da query.)
9. Retorna lista pro MCP client.

Para ask_memory:
10. Monta context_str com os top hits.
11. OllamaClient.generate(rag_prompt + context_str + question) → answer.
12. Retorna {answer, sources, model, ollama_available}.
```

### 4.4. Watching incremental

```
- watchdog.observers.Observer com PollingObserver como fallback (Windows + drives de rede).
- Handler bufferiza eventos por path em janela de 1.5s (debounce).
- Quando a janela fecha:
    - created/modified → enfileira indexação.
    - deleted → vector_store.delete_by_path + metadata_store.mark_deleted.
    - moved → delete + index_new_path.
- Filtros aplicados antes da fila: ignore patterns (.git, node_modules, .env, *.key, id_rsa, etc.)
- Worker assíncrono consome a fila com concorrência configurável (default 2).
```

---

## 5. Schema das MCP tools

### 5.1. `search_memory`

```python
@mcp.tool()
async def search_memory(
    query: str,
    top_k: int = 10,
    file_types: list[str] | None = None,        # [".py", ".md", ...]
    sources: list[str] | None = None,           # ["~/Projects", ...] — match por prefixo
    date_from: str | None = None,               # ISO 8601
    date_to: str | None = None,
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
) -> SearchResponse:
    """
    Busca semântica (+ keyword) sobre todos os dados indexados.

    Returns:
        SearchResponse with list of hits, each containing:
            path, chunk_index, score, preview (≤200 chars),
            file_type, mtime, source_root.
    """
```

### 5.2. `list_sources`

```python
@mcp.tool()
async def list_sources() -> SourcesReport:
    """
    Lista todas as fontes monitoradas e estatísticas de indexação.

    Returns:
        SourcesReport with per-source:
            path, recursive, extensions, n_files_indexed,
            n_files_pending, n_files_errored, last_scan_at,
            total_chunks, disk_size_bytes.
    """
```

### 5.3. `get_document`

```python
@mcp.tool()
async def get_document(file_path: str, max_bytes: int = 200_000) -> DocumentResponse:
    """
    Retorna o conteúdo de um documento indexado.

    Lê DIRETO do filesystem (não do índice) — permite ver versão atual.
    Truncado em max_bytes pra evitar dump enorme no contexto.

    Raises:
        DocumentNotFoundError: path não está em nenhuma source.
        DocumentTooLargeError: arquivo > max_bytes (retorna início + sumário).
    """
```

### 5.4. `index_now`

```python
@mcp.tool()
async def index_now(path: str, recursive: bool = True) -> IndexResponse:
    """
    Força reindexação de um path específico.

    Valida que o path está dentro de alguma source configurada.
    Retorna estatísticas: n_files_processed, n_chunks, duration_ms.
    """
```

### 5.5. `ask_memory`

```python
@mcp.tool()
async def ask_memory(
    question: str,
    top_k: int = 5,
    model: str | None = None,
) -> AskMemoryResponse:
    """
    RAG com LLM local via Ollama.

    Fluxo: search_memory(question, top_k) → monta context_str →
    OllamaClient.generate(prompt) → answer com fontes citadas.

    Requer `ollama.enabled: true` no config e Ollama rodando localmente.
    Quando Ollama não está disponível, retorna as fontes (igual search_memory)
    com hint de instalação — nunca quebra.

    Returns:
        AskMemoryResponse:
            answer: str | None,
            sources: list[SearchHit],
            model: str,
            ollama_available: bool,
            hint: str | None  # só quando unavailable
    """
```

---

## 6. Schema do `config.yaml`

```yaml
# MemoryMesh — config example

sources:
  - name: documents               # opcional, identificador legível
    path: ~/Documents
    recursive: true
    extensions: [.txt, .md, .pdf, .docx, .json]
    ignore:                       # globs adicionais
      - "**/Backup/**"
      - "**/*.tmp"
  - name: projects
    path: ~/Projects
    recursive: true
    extensions: [.py, .js, .ts, .rs, .go, .md, .yaml, .toml]
  - name: notes
    path: ~/Notes
    recursive: true
    # se extensions omitido, usa defaults (ver supported_extensions)

# Patterns globais sempre ignorados
global_ignore:
  - "**/.git/**"
  - "**/node_modules/**"
  - "**/__pycache__/**"
  - "**/.venv/**"
  - "**/venv/**"
  - "**/.env"
  - "**/.env.*"
  - "**/*.key"
  - "**/*.pem"
  - "**/id_rsa*"
  - "**/secrets/**"
  - "**/.ssh/**"
  - "**/.aws/**"
  - "**/dist/**"
  - "**/build/**"
  - "**/target/**"

embeddings:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2
  device: auto                    # auto | cpu | cuda | mps
  batch_size: 32
  normalize: true                 # cosine ↔ dot product equiv

chunking:
  default:
    strategy: recursive
    chunk_size: 512                # tokens (aprox via tiktoken)
    chunk_overlap: 50
  markdown:
    strategy: by_heading
    max_chunk_size: 800
  code:
    strategy: tree_sitter
    max_chunk_size: 1024
    fallback: recursive            # se linguagem não suportada

ocr:
  enabled: false
  backend: tesseract               # tesseract | easyocr
  languages: [eng, por]
  trigger: empty_text_only         # só roda OCR se parser nativo retornou vazio
  max_file_size_mb: 50

search:
  default_top_k: 10
  hybrid:
    enabled: true
    rrf_k: 60                      # constante do reciprocal rank fusion
    over_fetch_factor: 3           # busca top_k*3 em cada índice antes de fundir
  reranker:
    enabled: false                 # placeholder pós-MVP

storage:
  base_dir: ~/.memorymesh
  vector_db_subdir: chroma_db
  metadata_db: metadata.sqlite3
  bm25_pickle: bm25.pkl
  audit_log: audit.jsonl

server:
  transport: stdio                 # stdio | streamable-http
  http:
    host: 127.0.0.1                # NUNCA 0.0.0.0 sem opt-in explícito
    port: 8765

watcher:
  enabled: true                    # ignored em modo on-demand
  debounce_ms: 1500
  worker_concurrency: 2
  use_polling: false               # true para drives de rede / WSL mounts

logging:
  level: INFO
  format: pretty                   # pretty | json
  file: ~/.memorymesh/logs/memorymesh.log
  rotation: 10 MB
  retention: 14 days
```

---

## 7. Layout do estado em disco

```
~/.memorymesh/
├── config.yaml                  # symlink ou cópia, opcional
├── daemon.pid
├── chroma_db/                   # collection pasta gerenciada pelo Chroma
├── metadata.sqlite3             # tabelas: files, sources, chunks_index
├── bm25.pkl
├── audit.jsonl                  # query_id, ts, n_results, latency_ms (sem query)
└── logs/
    └── memorymesh.log
```

### Schema SQLite (`metadata.sqlite3`)

```sql
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    mtime REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    status TEXT NOT NULL,           -- indexed | parse_error | unsupported | deleted
    error_message TEXT,
    indexed_at REAL NOT NULL,
    embedding_model_id TEXT NOT NULL
);

CREATE INDEX idx_files_source ON files(source_name);
CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_mtime ON files(mtime);

CREATE TABLE sources (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    recursive INTEGER NOT NULL,
    last_full_scan_at REAL
);
```

---

## 8. CLI — comandos

```bash
memorymesh init               # cria ~/.memorymesh, config.example.yaml na raiz do projeto
memorymesh start              # sobe daemon (streamable-http) em foreground
memorymesh start --detach     # daemon em background (usa pidfile)
memorymesh stop               # SIGTERM no daemon
memorymesh status             # mostra: pid, uptime, n_files, n_chunks, last_indexed
memorymesh index <path>       # indexa um arquivo/pasta on-demand (sem precisar do daemon)
memorymesh reindex --all      # apaga índice e reconstrói tudo (com confirmação)
memorymesh info               # versão, modelo de embedding, tamanho do índice
memorymesh search "<query>"   # busca direta (debug, bypassa MCP)
memorymesh serve --stdio      # entrypoint pro Claude Desktop spawnar
```

---

## 9. Logging & Audit

- Logs operacionais: `~/.memorymesh/logs/memorymesh.log` (rotacionado).
- Audit log de queries: `~/.memorymesh/audit.jsonl` — uma linha por query MCP. Campos: `ts`, `tool`, `query_hash` (sha256 da query truncado a 16 chars — não recuperável), `n_results`, `latency_ms`, `client_id_if_known`.
- Logs **nunca** contêm texto de documento, contúdo de chunks ou queries em claro.

---

## 10. Pontos de extensão pós-MVP (já preparados na arquitetura)

| Extensão | Onde encaixa |
|---|---|
| Reranker (cross-encoder) | `search/engine.py` aceita um `Reranker` opcional após RRF |
| Email source | Novo `Parser` + novo tipo de `Source` em `config.py` |
| Calendar source | Idem |
| Browser history | Idem (parser de SQLite do Chrome/Firefox) |
| OpenAI embeddings | Novo `EmbeddingProvider`, opt-in com warning de privacidade |
| Per-agent permissions | Middleware no `server/app.py` que valida `client_id` contra ACL |
| Encryption at rest | `EncryptionManager` usa Fernet (AES-128-CBC + HMAC-SHA256) para exportar backups cifrados do SQLite via `MetadataStore.export_encrypted()` e para o audit log. O ChromaDB não é cifrado em disco; para proteção do índice vetorial use disk-level encryption (LUKS, FileVault, BitLocker). |
| Multi-modal embeddings (text+image) | Novo `Parser` que retorna texto+imagem + `EmbeddingProvider` multimodal |

---

## 11. Não-objetivos do MVP (importante)

- ❌ Cloud, sync entre máquinas, backup remoto.
- ❌ UI gráfica.
- ❌ Multi-usuário com auth.
- ❌ Telemetria de uso.
- ❌ Plugin system de terceiros.
- ❌ Reranker, query rewrite, LLM local.
- ❌ Knowledge graph.
- ❌ APIs externas (Gmail, Notion, etc.).
- ❌ Suporte a Office Online / Google Docs.

Tudo isso pode entrar na v2 quando o MVP estiver sólido.
