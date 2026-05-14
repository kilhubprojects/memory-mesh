# ROADMAP.md — MemoryMesh

Visão de longo prazo, do MVP ao "where this all leads".
Última atualização: 2026-05-13. Baseado em análise competitiva de 14 projetos
similares + auto-crítica do código + papers recentes (2024-2025) sobre retrieval
e memória de agentes.

---

## Histórico de versões

| Versão | Foco | Status |
|---|---|---|
| **v0.1** | Core: busca híbrida, 4 MCP tools, transporte stdio, indexer, reconciliação pós-crash | ✅ entregue |
| **v0.2** | CI/CD, Parent Document Retriever, Docker, segurança | ✅ entregue |
| **v0.3** | Reranker, query expansion + HyDE, RAG (Ollama), 6 novos parsers, eval framework | ✅ entregue |
| **v0.5** | Permissões por agente (ACL/rate-limit/revogação), hot/warm/cold tiers, episodic timeline, memory control tools, embedding cache, health endpoint, CLIP/Whisper | ✅ entregue |
| **v0.8** | CLIP+Whisper reais, Knowledge Graph, Encryption at rest, REST API (11 endpoints), extensões VS Code + browser, 47 conectores, 15 MCP tools | ✅ entregue |
| **v1.0** | Agent OS integration — camada de memória para sistemas multi-agente | ~6 meses |
| **v2.0** | Hardware agents — ESP32/Arduino consultando o hub via BLE/WiFi | ~12 meses |

---

## Decisões arquiteturais travadas (2026-05-03)

Decisões fechadas durante o planejamento de v0.5. Não reabrir sem motivo forte.

**Stack final em uma linha:** Ollama + ChromaDB (→ LanceDB quando bater ceiling) +
(all-MiniLM default, BGE-M3 first-class) + sem sync entre máquinas + sem plugin
system externo + dashboard opt-in + single-user multi-agent.

| # | Decisão | Resolução |
|---|---|---|
| D1 | LLM local | Ollama default + abstração `LLMProvider` ABC. llama.cpp como segundo provider em v0.9/v1.0 — não bloqueia release principal. |
| D2 | Vector store | ChromaDB default + LanceDB como segundo provider quando necessário. Qdrant rejeitado — requer servidor externo, incompatível com local-first. ABC `VectorStore` mantém abertura futura. |
| D3 | Embedding default | all-MiniLM default (90 MB, leve no primeiro boot) + BGE-M3 first-class para usuário multilingual ou PT-BR via 1 linha no config. CLI `memorymesh model recommend` detecta idioma e sugere modelo. |
| D4 | Sync entre máquinas | NÃO em v0.x. Adia para v1.0 ou nunca. Princípio: construir para dor real, não imaginária. |
| D5 | Plugin system formal | NÃO em v0.x. Provider pattern interno apenas. Plugin system externo → v1.0 (Agent OS vai precisar). |
| D6 | Web UI | Dashboard mínimo HTML opt-in (`server.dashboard.enabled: true`), servido em `/dashboard`. Zero framework JS. Full web UI → v1.0 se demanda real surgir. |
| D7 | Multi-tenancy multi-usuário | NÃO em v0.x. Modelo: 1 daemon = 1 humano, N agentes. Multi-user → v1.0 se fizer sentido lá. |

---

## Análise competitiva — o que adotamos

| Projeto | O que foi adotado |
|---|---|
| **Haystack** | Eval framework rigoroso (Precision@k, Recall@k, MRR, NDCG@k, métricas custom) |
| **LlamaIndex** | Multi-vector retriever; auto-merging retriever; técnicas de retrieval (sem dependência) |
| **LangChain** | Parent Document Retriever (`extended_preview`); Self-Query Retriever |
| **PrivateGPT** | Fluxo `ask_memory` com Ollama; RAG 100% offline |
| **AnythingLLM** | Ingestor de chat histories (Claude.ai / ChatGPT JSON exports) |
| **Letta (MemGPT)** | Memória hierárquica formal (hot/warm/cold); self-editing memory patterns |
| **Cognee** | Knowledge graph opt-in com entity extraction via LLM local |
| **mcp-memory-service** | Decay/forgetting por score temporal; forgetting policy configurable |
| **HyDE** (paper 2022) | Gera doc hipotético → embeda → busca. Recall +20% em queries vagas. Implementado com Ollama opt-in. |
| **HippoRAG** (paper 2024) | Personalized PageRank sobre grafo de entidades — base do `graph_memory` tool |
| **LEANN** | PQ compression de embeddings (planejado v0.9) |

**Embedding providers suportados além do default (all-MiniLM):**
- BGE-M3 (BAAI) — multilingual + dense + sparse + multi-vector num modelo. Recomendado para PT-BR.
- Nomic Embed v2 — open-weights, contexto longo (8k tokens).
- Jina v3 — multilingual, instruction-tuned.
- OpenAI text-embedding-3 — opt-in cloud (warning de privacidade).
- Voyage AI — opt-in cloud (warning de privacidade).

---

## v1.0 — Agent OS integration (~6 meses)

> "Um sistema operacional para agentes de IA — permissões, scheduling,
> comunicação inter-agente."

MemoryMesh entra como **camada de memória/dados** do Agent OS (projeto separado).

- [ ] Spec versionada da API MCP do MemoryMesh — semver. Breaking changes só em majors.
- [ ] Long-running query subscriptions — agente se inscreve em "todo email novo do meu chefe".
- [ ] Time-aware queries — "o que mudou hoje?", "qual era meu top-of-mind essa semana?".
- [ ] Memory consolidation — job periódico que sumariza e arquiva dados antigos.
- [ ] Graph layer expandido — entidades com edges tipados, traversal queries, PageRank.
- [ ] Plugin system externo — outros projetos extendem MemoryMesh sem fork.
- [ ] Sync entre máquinas opt-in — se demanda real surgir antes do v2.0.
- [ ] llama.cpp como segundo LLM provider (além de Ollama).
- [ ] LanceDB como segundo vector store (além de ChromaDB).

---

## v2.0 — Hardware agent framework (~12 meses)

> "Conectar LLMs a Arduino, ESP32, sensores, atuadores via abstração padronizada."

MemoryMesh como **memória persistente compartilhada** de agentes embarcados.

- [ ] Embedding remoto — dispositivo envia texto, MemoryMesh embeda + busca + retorna chunks.
- [ ] Wire format compacto — protobuf ou MessagePack opcional para links lentos (BLE, LoRa).
- [ ] Sync entre máquinas — laptop indexa, robô consulta via WiFi local.
- [ ] Storage tiering embarcado — chunks "quentes" no dispositivo, "frios" no hub.

Esse milestone fecha o triângulo:
**dados pessoais (MemoryMesh) ↔ agentes (Agent OS) ↔ corpo físico (Hardware Framework)**.

---

## Arquitetura alvo v0.5+ (ASCII)

```
                    ┌─────────────────────────────────┐
                    │   Claude Desktop / Cursor / etc │
                    │   (MCP clients)                  │
                    └──────────────┬──────────────────┘
                                   │ MCP (stdio | streamable-http)
                                   ▼
                ┌──────────────────────────────────────────┐
                │       MemoryMesh MCP Server              │
                │  ┌────────────────────────────────────┐  │
                │  │ Tools: search, ask, list, get,     │  │
                │  │   index, pin, forget, timeline,    │  │
                │  │   sync, entity, graph, date...     │  │
                │  └────────────────────────────────────┘  │
                │  ┌────────────────────────────────────┐  │
                │  │ Auth: identity, ACL, rate_limit,   │  │
                │  │   revocation                       │  │
                │  └────────────────────────────────────┘  │
                └──────────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  ┌───────────┐            ┌───────────────┐         ┌────────────────┐
  │ Search    │            │   Indexer     │         │   Memory       │
  │  Engine   │            │  + Watcher    │         │  Manager       │
  │           │            │               │         │  (tiers,       │
  │ - Dense   │            │ - SHA-256     │         │   episodic,    │
  │ - BM25    │            │ - Backpress.  │         │   semantic,    │
  │ - RRF     │            │ - Reconcile   │         │   forgetting)  │
  │ - Rerank  │            │ - CLIP/Whisper│         │                │
  │ - HyDE    │            │               │         │                │
  │ - Multi-  │            │               │         │                │
  │   vector  │            │               │         │                │
  └─────┬─────┘            └───────┬───────┘         └────────┬───────┘
        │                          │                          │
        ▼                          ▼                          ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    Storage Layer                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
  │  │ Chroma   │  │ BM25     │  │ SQLite   │  │ Embedding    │ │
  │  │ (vectors)│  │ (sparse) │  │ (meta +  │  │ Cache        │ │
  │  │ + tiers  │  │          │  │  audit + │  │ (sha→vec)    │ │
  │  │          │  │          │  │  events) │  │              │ │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
  └───────────────────────────────────────────────────────────────┘
        ▲
        │ feed
  ┌─────┴─────────────────────────────────────────────────────────┐
  │                    Sources                                     │
  │  files | mbox | ics | browser | bookmarks | chats             │
  │  | obsidian | notion | images (CLIP) | audio (Whisper)        │
  │  | 47+ conectores externos                                     │
  └───────────────────────────────────────────────────────────────┘
        ▲
        │ opcional
  ┌─────┴───────────────────────┐
  │  Ollama (LLM local)         │
  │  Usado por: ask_memory,     │
  │   HyDE, query_expansion,    │
  │   sumarização, entity       │
  │   extraction.               │
  │  Não obrigatório pra boot.  │
  └─────────────────────────────┘
```

---

## Critérios de qualidade — referência v0.5

Metas que definiram a versão "definitiva" do hub standalone. Servem de baseline
para v1.0 — nada pode regredir.

| Categoria | Métrica | Alvo |
|---|---|---|
| Performance | Search p50 (100k chunks) | < 100ms |
| Performance | Search p95 (100k chunks) | < 250ms |
| Performance | Indexação de 1k arquivos médios | < 60s |
| Qualidade | NDCG@10 (eval set pessoal) | > 0.85 |
| Qualidade | Recall@10 | > 0.90 |
| Cobertura de testes | Global | > 85% |
| Cobertura de testes | Módulos críticos (storage, search, indexer, auth) | > 90% |
| Cross-platform | CI verde Win/Linux/Mac × Py 3.11/3.12/3.13 | 100% |
| Privacidade | Logs com cleartext de documento | 0 ocorrências |
| Privacidade | Telemetria default | desligada |
| Bugs | Critical issues abertos | 0 |
| Documentação | Toda public API documentada | 100% |

---

## Princípios imutáveis

1. **Local-first.** Sempre. Toda feature que requer rede é opt-in com warning explícito.
2. **Privacidade como precondição.** Default seguro mesmo para usuário que nunca leia docs.
3. **Sem breaking changes nas MCP tools.** As ferramentas do v0.1 funcionam no v2.0.
4. **Cross-platform real.** Toda PR roda em Windows/Linux/Mac na CI.
5. **Daemon não pode crashar.** Regra do v0.1 preservada em todo código novo.
6. **Logs nunca contêm conteúdo de documento.** Invariante de privacidade.
7. **Abstrações limpas > otimizações prematuras.** `EmbeddingProvider` plugável é mais
   importante que economizar 5% de RAM.
8. **Documentação é parte do produto.** Cada milestone tem doc atualizado antes do merge.

---

## O que foi descartado (escopo fechado)

| Feature | Motivo do descarte |
|---|---|
| Web UI proprietária completa | Claude Desktop, Cursor, Cline já resolvem; duplica esforço |
| FAISS como vector store | ChromaDB aguenta ~5M chunks; troca é prematura |
| Gravar tela/áudio (Rewind-style) | Viola privacidade como precondição |
| LangChain/LlamaIndex como dependência | Código próprio é superior nas partes críticas |
| Pinecone / cloud vector stores | Local-first não negocia |
| WhatsApp/iMessage | Stores encriptados; acesso confiável inviável cross-platform |
| Mobile clients (iOS/Android nativo) | Fora do escopo v0.x |
| Cloud sync entre máquinas | Adia para v1.0 ou nunca — princípio: dor real primeiro |
| Plugin system externo | → v1.0 |
| Multi-tenancy multi-usuário | 1 daemon = 1 humano, N agentes. Multi-user → v1.0 |
| LLM próprio embarcado | Ollama resolve sem a manutenção de um runtime |
| Speech in/out tempo real | TTS/STT online violam local-first |
| Hardware integration (Arduino/ESP32) | → v2.0 |
| Agent OS features (scheduling, inter-agent comm) | → v1.0 (projeto separado) |
| Qdrant como vector store | Requer servidor externo — incompatível com local-first |
