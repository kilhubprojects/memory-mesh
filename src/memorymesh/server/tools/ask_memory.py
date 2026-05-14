"""MCP tool: ask_memory.

Searches the indexed knowledge base and generates an answer using a local LLM
via Ollama.  When Ollama is unavailable the tool still returns the top-k source
passages and a helpful installation hint - it never fails silently.

Requires ``ollama.enabled: true`` in ``config.yaml`` and a running Ollama
process with at least one model pulled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from loguru import logger
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.llm.ollama_client import OllamaClient
    from memorymesh.server.app import AppContext

# Maximum number of characters from each source passage included in the prompt.
_MAX_CONTEXT_CHARS_PER_HIT = 800

# System prompt preamble for the RAG call.
_RAG_SYSTEM_PREAMBLE = (
    "You are a helpful assistant answering questions about the user's personal "
    "knowledge base. Use only the provided source passages to answer. "
    "If the answer is not in the sources, say so clearly. "
    "Cite the source file paths when relevant."
)


def _build_rag_prompt(question: str, hits: list[dict]) -> str:
    """Construct the RAG prompt from the question and retrieved passages.

    Args:
        question: The user's original question.
        hits: List of hit dicts (``path``, ``preview``) from ``search_memory``.

    Returns:
        Fully formatted prompt string ready to send to Ollama.
    """
    parts: list[str] = [_RAG_SYSTEM_PREAMBLE, "", "SOURCES:", ""]
    for i, hit in enumerate(hits, 1):
        preview = str(hit.get("preview", ""))[:_MAX_CONTEXT_CHARS_PER_HIT]
        path = hit.get("path", "")
        parts.append(f"[{i}] {path}\n{preview}")
        parts.append("")

    parts += ["QUESTION:", question, "", "ANSWER:"]
    return "\n".join(parts)


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``ask_memory`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """
    # Retrieve the OllamaClient from ctx (added in C2); gracefully handle
    # older AppContext that does not yet have the attribute.
    ollama_client: OllamaClient | None = getattr(ctx, "ollama_client", None)

    @mcp.tool()
    def ask_memory(
        question: Annotated[str, "Natural-language question to answer from the knowledge base."],
        top_k: Annotated[int, "Number of source passages to retrieve before answering (1-20)."] = 5,
        model: Annotated[
            str | None,
            "Ollama model name override. Omit to use the default configured model.",
        ] = None,
    ) -> dict:
        """Search the knowledge base and generate an answer using a local LLM.

        Retrieves the top-k most relevant passages via hybrid search, builds a
        RAG prompt, and calls the local Ollama LLM to generate an answer.

        When Ollama is unavailable the ``answer`` field is ``null`` and a
        ``hint`` field explains how to install Ollama and pull a model.

        Args:
            question: The question to answer.
            top_k: How many passages to retrieve (clamped to 1-20).
            model: Optional model override.

        Returns:
            Dict with keys ``answer``, ``sources``, ``model``, ``ollama_available``.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        top_k = max(1, min(20, top_k))

        try:
            response = ctx.engine.search(question, top_k=top_k, mode="hybrid")
            hits = [
                {
                    "path": h.path,
                    "chunk_index": h.chunk_index,
                    "score": h.score,
                    "preview": h.preview,
                    "file_type": h.file_type,
                    "source": h.source_root,
                }
                for h in response.hits
            ]
        except Exception as exc:
            logger.warning(f"ask_memory: search failed: {exc}")
            hits = []

        ollama_available = False
        if ollama_client is not None and ctx.config.ollama.enabled:
            try:
                ollama_available = ollama_client.is_available()
            except Exception as exc:
                logger.warning(f"ask_memory: Ollama availability check failed: {exc}")

        effective_model = model or ctx.config.ollama.model

        if not ollama_available:
            logger.warning("ask_memory: Ollama is not available - returning sources only")
            return {
                "answer": None,
                "sources": hits,
                "model": effective_model,
                "ollama_available": False,
                "hint": (
                    f"Install Ollama from https://ollama.ai and run: ollama pull {effective_model}"
                ),
            }

        prompt = _build_rag_prompt(question, hits)
        assert ollama_client is not None  # guarded above
        answer = ollama_client.generate(prompt, model=model)

        if not answer:
            logger.warning("ask_memory: Ollama returned empty answer")
            answer = ""

        ctx.audit_logger.log_query(
            tool="ask_memory",
            query=question,
            n_results=len(hits),
            latency_ms=0.0,  # Latency not tracked separately here
        )

        return {
            "answer": answer,
            "sources": hits,
            "model": effective_model,
            "ollama_available": True,
        }
