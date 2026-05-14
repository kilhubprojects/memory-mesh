"""Unit tests for the ask_memory MCP tool.

Mocks SearchEngine and OllamaClient to avoid live index or LLM dependencies.
"""

from __future__ import annotations

import unittest.mock as mock

from memorymesh.core.models import (
    MemoryMeshConfig,
    OllamaConfig,
    SearchHit,
    SearchResponse,
)


def _make_hits(n: int = 3) -> list[SearchHit]:
    return [
        SearchHit(
            path=f"/fake/file_{i}.py",
            chunk_index=i,
            score=1.0 / (i + 1),
            preview=f"Content of file {i} relevant to the query.",
            file_type=".py",
            mtime=0.0,
            source_root="test",
        )
        for i in range(n)
    ]


def _make_search_response(n: int = 3) -> SearchResponse:
    return SearchResponse(
        hits=_make_hits(n),
        mode="hybrid",
        duration_ms=42.0,
    )


def _build_ctx(ollama_enabled: bool = True, ollama_client: object = None) -> object:
    """Build a minimal AppContext-like object with mocked dependencies."""
    config = MemoryMeshConfig()
    config = config.model_copy(
        update={"ollama": OllamaConfig(enabled=ollama_enabled, model="llama3", timeout_s=5)}
    )

    engine = mock.MagicMock()
    engine.search.return_value = _make_search_response(3)

    audit_logger = mock.MagicMock()
    audit_logger.log_query = mock.MagicMock()

    ctx = mock.MagicMock()
    ctx.config = config
    ctx.engine = engine
    ctx.audit_logger = audit_logger
    ctx.ollama_client = ollama_client
    return ctx


# Tests


class TestAskMemoryTool:
    def _invoke(self, ctx: object, **kwargs: object) -> dict:
        """Register the tool on a mock MCP and invoke the captured function."""
        from memorymesh.server.tools import ask_memory

        captured: list[object] = []

        class _CaptureMCP:
            def tool(self) -> object:
                def _dec(fn: object) -> object:
                    captured.append(fn)
                    return fn

                return _dec

        cap = _CaptureMCP()
        ask_memory.register(cap, ctx)  # type: ignore[arg-type]

        assert captured, "tool decorator was not called"
        fn = captured[0]
        return fn(**kwargs)  # type: ignore[operator]

    def test_returns_sources_when_ollama_unavailable(self) -> None:
        mock_client = mock.MagicMock()
        mock_client.is_available.return_value = False

        ctx = _build_ctx(ollama_enabled=True, ollama_client=mock_client)
        result = self._invoke(ctx, question="What is MemoryMesh?", top_k=3)

        assert result["ollama_available"] is False
        assert result["answer"] is None
        assert "hint" in result
        assert len(result["sources"]) == 3

    def test_hint_present_when_ollama_unavailable(self) -> None:
        mock_client = mock.MagicMock()
        mock_client.is_available.return_value = False

        ctx = _build_ctx(ollama_enabled=True, ollama_client=mock_client)
        result = self._invoke(ctx, question="test", top_k=2)

        assert "ollama" in result["hint"].lower() or "install" in result["hint"].lower()

    def test_answer_returned_when_ollama_available(self) -> None:
        mock_client = mock.MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "MemoryMesh is a local MCP hub."

        ctx = _build_ctx(ollama_enabled=True, ollama_client=mock_client)
        result = self._invoke(ctx, question="What is MemoryMesh?", top_k=3)

        assert result["ollama_available"] is True
        assert result["answer"] == "MemoryMesh is a local MCP hub."
        assert len(result["sources"]) == 3

    def test_top_k_respected(self) -> None:
        mock_client = mock.MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "ok"

        ctx = _build_ctx(ollama_enabled=True, ollama_client=mock_client)
        ctx.engine.search.return_value = _make_search_response(5)

        self._invoke(ctx, question="test", top_k=5)

        ctx.engine.search.assert_called_once()
        call_kwargs = ctx.engine.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 5 or call_kwargs.args[1] == 5

    def test_top_k_clamped_to_max(self) -> None:
        mock_client = mock.MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "ok"

        ctx = _build_ctx(ollama_enabled=True, ollama_client=mock_client)
        ctx.engine.search.return_value = _make_search_response(3)

        # Pass top_k=100 - should be clamped to 20
        self._invoke(ctx, question="test", top_k=100)

        call_kwargs = ctx.engine.search.call_args
        effective_k = call_kwargs.kwargs.get("top_k") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert effective_k is not None
        assert effective_k <= 20

    def test_no_ollama_client_returns_no_answer(self) -> None:
        """When ctx.ollama_client is None, answer is always None."""
        ctx = _build_ctx(ollama_enabled=True, ollama_client=None)
        result = self._invoke(ctx, question="test", top_k=3)

        assert result["ollama_available"] is False
        assert result["answer"] is None
