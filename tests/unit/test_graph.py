"""Unit tests for the knowledge graph endpoint and MCP tool.

All storage dependencies are mocked - no real SQLite or ChromaDB required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_entity(name: str, etype: str = "PERSON", mentions: int = 3) -> object:
    ent = MagicMock()
    ent.name = name
    ent.entity_type = etype
    ent.mention_count = mentions
    return ent


def _make_ctx(entities: list, chunk_map: dict[str, list[str]]) -> object:
    """Return a minimal AppContext mock for graph tests."""
    ctx = MagicMock()
    ctx.metadata_store.list_entities.return_value = entities
    ctx.metadata_store.get_entity_chunks.side_effect = lambda name, etype: chunk_map.get(name, [])
    return ctx


class TestBuildKnowledgeGraph:
    def setup_method(self) -> None:
        from memorymesh.server import dashboard

        dashboard._GRAPH_CACHE.clear()

    def _call(self, ctx: object, **kwargs: object) -> dict:
        from memorymesh.server.dashboard import _build_knowledge_graph

        return _build_knowledge_graph(ctx, **kwargs)  # type: ignore[arg-type]

    def test_empty_entities_returns_empty_graph(self) -> None:
        ctx = _make_ctx([], {})
        result = self._call(ctx)
        assert result == {"nodes": [], "edges": []}

    def test_nodes_match_entity_list(self) -> None:
        entities = [_make_entity("Alice"), _make_entity("Bob")]
        ctx = _make_ctx(entities, {"Alice": ["doc:0"], "Bob": ["doc:1"]})
        result = self._call(ctx)
        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"Alice", "Bob"}

    def test_shared_chunk_creates_edge(self) -> None:
        entities = [_make_entity("Alice"), _make_entity("Bob")]
        ctx = _make_ctx(entities, {"Alice": ["doc:0", "doc:1"], "Bob": ["doc:1", "doc:2"]})
        result = self._call(ctx)
        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert {edge["source"], edge["target"]} == {"Alice", "Bob"}
        assert edge["weight"] == 1

    def test_no_shared_chunk_no_edge(self) -> None:
        entities = [_make_entity("Alice"), _make_entity("Bob")]
        ctx = _make_ctx(entities, {"Alice": ["doc:0"], "Bob": ["doc:1"]})
        result = self._call(ctx)
        assert result["edges"] == []

    def test_cache_avoids_second_store_call(self) -> None:

        from memorymesh.server import dashboard

        # Clear cache entry for this key
        cache_key = "1:"
        dashboard._GRAPH_CACHE.pop(cache_key, None)

        entities = [_make_entity("Alice", mentions=1)]
        ctx = _make_ctx(entities, {"Alice": []})

        # Prime cache
        dashboard._build_knowledge_graph(ctx, min_mentions=1)
        call_count_after_first = ctx.metadata_store.list_entities.call_count

        # Second call within TTL should not hit the store again
        dashboard._build_knowledge_graph(ctx, min_mentions=1)
        assert ctx.metadata_store.list_entities.call_count == call_count_after_first

    def test_stale_cache_refreshes(self) -> None:
        from memorymesh.server import dashboard

        cache_key = "99:"
        dashboard._GRAPH_CACHE[cache_key] = (0.0, {"nodes": [], "edges": []})

        entities = [_make_entity("Carol", mentions=99)]
        ctx = _make_ctx(entities, {"Carol": []})

        result = dashboard._build_knowledge_graph(ctx, min_mentions=99)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "Carol"


def _extract_tool_fn(entities: list, chunk_map: dict[str, list[str]]) -> object:
    """Register graph_memory on a throw-away FastMCP and return the raw callable."""
    from mcp.server.fastmcp import FastMCP

    from memorymesh.server.tools import graph_memory

    captured: dict = {}

    class CapturingFastMCP(FastMCP):
        def tool(self, *args: object, **kwargs: object):  # type: ignore[override]
            decorator = super().tool(*args, **kwargs)

            def wrapper(fn):  # type: ignore[return-value]
                captured["fn"] = fn
                return decorator(fn)

            return wrapper

    mcp = CapturingFastMCP("test")
    ctx = _make_ctx(entities, chunk_map)
    ctx.audit_logger = MagicMock()
    graph_memory.register(mcp, ctx)
    return captured["fn"]


class TestGraphMemoryTool:
    def test_returns_node_and_edge_counts(self) -> None:
        entities = [_make_entity("Alice"), _make_entity("Bob")]
        chunk_map = {"Alice": ["x:0", "x:1"], "Bob": ["x:1"]}
        fn = _extract_tool_fn(entities, chunk_map)
        with patch("memorymesh.server.auth_guard.check_access", return_value=None):
            result = fn()
        assert result["node_count"] == 2
        assert result["edge_count"] == 1

    def test_empty_graph_returns_zeros(self) -> None:
        fn = _extract_tool_fn([], {})
        with patch("memorymesh.server.auth_guard.check_access", return_value=None):
            result = fn()
        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    def test_auth_error_propagated(self) -> None:
        fn = _extract_tool_fn([], {})
        with patch(
            "memorymesh.server.auth_guard.check_access",
            return_value={"error": "forbidden"},
        ):
            result = fn()
        assert result == {"error": "forbidden"}
