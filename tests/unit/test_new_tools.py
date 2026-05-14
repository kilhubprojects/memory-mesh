"""Unit tests for the 6 new MCP tools added in the comprehensive wave.

All storage / connector / embedding dependencies are mocked.  No real index
or HTTP calls are made.
"""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

from memorymesh.core.models import MemoryMeshConfig


class _CaptureMCP:
    """Minimal FastMCP stand-in that captures the first registered tool."""

    def __init__(self) -> None:
        self._fn: Any = None

    def tool(self) -> Any:
        def _dec(fn: Any) -> Any:
            self._fn = fn
            return fn

        return _dec

    def __call__(self, **kwargs: Any) -> Any:
        assert self._fn is not None, "No tool registered"
        return self._fn(**kwargs)


class _MultiCaptureMCP:
    """FastMCP stand-in that captures ALL registered tools by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def tool(self) -> Any:
        def _dec(fn: Any) -> Any:
            self._tools[fn.__name__] = fn
            return fn

        return _dec

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        fn = self._tools.get(name)
        assert fn is not None, f"Tool {name!r} not registered"
        return fn(**kwargs)


def _base_ctx(**overrides: Any) -> mock.MagicMock:
    """Return a minimal AppContext mock with sane defaults."""
    ctx = mock.MagicMock()
    ctx.config = MemoryMeshConfig()
    # Auth guard: check_access always returns None (permitted)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _register_and_invoke(module_path: str, ctx: Any, **kwargs: Any) -> dict:
    """Import *module_path*, register the tool, invoke it with *kwargs*."""
    import importlib

    mod = importlib.import_module(module_path)
    mcp = _CaptureMCP()
    with mock.patch("memorymesh.server.auth_guard.check_access", return_value=None):
        mod.register(mcp, ctx)
        return mcp(**kwargs)


def _register_and_invoke_named(
    module_path: str,
    tool_name: str,
    ctx: Any,
    *,
    access_return: Any = None,
    **kwargs: Any,
) -> dict:
    """Register all tools from *module_path*, invoke the one named *tool_name*.

    Args:
        module_path: Dotted import path of the tool module.
        tool_name: Name of the inner function to invoke.
        ctx: AppContext mock.
        access_return: Value returned by ``check_access`` (``None`` = allowed).
        **kwargs: Arguments forwarded to the tool function.
    """
    import importlib

    mod = importlib.import_module(module_path)
    mcp = _MultiCaptureMCP()
    with mock.patch("memorymesh.server.auth_guard.check_access", return_value=access_return):
        mod.register(mcp, ctx)
        return mcp.call_tool(tool_name, **kwargs)


# sync_source


class TestSyncSourceTool:
    def _ctx_with_connectors(self, connector_cfgs: list[Any]) -> mock.MagicMock:
        ctx = _base_ctx()
        cfg = MemoryMeshConfig()
        ctx.config = cfg
        ctx.config.connectors = connector_cfgs
        # Attach a fake indexer
        ctx.indexer = mock.MagicMock()
        ctx.indexer.index_parsed_document.return_value = mock.MagicMock(status="indexed")
        return ctx

    def test_no_connectors_returns_no_connectors(self) -> None:
        ctx = self._ctx_with_connectors([])
        result = _register_and_invoke("memorymesh.server.tools.sync_source", ctx)
        assert result["status"] == "no_connectors"

    def test_dry_run_does_not_call_indexer(self) -> None:
        from memorymesh.core.models import ConnectorConfig

        conn_cfg = ConnectorConfig(type="jira", enabled=True, config={})
        ctx = self._ctx_with_connectors([conn_cfg])

        fake_doc = mock.MagicMock()

        with mock.patch(
            "memorymesh.connectors.registry.get_connector_classes",
            return_value=(
                mock.MagicMock(return_value=mock.MagicMock()),
                mock.MagicMock(
                    return_value=mock.MagicMock(
                        fetch_documents=mock.MagicMock(return_value=[fake_doc])
                    )
                ),
            ),
        ):
            result = _register_and_invoke(
                "memorymesh.server.tools.sync_source",
                ctx,
                dry_run=True,
            )
        assert result["dry_run"] is True
        ctx.indexer.index_parsed_document.assert_not_called()

    def test_unknown_connector_type_increments_errors(self) -> None:
        from memorymesh.core.models import ConnectorConfig

        conn_cfg = ConnectorConfig(type="unknown_xyz", enabled=True, config={})
        ctx = self._ctx_with_connectors([conn_cfg])

        with mock.patch(
            "memorymesh.connectors.registry.get_connector_classes",
            side_effect=KeyError("unknown_xyz"),
        ):
            result = _register_and_invoke("memorymesh.server.tools.sync_source", ctx)
        assert result["total_errors"] >= 1

    def test_source_type_filter_skips_non_matching(self) -> None:
        from memorymesh.core.models import ConnectorConfig

        conn_cfg = ConnectorConfig(type="slack", enabled=True, config={})
        ctx = self._ctx_with_connectors([conn_cfg])

        result = _register_and_invoke(
            "memorymesh.server.tools.sync_source",
            ctx,
            source_type="jira",
        )
        # slack != jira, so nothing matches
        assert result["status"] == "no_connectors"


# get_entity


class TestGetEntityTool:
    def test_found(self) -> None:
        entity = mock.MagicMock()
        entity.name = "alice"
        entity.entity_type = "person"

        ctx = _base_ctx()
        ctx.metadata_store.get_entity.return_value = entity
        ctx.metadata_store.get_entity_mentions.return_value = ["chunk1", "chunk2"]

        result = _register_and_invoke(
            "memorymesh.server.tools.get_entity",
            ctx,
            name="Alice",
        )
        assert result["found"] is True
        assert result["name"] == "alice"
        assert result["mention_count"] == 2

    def test_not_found(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.get_entity.return_value = None

        result = _register_and_invoke(
            "memorymesh.server.tools.get_entity",
            ctx,
            name="nobody",
        )
        assert result["found"] is False

    def test_type_mismatch(self) -> None:
        entity = mock.MagicMock()
        entity.name = "python"
        entity.entity_type = "concept"

        ctx = _base_ctx()
        ctx.metadata_store.get_entity.return_value = entity
        ctx.metadata_store.get_entity_mentions.return_value = []

        result = _register_and_invoke(
            "memorymesh.server.tools.get_entity",
            ctx,
            name="python",
            entity_type="person",
        )
        assert result["found"] is False
        assert "mismatch" in result["message"]

    def test_chunk_ids_capped_at_100(self) -> None:
        entity = mock.MagicMock()
        entity.name = "bigent"
        entity.entity_type = "project"

        ctx = _base_ctx()
        ctx.metadata_store.get_entity.return_value = entity
        ctx.metadata_store.get_entity_mentions.return_value = [f"chunk{i}" for i in range(200)]

        result = _register_and_invoke(
            "memorymesh.server.tools.get_entity",
            ctx,
            name="bigent",
        )
        assert len(result["chunk_ids"]) == 100


# related_documents


class TestRelatedDocumentsTool:
    def test_not_found_when_no_chunks(self) -> None:
        ctx = _base_ctx()
        ctx.vector_store.get_by_path.return_value = []

        result = _register_and_invoke(
            "memorymesh.server.tools.related_documents",
            ctx,
            path="/some/file.md",
        )
        assert result["status"] == "not_found"

    def _make_hit(self, path: str, score: float = 0.9, text: str = "content") -> Any:
        h = mock.MagicMock()
        h.path = path
        h.score = score
        h.preview = text
        return h

    def test_returns_related(self) -> None:
        ctx = _base_ctx()
        ctx.vector_store.get_by_path.return_value = [
            {"text": "anchor text here", "path": "/anchor.md"}
        ]
        ctx.provider.embed_query.return_value = [0.1, 0.2, 0.3]
        ctx.vector_store.search.return_value = [
            self._make_hit("/other.md", 0.9, "related content"),
            self._make_hit("/anchor.md", 0.99, "self"),
            self._make_hit("/third.md", 0.8, "third doc"),
        ]

        result = _register_and_invoke(
            "memorymesh.server.tools.related_documents",
            ctx,
            path="/anchor.md",
            top_k=5,
            exclude_self=True,
        )
        assert result["status"] == "ok"
        paths = [r["path"] for r in result["related"]]
        assert "/anchor.md" not in paths

    def test_top_k_clamped_to_20(self) -> None:
        ctx = _base_ctx()
        ctx.vector_store.get_by_path.return_value = [{"text": "text"}]
        ctx.provider.embed_query.return_value = [0.1]
        ctx.vector_store.search.return_value = []

        result = _register_and_invoke(
            "memorymesh.server.tools.related_documents",
            ctx,
            path="/f.md",
            top_k=999,
        )
        # Should not crash; top_k clamped
        assert result["status"] == "ok"

    def test_exclude_self_false_includes_anchor(self) -> None:
        ctx = _base_ctx()
        ctx.vector_store.get_by_path.return_value = [{"text": "anchor", "path": "/anchor.md"}]
        ctx.provider.embed_query.return_value = [0.1]
        ctx.vector_store.search.return_value = [
            self._make_hit("/anchor.md", 0.99, "self content"),
        ]

        result = _register_and_invoke(
            "memorymesh.server.tools.related_documents",
            ctx,
            path="/anchor.md",
            exclude_self=False,
        )
        paths = [r["path"] for r in result["related"]]
        assert "/anchor.md" in paths


# search_by_date


class TestSearchByDateTool:
    def _make_records(self, n: int = 5) -> list[mock.MagicMock]:
        recs = []
        for i in range(n):
            r = mock.MagicMock()
            r.path = f"/file_{i}.md"
            r.source_name = "documents"
            r.file_type = ".md"
            r.n_chunks = 3
            r.indexed_at = 1_700_000_000.0 + i * 86400
            r.mtime = 1_699_000_000.0 + i * 86400
            recs.append(r)
        return recs

    def test_returns_records(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(3)

        result = _register_and_invoke("memorymesh.server.tools.search_by_date", ctx)
        assert result["status"] == "ok"
        assert result["count"] == 3

    def test_source_filter(self) -> None:
        recs = self._make_records(3)
        recs[0].source_name = "jira"

        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = recs

        result = _register_and_invoke(
            "memorymesh.server.tools.search_by_date",
            ctx,
            source="jira",
        )
        assert result["count"] == 1

    def test_invalid_date_returns_error(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = []

        result = _register_and_invoke(
            "memorymesh.server.tools.search_by_date",
            ctx,
            after="not-a-date",
        )
        assert result["status"] == "error"

    def test_limit_respected(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(10)

        result = _register_and_invoke(
            "memorymesh.server.tools.search_by_date",
            ctx,
            limit=3,
        )
        assert result["count"] <= 3


# forget_source


class TestForgetSourceTool:
    def _make_records(self, source: str, n: int = 3) -> list[mock.MagicMock]:
        recs = []
        for i in range(n):
            r = mock.MagicMock()
            r.path = f"/file_{i}.jira"
            r.source_name = source
            recs.append(r)
        return recs

    def test_not_found(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = []

        result = _register_and_invoke(
            "memorymesh.server.tools.forget_source",
            ctx,
            source="jira",
        )
        assert result["status"] == "not_found"

    def test_dry_run_does_not_delete(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records("jira")

        result = _register_and_invoke(
            "memorymesh.server.tools.forget_source",
            ctx,
            source="jira",
            dry_run=True,
        )
        assert result["status"] == "dry_run"
        assert result["would_remove"] == 3
        ctx.vector_store.delete_by_path.assert_not_called()

    def test_removes_from_all_stores(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records("jira", 2)

        result = _register_and_invoke(
            "memorymesh.server.tools.forget_source",
            ctx,
            source="jira",
        )
        assert result["status"] == "ok"
        assert result["removed"] == 2
        assert ctx.vector_store.delete_by_path.call_count == 2
        assert ctx.bm25.delete_by_path.call_count == 2

    def test_error_counted_on_store_failure(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records("slack", 1)
        ctx.vector_store.delete_by_path.side_effect = RuntimeError("disk full")

        result = _register_and_invoke(
            "memorymesh.server.tools.forget_source",
            ctx,
            source="slack",
        )
        assert result["errors"] == 1


# summarize_source


class TestSummarizeSourceTool:
    def _make_records(self, sources: list[str]) -> list[mock.MagicMock]:
        recs = []
        for i, src in enumerate(sources):
            r = mock.MagicMock()
            r.path = f"/file_{i}.md"
            r.source_name = src
            r.file_type = ".md"
            r.n_chunks = 5
            r.indexed_at = 1_700_000_000.0 + i
            recs.append(r)
        return recs

    def test_summarizes_all_sources(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(
            ["documents", "jira", "jira"]
        )

        result = _register_and_invoke("memorymesh.server.tools.summarize_source", ctx)
        assert result["status"] == "ok"
        names = {s["source"] for s in result["sources"]}
        assert "documents" in names
        assert "jira" in names

    def test_filter_by_source(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(["jira", "slack", "jira"])

        result = _register_and_invoke(
            "memorymesh.server.tools.summarize_source",
            ctx,
            source="jira",
        )
        assert result["status"] == "ok"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "jira"
        assert result["sources"][0]["documents"] == 2

    def test_source_not_found(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(["slack"])

        result = _register_and_invoke(
            "memorymesh.server.tools.summarize_source",
            ctx,
            source="jira",
        )
        assert result["status"] == "not_found"

    def test_totals_aggregated(self) -> None:
        ctx = _base_ctx()
        ctx.metadata_store.list_files.return_value = self._make_records(["a", "a", "b"])

        result = _register_and_invoke("memorymesh.server.tools.summarize_source", ctx)
        assert result["total_documents"] == 3
        assert result["total_chunks"] == 15  # 3 records x 5 chunks each


# pin_memory / unpin_memory

_PIN_MODULE = "memorymesh.server.tools.pin_memory"


class TestPinMemoryTool:
    def test_pin_happy_path(self) -> None:
        ctx = _base_ctx()
        ctx.tiered_memory = mock.MagicMock()

        result = _register_and_invoke_named(_PIN_MODULE, "pin_memory", ctx, chunk_id="doc.md:0")

        assert result == {"chunk_id": "doc.md:0", "tier": "hot", "pinned": True}
        ctx.tiered_memory.pin.assert_called_once_with("doc.md:0")
        ctx.audit_logger.log_query.assert_called_once()

    def test_pin_tiers_disabled_returns_error(self) -> None:
        ctx = _base_ctx()
        ctx.tiered_memory = None

        result = _register_and_invoke_named(_PIN_MODULE, "pin_memory", ctx, chunk_id="x:0")

        assert "error" in result
        assert "not enabled" in result["error"]

    def test_pin_access_denied_returns_error_payload(self) -> None:
        ctx = _base_ctx()
        ctx.tiered_memory = mock.MagicMock()
        denied = {"error": "PERMISSION_DENIED", "required": "index"}

        result = _register_and_invoke_named(
            _PIN_MODULE, "pin_memory", ctx, access_return=denied, chunk_id="x:0"
        )

        assert result == denied
        ctx.tiered_memory.pin.assert_not_called()

    def test_unpin_happy_path(self) -> None:
        ctx = _base_ctx()
        ctx.tiered_memory = mock.MagicMock()

        result = _register_and_invoke_named(_PIN_MODULE, "unpin_memory", ctx, chunk_id="doc.md:0")

        assert result == {"chunk_id": "doc.md:0", "pinned": False}
        ctx.tiered_memory.unpin.assert_called_once_with("doc.md:0")

    def test_unpin_tiers_disabled_returns_error(self) -> None:
        ctx = _base_ctx()
        ctx.tiered_memory = None

        result = _register_and_invoke_named(_PIN_MODULE, "unpin_memory", ctx, chunk_id="x:0")

        assert "error" in result


# query_timeline / record_event

_TIMELINE_MODULE = "memorymesh.server.tools.query_timeline"


def _episodic_ctx(*, enabled: bool = True) -> mock.MagicMock:
    """Return a ctx mock with episodic memory configured."""
    ctx = mock.MagicMock()
    ctx.config = mock.MagicMock()
    ctx.config.memory.episodic.enabled = enabled
    return ctx


def _make_event(**overrides: Any) -> mock.MagicMock:
    """Return a minimal EpisodicEvent-shaped mock."""
    ev = mock.MagicMock()
    ev.event_id = overrides.get("event_id", "evt-1")
    ev.timestamp = overrides.get("timestamp", 1_700_000_000.0)
    ev.event_type = overrides.get("event_type", "retrieval")
    ev.source = overrides.get("source", "file.md")
    ev.chunk_ids = overrides.get("chunk_ids", [])
    ev.client_id = overrides.get("client_id", "")
    ev.metadata = overrides.get("metadata", {})
    return ev


class TestQueryTimelineTool:
    def test_returns_events(self) -> None:
        ctx = _episodic_ctx(enabled=True)
        ctx.metadata_store.list_episodic_events.return_value = [_make_event()]

        result = _register_and_invoke_named(_TIMELINE_MODULE, "query_timeline", ctx)

        assert result["total"] == 1
        assert result["events"][0]["event_id"] == "evt-1"
        assert result["events"][0]["event_type"] == "retrieval"
        assert "since_ts" in result and "until_ts" in result

    def test_episodic_disabled_returns_error(self) -> None:
        ctx = _episodic_ctx(enabled=False)

        result = _register_and_invoke_named(_TIMELINE_MODULE, "query_timeline", ctx)

        assert "error" in result
        assert "not enabled" in result["error"]

    def test_limit_clamped_to_minimum(self) -> None:
        ctx = _episodic_ctx(enabled=True)
        ctx.metadata_store.list_episodic_events.return_value = []

        result = _register_and_invoke_named(_TIMELINE_MODULE, "query_timeline", ctx, limit=0)

        _, call_kwargs = ctx.metadata_store.list_episodic_events.call_args
        assert call_kwargs["limit"] >= 1
        assert "total" in result

    def test_event_type_filter_forwarded(self) -> None:
        ctx = _episodic_ctx(enabled=True)
        ctx.metadata_store.list_episodic_events.return_value = []

        _register_and_invoke_named(_TIMELINE_MODULE, "query_timeline", ctx, event_type="pin")

        _, call_kwargs = ctx.metadata_store.list_episodic_events.call_args
        assert call_kwargs["event_type"] == "pin"

    def test_access_denied_short_circuits(self) -> None:
        ctx = _episodic_ctx(enabled=True)
        denied = {"error": "PERMISSION_DENIED"}

        result = _register_and_invoke_named(
            _TIMELINE_MODULE, "query_timeline", ctx, access_return=denied
        )

        assert result == denied
        ctx.metadata_store.list_episodic_events.assert_not_called()


class TestRecordEventTool:
    def test_record_persists_event(self) -> None:
        ctx = _episodic_ctx(enabled=True)

        result = _register_and_invoke_named(
            _TIMELINE_MODULE,
            "record_event",
            ctx,
            event_type="user_note",
            source="notes.md",
            note="important",
            chunk_ids=[],
        )

        ctx.metadata_store.upsert_episodic_event.assert_called_once()
        assert result["event_type"] == "user_note"
        assert result["source"] == "notes.md"
        assert "event_id" in result

    def test_note_stored_in_metadata(self) -> None:
        ctx = _episodic_ctx(enabled=True)

        _register_and_invoke_named(
            _TIMELINE_MODULE,
            "record_event",
            ctx,
            event_type="pin",
            note="remember this",
            chunk_ids=[],
        )

        stored_event = ctx.metadata_store.upsert_episodic_event.call_args[0][0]
        assert stored_event.metadata.get("note") == "remember this"

    def test_no_note_yields_empty_metadata(self) -> None:
        ctx = _episodic_ctx(enabled=True)

        _register_and_invoke_named(
            _TIMELINE_MODULE,
            "record_event",
            ctx,
            event_type="pin",
            chunk_ids=[],
        )

        stored_event = ctx.metadata_store.upsert_episodic_event.call_args[0][0]
        assert stored_event.metadata == {}

    def test_episodic_disabled_returns_error(self) -> None:
        ctx = _episodic_ctx(enabled=False)

        result = _register_and_invoke_named(
            _TIMELINE_MODULE, "record_event", ctx, event_type="user_note", chunk_ids=[]
        )

        assert "error" in result
        ctx.metadata_store.upsert_episodic_event.assert_not_called()
