"""MCP tool: sync_source.

Runs one or all enabled external data connectors and indexes the fetched
documents into the MemoryMesh vector + BM25 + SQLite stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``sync_source`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def sync_source(
        source_type: Annotated[
            str | None,
            (
                "Connector type to sync, e.g. 'jira' or 'slack'.  "
                "Pass null to sync all enabled connectors."
            ),
        ] = None,
        dry_run: Annotated[
            bool,
            "If true, fetch documents but do not write to the index.",
        ] = False,
    ) -> dict:
        """Fetch documents from one or all configured external connectors.

        Runs the enabled connectors defined in ``config.yaml`` and indexes
        each fetched document into the search index.

        Args:
            source_type: Connector type key to run, or ``null`` for all.
            dry_run: Fetch without indexing.
        """
        from memorymesh.connectors.registry import get_connector_classes
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "index")) is not None:
            return err

        connectors_to_run = [
            c
            for c in ctx.config.connectors
            if c.enabled and (source_type is None or c.type == source_type)
        ]

        if not connectors_to_run:
            label = f"'{source_type}'" if source_type else "any"
            return {
                "status": "no_connectors",
                "message": f"No enabled connector found matching {label}.",
            }

        total_docs = 0
        total_errors = 0
        results: list[dict] = []

        for conn_cfg in connectors_to_run:
            entry: dict = {"type": conn_cfg.type, "docs": 0, "errors": 0}
            try:
                cfg_cls, conn_cls = get_connector_classes(conn_cfg.type)
                config_obj = cfg_cls(**conn_cfg.config)
                connector = conn_cls(config_obj)
            except (KeyError, Exception) as exc:
                entry["errors"] = 1
                entry["error"] = str(exc)
                results.append(entry)
                total_errors += 1
                continue

            source_name = getattr(config_obj, "source_name", conn_cfg.type)
            doc_count = 0

            try:
                for doc in connector.fetch_documents():
                    doc_count += 1
                    if not dry_run:
                        result = ctx.indexer.index_parsed_document(doc, source_name)
                        if result.status == "parse_error":
                            entry["errors"] = entry["errors"] + 1
                            total_errors += 1
            except Exception as exc:
                entry["errors"] = entry["errors"] + 1
                entry["error"] = str(exc)
                total_errors += 1

            entry["docs"] = doc_count
            results.append(entry)
            total_docs += doc_count

        return {
            "status": "ok",
            "dry_run": dry_run,
            "total_docs": total_docs,
            "total_errors": total_errors,
            "connectors": results,
        }
