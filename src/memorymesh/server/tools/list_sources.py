"""MCP tool: list_sources.

Returns statistics for every configured source directory — how many files
have been indexed, how many are pending or errored, and disk totals.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``list_sources`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def list_sources() -> dict:
        """List all configured source directories with indexing statistics.

        Returns a summary for each source: file counts by status, total chunk
        count, and aggregate disk size.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        t0 = time.perf_counter()

        sources_out = []
        for src_cfg in ctx.config.sources:
            name = src_cfg.name or str(src_cfg.path)

            all_records = ctx.metadata_store.list_files()
            matching = [r for r in all_records if r.source_name == name]

            n_indexed = sum(1 for r in matching if r.status == "indexed")
            n_pending = sum(1 for r in matching if r.status == "pending_reindex")
            n_errored = sum(1 for r in matching if r.status == "parse_error")
            total_chunks = sum(r.n_chunks for r in matching if r.status == "indexed")
            disk_bytes = sum(r.size_bytes for r in matching)
            last_scan = max((r.indexed_at for r in matching), default=None)

            sources_out.append(
                {
                    "name": name,
                    "path": str(src_cfg.path),
                    "recursive": src_cfg.recursive,
                    "n_files_indexed": n_indexed,
                    "n_files_pending": n_pending,
                    "n_files_errored": n_errored,
                    "total_chunks": total_chunks,
                    "disk_size_bytes": disk_bytes,
                    "last_scan_at": last_scan,
                }
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        ctx.audit_logger.log_query(
            tool="list_sources",
            query="list_sources",
            n_results=len(sources_out),
            latency_ms=latency_ms,
        )

        return {"sources": sources_out, "duration_ms": round(latency_ms, 2)}
