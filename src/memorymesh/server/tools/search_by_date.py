"""MCP tool: search_by_date.

Search indexed documents by modification or indexing date range.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``search_by_date`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def search_by_date(
        after: Annotated[
            str | None,
            "ISO-8601 date string; return only documents indexed/modified after this date.",
        ] = None,
        before: Annotated[
            str | None,
            "ISO-8601 date string; return only documents indexed/modified before this date.",
        ] = None,
        source: Annotated[
            str | None,
            "Restrict to a specific source name.",
        ] = None,
        file_type: Annotated[
            str | None,
            "Restrict to a specific file type extension, e.g. '.jira' or '.pdf'.",
        ] = None,
        limit: Annotated[int, "Maximum number of records to return (1-500)."] = 50,
    ) -> dict:
        """List indexed documents within a date range.

        Returns file records matching the given date window.  Useful for
        discovering recently indexed content or auditing what was indexed in
        a specific period.

        Args:
            after: Only include documents indexed/modified after this date.
            before: Only include documents indexed/modified before this date.
            source: Restrict to this source name.
            file_type: Restrict to this file type (e.g. ``".pdf"``).
            limit: Maximum records to return.
        """
        from datetime import UTC, datetime

        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        limit = max(1, min(500, limit))

        after_ts: float | None = None
        before_ts: float | None = None

        if after:
            try:
                after_ts = datetime.fromisoformat(after.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid 'after' date format: {after!r}",
                }

        if before:
            try:
                before_ts = datetime.fromisoformat(before.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid 'before' date format: {before!r}",
                }

        records = ctx.metadata_store.list_files(status="indexed")

        filtered = []
        for rec in records:
            if source and rec.source_name != source:
                continue
            if file_type and rec.file_type != file_type:
                continue
            ts = rec.indexed_at or rec.mtime
            if after_ts and ts < after_ts:
                continue
            if before_ts and ts > before_ts:
                continue
            filtered.append(rec)
            if len(filtered) >= limit:
                break

        def _iso(ts: float | None) -> str:
            if ts is None:
                return ""
            return datetime.fromtimestamp(ts, tz=UTC).isoformat()

        return {
            "status": "ok",
            "count": len(filtered),
            "records": [
                {
                    "path": r.path,
                    "source": r.source_name,
                    "file_type": r.file_type,
                    "n_chunks": r.n_chunks,
                    "indexed_at": _iso(r.indexed_at),
                    "mtime": _iso(r.mtime),
                }
                for r in filtered
            ],
        }
