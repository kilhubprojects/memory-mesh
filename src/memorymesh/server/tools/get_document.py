"""MCP tool: get_document.

Reads and returns the contents of a file that has been indexed by MemoryMesh.
The file must be under a configured source directory.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext

_MAX_BYTES_DEFAULT = 1 * 1024 * 1024  # 1 MB


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``get_document`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def get_document(
        path: Annotated[str, "Absolute path to the file (as returned by search_memory)."],
        max_bytes: Annotated[
            int,
            "Maximum bytes of content to return (default 1 MB).",
        ] = _MAX_BYTES_DEFAULT,
    ) -> dict:
        """Read and return the contents of an indexed file.

        The file must currently exist on disk and must have been indexed by
        MemoryMesh (i.e. its path must be present in the metadata store).

        Args:
            path: Absolute file path.
            max_bytes: Content is truncated to this many bytes when the file
                is large (``truncated: true`` is set in the response).
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "read")) is not None:
            return err

        t0 = time.perf_counter()

        record = ctx.metadata_store.get_file(path)
        if record is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            ctx.audit_logger.log_query(
                tool="get_document",
                query=path,
                n_results=0,
                latency_ms=latency_ms,
            )
            return {"error": f"File not found in index: {path}"}

        file_path = Path(path)
        if not file_path.exists():
            latency_ms = (time.perf_counter() - t0) * 1000
            ctx.audit_logger.log_query(
                tool="get_document",
                query=path,
                n_results=0,
                latency_ms=latency_ms,
            )
            return {"error": f"File no longer exists on disk: {path}"}

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            ctx.audit_logger.log_query(
                tool="get_document",
                query=path,
                n_results=0,
                latency_ms=latency_ms,
            )
            return {"error": f"Cannot read file: {exc}"}

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]

        try:
            content = raw.decode("utf-8", errors="replace")
        except Exception:
            content = raw.decode("latin-1", errors="replace")

        stat = file_path.stat()
        latency_ms = (time.perf_counter() - t0) * 1000
        ctx.audit_logger.log_query(
            tool="get_document",
            query=path,
            n_results=1,
            latency_ms=latency_ms,
        )

        return {
            "path": path,
            "content": content,
            "file_type": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "truncated": truncated,
            "duration_ms": round(latency_ms, 2),
        }
