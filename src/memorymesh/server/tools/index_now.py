"""MCP tool: index_now.

Triggers an immediate (re-)index of a specific path or all configured sources.
Useful when the filesystem watcher is disabled or after adding new files.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from loguru import logger
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


def register(mcp: FastMCP, ctx: AppContext) -> None:
    """Register the ``index_now`` tool on *mcp* with *ctx* injected.

    Args:
        mcp: The FastMCP instance to register onto.
        ctx: Shared application context (injected via closure).
    """

    @mcp.tool()
    def index_now(
        path: Annotated[
            str | None,
            "Absolute path to a file or directory to index. "
            "Omit to re-index all configured sources.",
        ] = None,
        force: Annotated[
            bool,
            "Re-index even if the file hash has not changed.",
        ] = False,
    ) -> dict:
        """Trigger an immediate (re-)index of a file, directory, or all sources.

        Args:
            path: Target path.  When ``None``, all :attr:`AppContext.config.sources`
                are scanned.
            force: Skip hash comparison and always re-index.
        """
        from memorymesh.server.auth_guard import check_access

        if (err := check_access(ctx, "index")) is not None:
            return err

        t0 = time.perf_counter()
        errors: list[str] = []
        n_files = 0
        n_chunks = 0

        if path is not None:
            target = Path(path)
            if not target.exists():
                return {"error": f"Path does not exist: {path}"}

            if target.is_file():
                result = ctx.indexer.index_file(target, force=force)
                n_files = 1
                n_chunks = result.n_chunks
                if result.error:
                    errors.append(f"{path}: {result.error}")
            else:
                results = ctx.indexer.index_directory(target, force=force)
                n_files = len(results)
                n_chunks = sum(r.n_chunks for r in results)
                errors = [f"{r.path}: {r.error}" for r in results if r.error]
        else:
            for src in ctx.config.sources:
                src_name = src.name or str(src.path)
                if not src.path.exists():
                    errors.append(f"Source path not found: {src.path}")
                    continue
                results = ctx.indexer.index_directory(
                    src.path,
                    source_name=src_name,
                    recursive=src.recursive,
                    extensions=src.extensions or None,
                    ignore_patterns=src.ignore or None,
                    force=force,
                )
                n_files += len(results)
                n_chunks += sum(r.n_chunks for r in results)
                errors += [f"{r.path}: {r.error}" for r in results if r.error]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"index_now: files={n_files} chunks={n_chunks} "
            f"errors={len(errors)} duration={latency_ms:.0f}ms"
        )
        ctx.audit_logger.log_query(
            tool="index_now",
            query=path or "<all sources>",
            n_results=n_files,
            latency_ms=latency_ms,
        )

        return {
            "n_files_processed": n_files,
            "n_chunks": n_chunks,
            "duration_ms": round(latency_ms, 2),
            "errors": errors,
        }
