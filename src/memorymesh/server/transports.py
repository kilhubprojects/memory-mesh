"""Transport layer — wires FastMCP to stdio or streamable-http.

Supported transports
--------------------
* ``stdio`` (default) — speaks JSON-RPC over stdin/stdout.  Used by Claude
  Desktop, Cursor, and any MCP host that spawns MemoryMesh as a subprocess.
* ``streamable-http`` — HTTP server on 127.0.0.1:<port>.  Recommended when
  MemoryMesh runs as a long-lived daemon and clients connect over a local
  socket.

SSE (Server-Sent Events) transport was deprecated in the MCP spec revision of
June 2025 and is intentionally NOT implemented here.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from memorymesh.core.models import ServerConfig
    from memorymesh.server.app import AppContext

_SHUTDOWN_TOKEN_ENV = "MEMORYMESH_ADMIN_TOKEN"


def run_stdio(mcp: FastMCP) -> None:
    """Run the MCP server over stdio (blocking).

    Args:
        mcp: Configured :class:`~mcp.server.fastmcp.FastMCP` instance.
    """
    logger.info("MemoryMesh MCP server starting (transport=stdio)")
    mcp.run(transport="stdio")


def run_http(
    mcp: FastMCP,
    config: ServerConfig,
    pid_path: Path,
    rest_api: object | None = None,
) -> None:
    """Run the MCP server over streamable-http (blocking).

    Also installs a ``POST /admin/shutdown`` endpoint protected by a bearer
    token stored in the ``MEMORYMESH_ADMIN_TOKEN`` environment variable.
    Graceful shutdown: the server process exits on receiving the request.

    Args:
        mcp: Configured :class:`~mcp.server.fastmcp.FastMCP` instance.
        config: Server transport configuration (host + port).
        pid_path: Path to write the PID file for the stop command.
        rest_api: Optional FastAPI application to mount at ``/api``.
            When ``None``, the REST API is disabled.
    """
    host = config.http.host
    port = config.http.port

    # Write PID file so ``memorymesh stop`` can locate us.
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")
        logger.debug(f"PID file written: {pid_path}")
    except OSError as exc:
        logger.warning(f"Could not write PID file {pid_path}: {exc}")

    def _cleanup_pid() -> None:
        try:
            if pid_path.exists():
                pid_path.unlink()
        except OSError as exc:
            logger.debug(f"Could not remove PID file {pid_path}: {exc}")

    # Register SIGTERM/SIGINT handlers to clean up PID file.
    def _handle_signal(sig: int, frame: object) -> None:
        logger.info(f"MemoryMesh: received signal {sig}, shutting down")
        _cleanup_pid()
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sig, _handle_signal)

    # Inject the admin shutdown endpoint via a starlette middleware mount.
    # FastMCP exposes the underlying Starlette app via .streamable_http_app().
    import secrets

    admin_token = os.environ.get(_SHUTDOWN_TOKEN_ENV, "")
    if not admin_token:
        admin_token = secrets.token_hex(32)
        logger.warning(
            f"MEMORYMESH_ADMIN_TOKEN not set. Generated token for this session: {admin_token}"
        )

    try:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        mcp_app = mcp.streamable_http_app()

        async def admin_shutdown(request: Request) -> JSONResponse:
            """Gracefully terminate the daemon after verifying the admin token."""
            token = request.headers.get("X-MemoryMesh-Token", "")
            if admin_token and token != admin_token:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            logger.info("MemoryMesh: admin shutdown requested")
            _cleanup_pid()
            # Schedule OS-level termination after replying.
            threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            return JSONResponse({"status": "shutting down"})

        routes: list[Route | Mount] = [
            Route("/admin/shutdown", admin_shutdown, methods=["POST"]),
        ]

        # Mount REST API if enabled
        if rest_api is not None:
            routes.append(Mount("/api", app=rest_api))  # type: ignore[arg-type]

        routes.append(Mount("/", app=mcp_app))  # type: ignore[arg-type]

        app = Starlette(routes=routes)

        logger.info(f"MemoryMesh MCP server starting (transport=streamable-http, {host}:{port})")
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except ImportError as exc:
        logger.error(f"streamable-http transport requires 'uvicorn' and 'starlette': {exc}")
        raise
    finally:
        _cleanup_pid()


def run(mcp: FastMCP, ctx: AppContext, enable_rest_api: bool = False) -> None:
    """Select and run the transport configured in *ctx*.

    Args:
        mcp: The FastMCP instance to serve.
        ctx: Application context (provides ``config.server`` and ``config.storage``).
        enable_rest_api: When ``True`` and transport is ``streamable-http``,
            mount the REST API at ``/api``.
    """
    transport = ctx.config.server.transport
    if transport == "streamable-http":
        pid_path = ctx.config.storage.base_dir / "daemon.pid"
        rest_app = None
        if enable_rest_api:
            from memorymesh.server.rest_api import build_rest_api

            rest_app = build_rest_api(ctx)
            logger.info("REST API enabled at /api")
        run_http(mcp, ctx.config.server, pid_path, rest_api=rest_app)
    else:
        run_stdio(mcp)
