"""Health check and metrics endpoint for MemoryMesh (Wave 6).

Exposes a lightweight ``/health`` HTTP endpoint (independent of the MCP
transport) that monitoring systems and the CLI ``status`` command can poll.

The endpoint returns a JSON blob with:
- ``status`` — ``"ok"`` or ``"degraded"`` or ``"starting"``.
- ``version`` — the installed package version.
- ``uptime_s`` — seconds since the daemon started.
- ``sources`` — per-source file counts and last-scan timestamps.
- ``index`` — aggregate chunk counts and storage sizes.
- ``embedding_cache`` — hit/miss stats when a :class:`CachedEmbeddingProvider`
  is in use.
- ``memory_tiers`` — hot/warm/cold chunk distribution when tiers are enabled.

The health server runs in a **background thread** on ``127.0.0.1:8766``
(configurable).  It is intentionally kept separate from the MCP transport so
it remains accessible even when the MCP stdio transport is busy.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("memorymesh")
except Exception:
    _VERSION = "0.0.0"


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves ``/health`` as JSON."""

    ctx: AppContext
    start_time: float

    def do_GET(self) -> None:
        """Serve ``/health`` (liveness) and ``/metrics`` (observability) as JSON."""
        if self.path in ("/health", "/health/"):
            payload = _build_payload(self.ctx, self.start_time)
        elif self.path in ("/metrics", "/metrics/"):
            from memorymesh.observability.metrics import get_metrics

            payload = get_metrics().snapshot()
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = _to_json(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return None


def _to_json(obj: object) -> str:
    """Convert *obj* to a compact JSON string without importing the json module
    at module level (keeps startup fast)."""
    import json

    return json.dumps(obj, default=str)


def _build_payload(ctx: AppContext, start_time: float) -> dict:
    """Collect health metrics from *ctx* and return a serialisable dict."""
    uptime_s = round(time.time() - start_time, 1)
    status = "ok"

    # Source stats from MetadataStore.
    sources_payload: dict[str, object] = {}
    try:
        raw_stats = ctx.metadata_store.get_stats()
        sources_payload = {
            "indexed": raw_stats.get("indexed", 0),
            "errors": raw_stats.get("parse_error", 0),
            "pending": raw_stats.get("pending_reindex", 0),
        }
    except Exception as exc:
        logger.debug(f"health: metadata_store error: {exc}")
        sources_payload = {"error": str(exc)}
        status = "degraded"

    # Embedding cache stats.
    cache_payload: dict = {}
    try:
        from memorymesh.embeddings.cache import CachedEmbeddingProvider

        if isinstance(ctx.provider, CachedEmbeddingProvider):
            cache_payload = ctx.provider.cache_stats
    except Exception as exc:
        logger.debug(f"health: embedding cache stats unavailable: {exc}")

    # Memory tier distribution.
    tiers_payload: dict = {}
    if ctx.tiered_memory is not None:
        try:
            from memorymesh.core.models import MemoryTier

            tiers_payload = {
                tier.value: len(ctx.metadata_store.list_chunks_by_tier(tier, limit=None))
                for tier in MemoryTier
            }
        except Exception as exc:
            logger.debug(f"health: tier stats error: {exc}")

    return {
        "status": status,
        "version": _VERSION,
        "uptime_s": uptime_s,
        "index": sources_payload,
        "embedding_cache": cache_payload,
        "memory_tiers": tiers_payload,
    }


class HealthServer:
    """Background HTTP server exposing a ``/health`` endpoint.

    Args:
        ctx: The fully initialised :class:`AppContext`.
        host: Bind address (default ``"127.0.0.1"``).
        port: TCP port (default ``8766``).
    """

    def __init__(
        self,
        ctx: AppContext,
        host: str = "127.0.0.1",
        port: int = 8766,
    ) -> None:
        self._ctx = ctx
        self._host = host
        self._port = port
        self._start_time = time.time()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the health server in a daemon thread."""

        class _Handler(_HealthHandler):
            ctx = self._ctx
            start_time = self._start_time

        try:
            self._server = HTTPServer((self._host, self._port), _Handler)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="memorymesh-health",
            )
            self._thread.start()
            logger.info(f"HealthServer started on http://{self._host}:{self._port}/health")
        except OSError as exc:
            logger.warning(f"HealthServer: could not bind {self._host}:{self._port}: {exc}")

    def stop(self) -> None:
        """Shut down the health server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            logger.debug("HealthServer stopped")
