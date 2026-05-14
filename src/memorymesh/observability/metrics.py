"""Lightweight in-process metrics collector for MemoryMesh.

Tracks search query counts, latency histograms (p50/p95/p99), and indexing
throughput.  All state is in-memory and resets on daemon restart.

A module-level singleton (``_collector``) is accessed via ``get_metrics()`` so
any module can record events without threading an extra dependency through the
call graph.  Thread-safety is provided by a :class:`threading.Lock`.
"""

from __future__ import annotations

import threading
import time


class MetricsCollector:
    """Thread-safe in-memory metrics collector.

    Tracks:
    - ``search_total`` — total search requests handled.
    - ``search_latency_ms`` — list of per-query latencies (ms).
    - ``search_by_mode`` — count per search mode string.
    - ``index_total`` — total documents indexed.
    - ``index_errors`` — total indexing errors.
    - ``index_latency_ms`` — list of per-document indexing latencies (ms).
    - ``start_time`` — UNIX timestamp when the collector was created.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time: float = time.time()
        self.search_total: int = 0
        self.search_latency_ms: list[float] = []
        self.search_by_mode: dict[str, int] = {}
        self.index_total: int = 0
        self.index_errors: int = 0
        self.index_latency_ms: list[float] = []

    def record_search(
        self,
        duration_ms: float,
        mode: str = "hybrid",
        returned: int = 0,
    ) -> None:
        """Record a completed search query.

        Args:
            duration_ms: End-to-end query latency in milliseconds.
            mode: Search mode (``"hybrid"``, ``"dense"``, or ``"sparse"``).
            returned: Number of hits returned to the caller.
        """
        with self._lock:
            self.search_total += 1
            self.search_latency_ms.append(duration_ms)
            self.search_by_mode[mode] = self.search_by_mode.get(mode, 0) + 1
            _ = returned  # reserved for future per-mode returned-count tracking

    def record_index(self, duration_ms: float, *, error: bool = False) -> None:
        """Record a completed document indexing operation.

        Args:
            duration_ms: Time taken to index the document in milliseconds.
            error: ``True`` when the document failed to index.
        """
        with self._lock:
            if error:
                self.index_errors += 1
            else:
                self.index_total += 1
                self.index_latency_ms.append(duration_ms)

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of current metrics.

        Returns:
            Dict with ``search``, ``indexing``, and ``uptime_s`` keys.
        """
        with self._lock:
            uptime = round(time.time() - self.start_time, 1)
            search_snap = {
                "total": self.search_total,
                "by_mode": dict(self.search_by_mode),
                **_latency_stats(self.search_latency_ms),
            }
            index_snap = {
                "total": self.index_total,
                "errors": self.index_errors,
                **{f"index_{k}": v for k, v in _latency_stats(self.index_latency_ms).items()},
            }
            return {
                "uptime_s": uptime,
                "search": search_snap,
                "indexing": index_snap,
            }


def _latency_stats(samples: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99 and mean from *samples*."""
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    sorted_s = sorted(samples)
    n = len(sorted_s)
    return {
        "p50_ms": round(_percentile(sorted_s, 50), 2),
        "p95_ms": round(_percentile(sorted_s, 95), 2),
        "p99_ms": round(_percentile(sorted_s, 99), 2),
        "mean_ms": round(sum(sorted_s) / n, 2),
    }


def _percentile(sorted_data: list[float], pct: int) -> float:
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


# Module-level singleton — created once, shared across all callers.
_collector = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Return the process-wide :class:`MetricsCollector` singleton."""
    return _collector
