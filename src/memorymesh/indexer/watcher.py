"""Filesystem watcher with debounce, built on watchdog.

``WatcherService`` monitors one or more source directories and calls
:meth:`~memorymesh.indexer.file_indexer.FileIndexer.index_file` or
:meth:`~memorymesh.indexer.file_indexer.FileIndexer.delete_file` after the
debounce window expires.

Architecture:
- watchdog emits raw events on a background thread.
- Events are buffered per-path in ``_pending``.
- A timer per path fires after ``debounce_ms``; the fired callback submits
  the file to a :class:`~concurrent.futures.ThreadPoolExecutor`.
- ``use_polling=True`` selects watchdog's ``PollingObserver`` for NFS/WSL.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import MemoryMeshConfig, SourceConfig
    from memorymesh.indexer.file_indexer import FileIndexer


class _DebounceHandler:
    """watchdog event handler with per-path debounce timers."""

    def __init__(
        self,
        indexer: FileIndexer,
        source_config: SourceConfig,
        debounce_ms: int,
        executor: ThreadPoolExecutor,
        ignore_patterns: list[str],
    ) -> None:
        self._indexer = indexer
        self._source = source_config
        self._debounce_s = debounce_ms / 1000.0
        self._executor = executor
        self._ignore_patterns = ignore_patterns
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    # watchdog calls these on its thread
    def on_created(self, event: object) -> None:  # type: ignore[override]
        """Handle a file-created event from watchdog."""
        self._schedule(event, deleted=False)

    def on_modified(self, event: object) -> None:  # type: ignore[override]
        """Handle a file-modified event from watchdog."""
        self._schedule(event, deleted=False)

    def on_deleted(self, event: object) -> None:  # type: ignore[override]
        """Handle a file-deleted event from watchdog."""
        self._schedule(event, deleted=True)

    def on_moved(self, event: object) -> None:  # type: ignore[override]
        """Handle a file-moved event from watchdog, treating it as delete + create."""
        # Treat move as delete source + create destination
        src = getattr(event, "src_path", None)
        dest = getattr(event, "dest_path", None)
        if src:
            self._fire(src, deleted=True)
        if dest:
            self._fire(dest, deleted=False)

    def _schedule(self, event: object, *, deleted: bool) -> None:
        is_dir = getattr(event, "is_directory", False)
        if is_dir:
            return
        path_str: str = getattr(event, "src_path", "")
        if not path_str:
            return
        path = Path(path_str)
        from memorymesh.indexer.file_indexer import _should_ignore

        if _should_ignore(path, self._source.path, self._ignore_patterns):
            return

        with self._lock:
            existing = self._timers.pop(path_str, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(self._debounce_s, self._fire, args=(path_str, deleted))
            self._timers[path_str] = timer
            timer.start()

    def _fire(self, path_str: str, deleted: bool) -> None:
        with self._lock:
            self._timers.pop(path_str, None)
        path = Path(path_str)
        if deleted:
            self._executor.submit(self._do_delete, path)
        else:
            self._executor.submit(self._do_index, path)

    def _do_index(self, path: Path) -> None:
        try:
            result = self._indexer.index_file(path, source_name=self._source.name)
            logger.debug(f"WatcherService: indexed {path.name!r} status={result.status!r}")
        except Exception as exc:
            logger.warning(f"WatcherService: failed to index {path}: {exc}")

    def _do_delete(self, path: Path) -> None:
        try:
            self._indexer.delete_file(path)
        except Exception as exc:
            logger.warning(f"WatcherService: failed to delete {path}: {exc}")

    def cancel_all_timers(self) -> None:
        """Cancel all pending debounce timers, discarding any un-fired events."""
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()

    def pending_count(self) -> int:
        """Return the number of file events currently waiting in the debounce queue."""
        with self._lock:
            return len(self._timers)


class WatcherService:
    """Monitors configured source directories and triggers re-indexing on change.

    Args:
        config: Root configuration (used for source paths, debounce settings,
            ignore patterns, and ``use_polling``).
        indexer: :class:`~memorymesh.indexer.file_indexer.FileIndexer` to call
            for each changed file.
    """

    def __init__(self, config: MemoryMeshConfig, indexer: FileIndexer) -> None:
        self._config = config
        self._indexer = indexer
        self._observer: object | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._handlers: list[_DebounceHandler] = []

    def start(self) -> None:
        """Start watching all configured source directories."""
        try:
            if self._config.watcher.use_polling:
                from watchdog.observers.polling import (  # type: ignore[import-untyped]
                    PollingObserver as Observer,
                )
            else:
                from watchdog.observers import Observer  # type: ignore[import-untyped,assignment]
        except ImportError as exc:
            logger.error(f"WatcherService: watchdog not installed ({exc}); watcher disabled")
            return

        from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]

        concurrency = self._config.watcher.worker_concurrency
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="watcher")
        self._observer = Observer()
        self._handlers = []

        for source in self._config.sources:
            if not source.path.exists():
                logger.warning(f"WatcherService: source path does not exist: {source.path}")
                continue

            ignore = list(self._config.global_ignore) + list(source.ignore)
            handler = _DebounceHandler(
                indexer=self._indexer,
                source_config=source,
                debounce_ms=self._config.watcher.debounce_ms,
                executor=self._executor,
                ignore_patterns=ignore,
            )

            # Wrap our handler in a FileSystemEventHandler subclass
            class _Bridge(FileSystemEventHandler):
                def __init__(self, h: _DebounceHandler) -> None:
                    super().__init__()
                    self._h = h

                def on_created(self, event: object) -> None:
                    """Forward created events to the debounce handler."""
                    self._h.on_created(event)

                def on_modified(self, event: object) -> None:
                    """Forward modified events to the debounce handler."""
                    self._h.on_modified(event)

                def on_deleted(self, event: object) -> None:
                    """Forward deleted events to the debounce handler."""
                    self._h.on_deleted(event)

                def on_moved(self, event: object) -> None:
                    """Forward moved events to the debounce handler."""
                    self._h.on_moved(event)

            bridge = _Bridge(handler)
            self._observer.schedule(  # type: ignore[attr-defined]
                bridge, str(source.path), recursive=source.recursive
            )
            self._handlers.append(handler)
            logger.info(f"WatcherService: watching {source.path} (recursive={source.recursive})")

        self._observer.start()  # type: ignore[attr-defined]
        logger.info("WatcherService: started")

    def stop(self) -> None:
        """Stop all watchers and flush pending debounce timers."""
        for handler in self._handlers:
            handler.cancel_all_timers()

        if self._observer is not None:
            self._observer.stop()  # type: ignore[attr-defined]
            self._observer.join()  # type: ignore[attr-defined]
            self._observer = None

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("WatcherService: stopped")

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until no debounce timers are pending (useful in tests).

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            ``True`` if idle before timeout, ``False`` if timed out.
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(h.pending_count() == 0 for h in self._handlers):
                return True
            time.sleep(0.05)
        return False
