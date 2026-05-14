"""Unit tests for _DebounceHandler (watcher internals)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

from memorymesh.core.models import MemoryMeshConfig, SourceConfig
from memorymesh.indexer.watcher import WatcherService, _DebounceHandler


def _make_handler(
    tmp_path: Path,
    executor: ThreadPoolExecutor,
    debounce_ms: int = 50,
) -> tuple[_DebounceHandler, MagicMock]:
    indexer = MagicMock()
    indexer.index_file.return_value = MagicMock(status="indexed")
    src = SourceConfig(name="test", path=tmp_path)
    handler = _DebounceHandler(
        indexer=indexer,
        source_config=src,
        debounce_ms=debounce_ms,
        executor=executor,
        ignore_patterns=[],
    )
    return handler, indexer


def _fake_event(path: str, is_directory: bool = False) -> MagicMock:
    ev = MagicMock()
    ev.src_path = path
    ev.is_directory = is_directory
    return ev


class TestDebounceHandler:
    def test_directory_events_ignored(self, tmp_path: Path) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            handler, indexer = _make_handler(tmp_path, ex)
            ev = _fake_event(str(tmp_path / "dir"), is_directory=True)
            handler.on_created(ev)
            time.sleep(0.15)
            handler.cancel_all_timers()
        indexer.index_file.assert_not_called()

    def test_cancel_all_timers_clears_pending(self, tmp_path: Path) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            handler, _ = _make_handler(tmp_path, ex, debounce_ms=5000)
            f = tmp_path / "x.txt"
            f.write_text("hi")
            ev = _fake_event(str(f))
            handler.on_modified(ev)
            assert handler.pending_count() == 1
            handler.cancel_all_timers()
            assert handler.pending_count() == 0

    def test_on_moved_schedules_delete_and_create(self, tmp_path: Path) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            handler, indexer = _make_handler(tmp_path, ex, debounce_ms=10)
            src = tmp_path / "old.txt"
            dst = tmp_path / "new.txt"
            dst.write_text("content")
            ev = MagicMock()
            ev.src_path = str(src)
            ev.dest_path = str(dst)
            handler.on_moved(ev)
            time.sleep(0.15)
            ex.shutdown(wait=True)
        # delete was called for src, index for dst
        indexer.delete_file.assert_called_once()
        indexer.index_file.assert_called_once()

    def test_on_deleted_triggers_delete(self, tmp_path: Path) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            handler, indexer = _make_handler(tmp_path, ex, debounce_ms=10)
            f = tmp_path / "gone.txt"
            ev = _fake_event(str(f))
            handler.on_deleted(ev)
            time.sleep(0.15)
            ex.shutdown(wait=True)
        indexer.delete_file.assert_called_once()

    def test_pending_count_reflects_state(self, tmp_path: Path) -> None:
        with ThreadPoolExecutor(max_workers=1) as ex:
            handler, _ = _make_handler(tmp_path, ex, debounce_ms=5000)
            assert handler.pending_count() == 0
            f = tmp_path / "a.txt"
            f.write_text("x")
            handler.on_created(_fake_event(str(f)))
            assert handler.pending_count() == 1
            handler.cancel_all_timers()


class TestWatcherService:
    def test_init(self, tmp_path: Path) -> None:
        cfg = MemoryMeshConfig()
        indexer = MagicMock()
        svc = WatcherService(config=cfg, indexer=indexer)
        assert svc._observer is None
        assert svc._handlers == []

    def test_stop_on_unstarted_service(self, tmp_path: Path) -> None:
        cfg = MemoryMeshConfig()
        indexer = MagicMock()
        svc = WatcherService(config=cfg, indexer=indexer)
        svc.stop()  # must not raise

    def test_wait_idle_returns_true_when_no_handlers(self, tmp_path: Path) -> None:
        cfg = MemoryMeshConfig()
        indexer = MagicMock()
        svc = WatcherService(config=cfg, indexer=indexer)
        assert svc.wait_idle(timeout=0.1) is True
