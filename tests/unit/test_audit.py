"""Unit tests for AuditLogger."""

from __future__ import annotations

import json
from pathlib import Path

from memorymesh.observability.audit import AuditLogger


def test_log_query_creates_file(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    al.log_query(tool="search_memory", query="hello", n_results=3, latency_ms=12.5)
    assert log_path.exists()


def test_log_query_record_shape(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    al.log_query(tool="index_now", query="test query", n_results=0, latency_ms=5.0)
    record = json.loads(log_path.read_text().strip())
    assert record["tool"] == "index_now"
    assert "query_hash" in record
    assert "ts" in record
    assert record["n_results"] == 0
    assert "client_id" not in record


def test_log_query_includes_client_id(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    al.log_query(tool="search_memory", query="q", n_results=1, latency_ms=1.0, client_id="abc")
    record = json.loads(log_path.read_text().strip())
    assert record["client_id"] == "abc"


def test_multiple_writes_append(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    al.log_query(tool="t", query="a", n_results=1, latency_ms=1.0)
    al.log_query(tool="t", query="b", n_results=2, latency_ms=2.0)
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_query_text_never_stored(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    secret = "my secret query text"
    al.log_query(tool="t", query=secret, n_results=0, latency_ms=0.0)
    raw = log_path.read_text()
    assert secret not in raw


def test_creates_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dirs" / "audit.jsonl"
    al = AuditLogger(log_path)
    al.log_query(tool="t", query="x", n_results=0, latency_ms=0.0)
    assert log_path.exists()
